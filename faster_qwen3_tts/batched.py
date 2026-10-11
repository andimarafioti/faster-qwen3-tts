"""
Batched CUDA-graph decoding for voice cloning.

The predictor and talker graphs in this package decode one sequence at a time.
Upstream qwen-tts batches, but every decode step is bound by kernel-launch
overhead: a step costs about the same for 2 rows as for 32. This module captures
both graphs with a batch dimension, so B sequences share each replay.

The engine stands in for `talker.generate` of an upstream `Qwen3TTSModel`.
Prompt building, generation-config defaults, EOS trimming, the ICL ref-code
prepend and audio decoding all stay upstream code:

    engine = BatchedEngine(qwen3_tts_model)
    with engine:
        wavs, sr = qwen3_tts_model.generate_voice_clone(text=[...], ...)

`FasterQwen3TTS.generate_voice_clone(text=[...])` does this for you.
"""
from collections import OrderedDict
from types import SimpleNamespace

import torch
from transformers import StaticCache

from .sampling import sample_logits

DEFAULT_MAX_SEQ_LEN = 2048  # same default as FasterQwen3TTS; prompt + generated frames per row
BUCKETS = (1, 2, 4, 8, 16, 32, 48, 64)  # graph batch sizes; a batch is padded up to the next one
EOS_CHECK_EVERY = 8  # decode steps between host syncs that test whether every row has finished
MIN_GENERATE = 512  # minimum free cache positions to select graphs; otherwise use upstream generation


def _init_cache(cache: StaticCache, config, batch: int, dtype, device) -> None:
    """StaticCache layers take their batch size from the first key tensor; set it before capture."""
    heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dummy = torch.zeros(batch, heads, 1, head_dim, dtype=dtype, device=device)
    for layer in cache.layers:
        if not layer.is_initialized:
            layer.lazy_initialization(dummy, dummy)


def _rewind(cache: StaticCache) -> None:
    """Restart writing at position 0. A static layer ignores `cache_position` on update and appends at its
    own `cumulative_length`. Stale keys need no zeroing: the attention mask hides them."""
    for layer in cache.layers:
        layer.cumulative_length.zero_()


@torch.inference_mode()
def _capture(fn, device, warmup: int = 3) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(device):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


class BatchedPredictorGraph:
    """The 15-step code predictor loop for B rows, sampling included, as one CUDA graph.
    The cache rewind is part of the graph, so a replay needs no extra launches."""

    def __init__(self, code_predictor, talker_hidden: int, num_code_groups: int, batch: int,
                 sampling: dict, device, dtype):
        self.proj = code_predictor.small_to_mtp_projection
        self.model = code_predictor.model
        self.heads = code_predictor.lm_head
        self.embeds = code_predictor.model.codec_embedding
        self.sampling = sampling
        self.count = num_code_groups - 1
        config = self.model.config
        size = 2 + self.count
        self.cache = StaticCache(config=config, max_cache_len=size)
        _init_cache(self.cache, config, batch, dtype, device)
        self.prefill_pos = torch.arange(2, device=device)
        self.decode_pos = [torch.tensor([2 + i], device=device) for i in range(self.count - 1)]
        keys = torch.arange(size, device=device)
        sliding = "sliding_attention" in getattr(config, "layer_types", [])
        low = torch.finfo(dtype).min

        def mask(pos):
            allowed = {"full_attention": keys[None] <= pos[:, None]}
            if sliding:
                window = keys[None] > pos[:, None] - config.sliding_window
                allowed["sliding_attention"] = allowed["full_attention"] & window
            masks = {kind: torch.where(ok, 0.0, low).to(dtype)[None, None] for kind, ok in allowed.items()}
            return {kind: m.expand(batch, -1, -1, -1).contiguous() for kind, m in masks.items()}

        self.prefill_mask = mask(self.prefill_pos)
        self.decode_masks = [mask(pos) for pos in self.decode_pos]
        self.input_buf = torch.zeros(batch, 2, talker_hidden, dtype=dtype, device=device)
        self.output = torch.zeros(batch, self.count, dtype=torch.long, device=device)
        self.graph = _capture(self._loop, device)

    def _sample(self, hidden, index):
        return sample_logits(self.heads[index](hidden[:, -1]).float(), **self.sampling)

    def _loop(self):
        _rewind(self.cache)
        out = self.model(inputs_embeds=self.proj(self.input_buf), attention_mask=self.prefill_mask,
                         past_key_values=self.cache, cache_position=self.prefill_pos, use_cache=True)
        token = self._sample(out.last_hidden_state, 0)
        self.output[:, 0] = token
        for i in range(1, self.count):
            out = self.model(inputs_embeds=self.proj(self.embeds[i - 1](token[:, None])),
                             attention_mask=self.decode_masks[i - 1], past_key_values=self.cache,
                             cache_position=self.decode_pos[i - 1], use_cache=True)
            token = self._sample(out.last_hidden_state, i)
            self.output[:, i] = token

    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """pred_input [B, 2, H] (past hidden, first-codebook embed) -> codebooks 1..15 [B, 15]."""
        self.input_buf.copy_(pred_input)
        self.graph.replay()
        return self.output.clone()


class BatchedTalkerGraph:
    """One talker decode step for B left-padded rows as a CUDA graph. Instead of a per-position mask table
    (max_seq x B x max_seq), the attention mask and the rope positions are computed inside the graph from
    `cache_position`, a per-row `valid` key mask (False on left padding) and the per-row rope deltas."""

    def __init__(self, talker_model, config, batch: int, max_seq_len: int, device, dtype):
        self.model = talker_model
        self.max_seq_len = max_seq_len
        self.cache = StaticCache(config=config, max_cache_len=max_seq_len)
        _init_cache(self.cache, config, batch, dtype, device)
        self.keys = torch.arange(max_seq_len, device=device)
        self.valid = torch.ones(batch, max_seq_len, dtype=torch.bool, device=device)
        self.rope_deltas = torch.zeros(batch, 1, dtype=torch.float32, device=device)
        self.cache_position = torch.ones(1, dtype=torch.long, device=device)
        self.zero = torch.zeros((), dtype=dtype, device=device)
        self.low = torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device)
        self.input_buf = torch.zeros(batch, 1, config.hidden_size, dtype=dtype, device=device)
        self.output_buf = torch.zeros_like(self.input_buf)
        self.graph = _capture(self._step, device)

    def _step(self):
        allowed = (self.keys <= self.cache_position) & self.valid
        mask = torch.where(allowed, self.zero, self.low)[:, None, None, :]
        positions = (self.cache_position.to(torch.float32) + self.rope_deltas)[None].expand(3, -1, -1)
        out = self.model(inputs_embeds=self.input_buf, attention_mask=mask, past_key_values=self.cache,
                         cache_position=self.cache_position, position_ids=positions, use_cache=True)
        self.output_buf.copy_(out.last_hidden_state)

    def prefill(self, past_key_values, attention_mask: torch.Tensor, rope_deltas: torch.Tensor) -> int:
        """Copy the prefill DynamicCache in and set the padding mask and rope deltas. Each later `run`
        appends one position, so calls use positions length, length + 1, ... in order."""
        length = attention_mask.shape[1]
        positions = torch.arange(length, device=self.keys.device)
        _rewind(self.cache)
        for index, layer in enumerate(past_key_values.layers):
            self.cache.update(layer.keys, layer.values, index, {"cache_position": positions})
        pads = (attention_mask == 0).sum(dim=-1, keepdim=True)
        self.valid.copy_(self.keys[None] >= pads)
        self.rope_deltas.copy_(rope_deltas.reshape(-1, 1).to(torch.float32))
        return length

    def run(self, embeds: torch.Tensor, position: int) -> torch.Tensor:
        """embeds [B, 1, H] at absolute `position` -> hidden [B, 1, H] (a static buffer: use it now)."""
        self.input_buf.copy_(embeds)
        self.cache_position.fill_(position)
        self.graph.replay()
        return self.output_buf


def _bucket(size: int) -> int:
    return next((b for b in BUCKETS if b >= size), -(-size // 16) * 16)


def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
    """Pad the batch dimension to `rows` with copies of row 0."""
    extra = rows - tensor.shape[0]
    return torch.cat([tensor, tensor[:1].expand(extra, *tensor.shape[1:])]) if extra else tensor


class BatchedEngine:
    """Owns the captured graphs and stands in for `talker.generate` of an upstream `Qwen3TTSModel`.

    Use it as a context manager (or call `install()` to keep it in place). Each graph batch size holds its
    own static KV cache (B x max_seq_len x ~115 KB for the 1.7B talker), so at most `max_resident` sizes
    stay captured, least recently used first out. A smaller batch reuses a resident larger size: padding
    rows cost little once launch overhead is gone."""

    def __init__(self, model, max_seq_len: int = DEFAULT_MAX_SEQ_LEN, max_resident: int = 2,
                 min_new_tokens: int | None = None):
        self.talker = model.model.talker
        self.upstream = type(self.talker).generate.__get__(self.talker)
        self.config = model.model.config.talker_config
        self.max_seq_len = max_seq_len
        self.max_resident = max_resident
        # The upstream input builder fixes this at 2. Override it at the talker
        # hook so public voice-cloning calls can still select their minimum.
        self.min_new_tokens = min_new_tokens
        self.device = self.talker.device
        self.dtype = self.talker.dtype
        self.resident = OrderedDict()  # batch -> {"talker": graph, "predictors": {sampling key: graph}}
        self.last = {}  # timing and step counts of the last call

    def install(self) -> None:
        self.talker.generate = self.generate

    def uninstall(self) -> None:
        self.talker.__dict__.pop("generate", None)

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc):
        self.uninstall()

    def clear(self) -> None:
        """Drop all captured graphs and their static caches."""
        self.resident.clear()
        torch.cuda.empty_cache()

    def graphs(self, rows: int, sampling: dict) -> tuple:
        batch = next((b for b in sorted(self.resident) if b >= rows), None) or _bucket(rows)
        if batch not in self.resident:
            while len(self.resident) >= self.max_resident:
                self.resident.popitem(last=False)
                torch.cuda.empty_cache()
            talker = BatchedTalkerGraph(self.talker.model, self.config, batch, self.max_seq_len, self.device,
                                        self.dtype)
            self.resident[batch] = {"talker": talker, "predictors": {}}
        self.resident.move_to_end(batch)
        entry = self.resident[batch]
        key = tuple(sorted(sampling.items()))
        if key not in entry["predictors"]:
            entry["predictors"][key] = BatchedPredictorGraph(
                self.talker.code_predictor, self.config.hidden_size, self.config.num_code_groups, batch,
                sampling, self.device, self.dtype)
        return batch, entry["predictors"][key], entry["talker"]

    def generate(self, **kwargs):
        """Same keyword inputs as upstream `talker.generate` (as upstream calls it). A prompt that leaves
        fewer than MIN_GENERATE positions in the static cache runs through upstream instead."""
        if self.min_new_tokens is not None:
            kwargs["min_new_tokens"] = self.min_new_tokens
        if kwargs["inputs_embeds"].shape[1] > self.max_seq_len - MIN_GENERATE:
            return self.upstream(**kwargs)
        return self._generate(**kwargs)

    @torch.inference_mode()
    def _generate(self, inputs_embeds, attention_mask, trailing_text_hidden, tts_pad_embed,
                  max_new_tokens=4096, min_new_tokens=2, do_sample=True, top_k=50, top_p=1.0,
                  temperature=0.9, subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0,
                  subtalker_temperature=0.9, eos_token_id=None, repetition_penalty=1.05, suppress_tokens=(),
                  **_):
        """Returns the `hidden_states` layout upstream reads: one `((hidden,), codes [B, 16])` per step,
        the prefill entry with codes None. Finished rows keep emitting EOS; upstream trims at the first."""
        talker = self.talker
        rows = inputs_embeds.shape[0]
        subtalker = dict(do_sample=subtalker_dosample, top_k=subtalker_top_k, top_p=subtalker_top_p,
                         temperature=subtalker_temperature)
        batch, predictor, graph = self.graphs(rows, subtalker)
        inputs_embeds = _pad_rows(inputs_embeds, batch)  # padding rows are dropped before returning
        attention_mask = _pad_rows(attention_mask, batch)
        trailing_text_hidden = _pad_rows(trailing_text_hidden, batch)
        eos = eos_token_id if eos_token_id is not None else self.config.codec_eos_token_id

        torch.cuda.synchronize()
        start, prefilled, done = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        start.record()
        talker.rope_deltas = None
        out = talker.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True,
                             return_dict=True, trailing_text_hidden=trailing_text_hidden,
                             tts_pad_embed=tts_pad_embed, generation_step=None, past_hidden=None,
                             past_key_values=None)
        length = graph.prefill(out.past_key_values, attention_mask, talker.rope_deltas)
        prefilled.record()

        vocab = self.config.vocab_size
        suppress = torch.zeros(vocab, dtype=torch.bool, device=self.device)
        suppress[list(suppress_tokens)] = True
        suppress_eos = suppress.clone()
        suppress_eos[eos] = True
        sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p, do_sample=do_sample)

        # HF counts prefill as the first generation forward. That forward
        # samples a first-codebook token but produces no complete codec frame;
        # only the following max_new_tokens - 1 forwards produce frames.
        limit = min(max_new_tokens - 1, self.max_seq_len - length)
        codes = torch.zeros(batch, limit, self.config.num_code_groups, dtype=torch.long, device=self.device)
        seen = torch.zeros(batch, vocab, dtype=torch.bool, device=self.device)
        finished = torch.zeros(batch, dtype=torch.bool, device=self.device)
        eos_fill = torch.full((batch,), eos, dtype=torch.long, device=self.device)
        token = sample_logits(out.logits[:, -1, :].float(),
                              suppress_mask=suppress_eos if min_new_tokens > 0 else suppress, **sampling)
        finished |= token == eos
        past_hidden = out.past_hidden
        text_steps = trailing_text_hidden.shape[1]
        codec_embed = talker.get_input_embeddings()
        predictor_embeds = talker.code_predictor.get_input_embeddings()
        steps = 0
        for step in range(limit):
            if step and step % EOS_CHECK_EVERY == 0 and bool(finished.all()):
                break
            last = codec_embed(token[:, None])
            predicted = predictor.run(torch.cat((past_hidden, last), dim=1))
            codes[:, step, 0] = token
            codes[:, step, 1:] = predicted
            hiddens = [last] + [embed(predicted[:, i:i + 1]) for i, embed in enumerate(predictor_embeds)]
            embeds = torch.cat(hiddens, dim=1).sum(1, keepdim=True)
            embeds = embeds + (trailing_text_hidden[:, step:step + 1] if step < text_steps else tts_pad_embed)
            steps = step + 1
            if steps >= limit:
                break
            hidden = graph.run(embeds, length + step)
            logits = talker.codec_head(hidden[:, -1]).float()
            # HF repetition penalty on every generated first-codebook token, as a [B, vocab] mask
            seen.scatter_(1, token[:, None], True)
            if repetition_penalty != 1.0:
                penalized = torch.where(logits > 0, logits / repetition_penalty, logits * repetition_penalty)
                logits = torch.where(seen, penalized, logits)
            token = sample_logits(logits, suppress_mask=suppress_eos if steps < min_new_tokens else suppress,
                                  **sampling)
            token = torch.where(finished, eos_fill, token)
            finished |= token == eos
            past_hidden = hidden[:, -1:].clone()
        done.record()
        torch.cuda.synchronize()
        self.last = {"rows": rows, "batch": batch, "prefill_len": length, "steps": steps,
                     "prefill_ms": start.elapsed_time(prefilled), "decode_ms": prefilled.elapsed_time(done)}

        codes = codes[:rows, :steps]
        dummy = torch.zeros(rows, 1, 1, device=self.device)
        per_step = [((dummy,), codes[:, t]) for t in range(steps)]
        return SimpleNamespace(hidden_states=[((dummy,), None)] + per_step)
