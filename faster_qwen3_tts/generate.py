#!/usr/bin/env python3
"""
Non-streaming generation loop using CUDA graphs for both predictor and talker.
"""
import time
from typing import Optional, Tuple

import torch

from .predictor_graph import PredictorGraph
from .sampling import apply_repetition_penalty, build_suppress_mask, sample_logits
from .talker_graph import TalkerGraph

_EOS_TRACKER_CACHE: dict = {}


def get_eos_tracker(device) -> tuple:
    """Pinned host buffer + events for deferred EOS detection.

    Cached per device: pinned allocation costs several ms, which would land on
    every call's TTFA. Generation is single-stream so sharing scratch is safe.
    """
    key = str(device)
    tracker = _EOS_TRACKER_CACHE.get(key)
    if tracker is None:
        tracker = (
            torch.zeros(2, dtype=torch.long, pin_memory=True),
            (torch.cuda.Event(), torch.cuda.Event()),
        )
        _EOS_TRACKER_CACHE[key] = tracker
    return tracker


@torch.inference_mode()
def fast_generate(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    predictor_graph: PredictorGraph,
    talker_graph: TalkerGraph,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    subtalker_dosample: Optional[bool] = None,
    subtalker_top_k: Optional[int] = None,
    subtalker_top_p: Optional[float] = None,
    subtalker_temperature: Optional[float] = None,
    parity_mode: bool = False,
) -> Tuple[Optional[torch.Tensor], dict]:
    """
    Fast autoregressive generation with CUDA-graphed predictor and talker.
    """
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = talker_input_embeds.device
    
    suppress_mask = build_suppress_mask(vocab_size, eos_id, device)
    suppress_start = max(0, vocab_size - 1024)
    eos_suppress_ids = torch.tensor([eos_id], dtype=torch.long, device=device)

    if parity_mode:
        suppress_tokens = [i for i in range(suppress_start, vocab_size) if i != eos_id]
        t_start = time.time()
        talker_result = talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_id,
            suppress_tokens=suppress_tokens,
            subtalker_dosample=subtalker_dosample if subtalker_dosample is not None else do_sample,
            subtalker_top_k=subtalker_top_k if subtalker_top_k is not None else top_k,
            subtalker_top_p=subtalker_top_p if subtalker_top_p is not None else top_p,
            subtalker_temperature=subtalker_temperature if subtalker_temperature is not None else temperature,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        talker_codes = torch.stack(
            [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None],
            dim=1,
        )
        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == eos_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])
        talker_codes_list = [talker_codes[i, :length, :] for i, length in enumerate(effective_lengths)]

        torch.cuda.synchronize()
        total_time = time.time() - t_start
        steps = int(talker_codes_list[0].shape[0]) if talker_codes_list else 0
        timing = {
            'prefill_ms': 0.0,
            'decode_s': total_time,
            'steps': steps,
            'ms_per_step': (total_time / steps * 1000) if steps > 0 else 0.0,
            'steps_per_s': (steps / total_time) if total_time > 0 else 0.0,
        }
        return talker_codes_list[0] if talker_codes_list else None, timing
    
    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    predictor_codec_embeds = predictor.get_input_embeddings()
    
    # === PREFILL (still uses HF forward for variable-length prefill) ===
    t_start = time.time()
    
    out = talker.forward(
        inputs_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=None,
    )
    
    talker_past_kv = out.past_key_values
    past_hidden = out.past_hidden
    gen_step = out.generation_step
    
    logits = out.logits[:, -1, :]
    suppress_eos = min_new_tokens > 0
    token = sample_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        suppress_mask=suppress_mask,
        suppress_tokens=eos_suppress_ids if suppress_eos else None,
    )
    
    # Copy prefill KV cache into talker graph's static cache
    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    # Sync padding mask + rope deltas for decode parity
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(attention_mask, rope_deltas)
    
    # Deferred EOS detection: token values are copied to a pinned host buffer
    # asynchronously and checked one iteration late, so the CPU never blocks on
    # the GPU mid-step (a per-step .item() would drain the launch queue every
    # iteration). Slot k%2 holds the token consumed by iteration k. At the top
    # of iteration k we first check iteration k-1's token (its copy was enqueued
    # two iterations of GPU work ago and is almost always already complete),
    # then opportunistically check token k itself if its copy already landed —
    # the common case on hosts slower than the GPU, where this stops exactly at
    # EOS. Otherwise EOS costs one extra (overshoot) iteration that is popped.
    token_cpu, token_events = get_eos_tracker(device)
    token_cpu[0:1].copy_(token, non_blocking=True)
    token_events[0].record()

    torch.cuda.synchronize()
    t_prefill = time.time() - t_start

    # === DECODE LOOP ===
    t_decode_start = time.time()
    all_codec_ids = []
    eos_found = False

    for step_idx in range(max_new_tokens):
        if step_idx > 0:
            prev_slot = (step_idx - 1) % 2
            token_events[prev_slot].synchronize()
            if int(token_cpu[prev_slot]) == eos_id:
                # Previous iteration consumed EOS — drop its output and stop.
                all_codec_ids.pop()
                eos_found = True
                break
        cur_slot = step_idx % 2
        if token_events[cur_slot].query() and int(token_cpu[cur_slot]) == eos_id:
            # This iteration's own token is already visible and is EOS — stop
            # before doing any work (no overshoot).
            eos_found = True
            break

        # --- CUDA-Graphed Code Predictor ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))  # [1, 1, H]
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)  # [1, 2, H]
        codebook_token_ids = predictor_graph.run(pred_input)  # [15] long tensor
        
        # Build full codec: [first_cb, cb1, ..., cb15]
        all_cb = torch.cat([token.view(1), codebook_token_ids])  # [16]
        all_codec_ids.append(all_cb.detach())
        
        # --- Build input embedding for talker ---
        codec_hiddens = [last_id_hidden]
        for i in range(num_code_groups - 1):
            codec_hiddens.append(predictor_codec_embeds[i](codebook_token_ids[i].unsqueeze(0).unsqueeze(0)))
        inputs_embeds = torch.cat(codec_hiddens, dim=1).sum(1, keepdim=True)
        
        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed
        
        # --- CUDA-Graphed Talker decode step ---
        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            # Stop if we exceed max_seq_len
            break
        
        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)
        # hidden_states is the static output buffer - use it immediately
        
        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)
        
        if repetition_penalty != 1.0 and len(all_codec_ids) > 0:
            history = torch.stack([c[0] for c in all_codec_ids])
            logits = apply_repetition_penalty(logits, history, repetition_penalty)

        suppress_eos = len(all_codec_ids) < min_new_tokens
        token = sample_logits(
            logits.squeeze(0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=eos_suppress_ids if suppress_eos else None,
        )
        next_slot = (step_idx + 1) % 2
        token_cpu[next_slot:next_slot + 1].copy_(token, non_blocking=True)
        token_events[next_slot].record()
        past_hidden = hidden_states[:, -1:, :].clone()  # clone since it's the static buffer
        gen_step += 1

    torch.cuda.synchronize()
    # The loop can exit (budget/seq-len) with the newest entry still unchecked;
    # its token lives in slot (n-1)%2 since slots are keyed by iteration index.
    if not eos_found and all_codec_ids:
        last_slot = (len(all_codec_ids) - 1) % 2
        if int(token_cpu[last_slot]) == eos_id:
            all_codec_ids.pop()
    t_decode = time.time() - t_decode_start

    n_steps = len(all_codec_ids)
    timing = {
        'prefill_ms': t_prefill * 1000,
        'decode_s': t_decode,
        'steps': n_steps,
        'ms_per_step': (t_decode / n_steps * 1000) if n_steps > 0 else 0,
        'steps_per_s': (n_steps / t_decode) if t_decode > 0 else 0,
    }
    
    if all_codec_ids:
        return torch.stack(all_codec_ids), timing
    return None, timing
