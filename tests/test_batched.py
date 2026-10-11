"""Batched CUDA-graph voice clone (faster_qwen3_tts/batched.py).

Parity tests use greedy decoding, so upstream and batched runs are comparable token for token. Compared
on the same batch, the codes must be identical: the graphs change how the decode loop is launched, not
what it computes. (A row alone vs inside a batch differs after a few frames in upstream too: bf16
batch-size numerics.)
"""
import contextlib
import os
from types import SimpleNamespace

import pytest
import torch

from faster_qwen3_tts.batched import BatchedEngine, BatchedPredictorGraph, _bucket, _pad_rows

MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
REF_AUDIO = "ref_audio.wav"
REF_TEXT = (
    "I'm confused why some people have super short timelines, yet at the same time are bullish on scaling up "
    "reinforcement learning atop LLMs. If we're actually close to a human-like learner, then this whole approach "
    "of training on verifiable outcomes is doomed."
)
# different lengths, so the batch has left padding of several sizes
TEXTS = [
    "Hi.",
    "The quick brown fox jumps over the lazy dog.",
    "It rained all night, and the puddles on the street reflected the dim yellow lamps.",
    "She was quiet for a long time before she said: I did not want to tell you this, but keeping it "
    "a secret any longer would not help anyone, so you should decide for yourself.",
]
GREEDY = dict(do_sample=False, subtalker_dosample=False, max_new_tokens=400)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required.")


def test_bucket_sizes():
    assert [_bucket(n) for n in (1, 3, 5, 32, 33, 64, 65, 70)] == [1, 4, 8, 32, 48, 64, 80, 80]


def test_pad_rows_repeats_row_zero():
    rows = torch.arange(6).view(2, 3)
    padded = _pad_rows(rows, 4)
    assert padded.shape == (4, 3)
    assert torch.equal(padded[2:], rows[:1].expand(2, 3))
    assert _pad_rows(rows, 2) is rows


def test_predictor_top_k_uses_float32(monkeypatch):
    # BF16 temperature scaling rounds these distinct scores to the same value.
    # In FP32, top_k=1 must retain only the second token, as HF does.
    graph = BatchedPredictorGraph.__new__(BatchedPredictorGraph)
    graph.heads = [lambda hidden: torch.tensor([[1.8125, 1.8203125]], dtype=torch.bfloat16)]
    graph.sampling = dict(do_sample=True, top_k=1, top_p=1.0, temperature=0.9)
    probabilities = []

    def sample(probs, count):
        probabilities.append(probs)
        return probs.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", sample)
    graph._sample(torch.zeros(1, 1, 2), 0)
    assert probabilities[0].dtype == torch.float32
    assert torch.equal(probabilities[0], torch.tensor([[0.0, 1.0]]))


@pytest.mark.parametrize("max_seq_len,prefill_len,fallback", [
    (2048, 1536, False), (2048, 1537, True),
    (1024, 512, False), (1024, 513, True),
])
def test_minimum_override_reaches_graph_and_fallback(max_seq_len, prefill_len, fallback):
    calls = []

    class Talker:
        device = "cpu"
        dtype = torch.float32

        def generate(self, **kwargs):
            calls.append((True, kwargs))

    base = SimpleNamespace(model=SimpleNamespace(
        talker=Talker(), config=SimpleNamespace(talker_config=SimpleNamespace())
    ))
    engine = BatchedEngine(base, max_seq_len=max_seq_len, min_new_tokens=50)
    engine._generate = lambda **kwargs: calls.append((False, kwargs))
    engine.generate(inputs_embeds=torch.zeros(1, prefill_len, 4), min_new_tokens=2)
    assert calls[0][0] == fallback
    assert calls[0][1]["min_new_tokens"] == 50


@pytest.mark.parametrize("max_seq_len", [2048, 1024])
def test_public_batch_forwards_min_new_tokens_and_cache_length(monkeypatch, max_seq_len):
    from faster_qwen3_tts import FasterQwen3TTS

    class Base:
        def generate_voice_clone(self, **kwargs):
            self.kwargs = kwargs
            return [], 24000

    class Engine:
        def __init__(self, model, max_seq_len):
            self.max_seq_len = max_seq_len

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    base = Base()
    monkeypatch.setattr("faster_qwen3_tts.batched.BatchedEngine", Engine)
    model = FasterQwen3TTS(base, None, None, device="cpu", max_seq_len=max_seq_len)
    model.generate_voice_clone(text=["hello"], language="English", voice_clone_prompt=[object()],
                               min_new_tokens=50)
    assert base.kwargs["min_new_tokens"] == 50
    assert model._batched.min_new_tokens == 50
    assert model._batched.max_seq_len == max_seq_len


@pytest.fixture(scope="module")
def fast():
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(MODEL_ID, device="cuda", dtype=torch.bfloat16, attn_implementation="sdpa")
    yield model
    del model
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def prompt(fast):
    return fast.model.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)


def _codes(base, texts, items, non_streaming_mode, engine=None, **generation_kwargs):
    """Talker codes from upstream `Qwen3TTSForConditionalGeneration.generate`, optionally through the engine."""
    vcp = base._prompt_items_to_voice_clone_prompt(items)
    input_ids = base._tokenize_texts([base._build_assistant_text(t) for t in texts])
    ref_ids = [base._tokenize_texts([base._build_ref_text(it.ref_text)])[0] for it in items]
    with engine if engine is not None else contextlib.nullcontext():
        codes, _ = base.model.generate(
            input_ids=input_ids,
            ref_ids=ref_ids,
            voice_clone_prompt=vcp,
            languages=["English"] * len(texts),
            non_streaming_mode=non_streaming_mode,
            **base._merge_generate_kwargs(**(GREEDY | generation_kwargs)),
        )
    return [c.cpu() for c in codes]


@cuda
class TestBatchedVoiceClone:
    def test_predictor_graph_matches_eager(self, fast):
        talker = fast.model.model.talker
        config = fast.model.model.config.talker_config
        sampling = dict(do_sample=False, top_k=50, top_p=1.0, temperature=0.9)
        graph = BatchedPredictorGraph(talker.code_predictor, config.hidden_size, config.num_code_groups, 4,
                                      sampling, talker.device, talker.dtype)
        with torch.inference_mode():
            for seed in range(10):
                torch.manual_seed(seed)
                x = torch.randn(4, 2, config.hidden_size, device=talker.device, dtype=talker.dtype)
                eager = talker.code_predictor.generate(inputs_embeds=x, max_new_tokens=config.num_code_groups - 1,
                                                       do_sample=False)
                assert torch.equal(graph.run(x), eager)

    @pytest.mark.parametrize("non_streaming_mode", [False, True])
    @pytest.mark.parametrize("batch", [4, 1])
    def test_codes_match_upstream(self, fast, prompt, batch, non_streaming_mode):
        texts = TEXTS[:batch]
        expected = _codes(fast.model, texts, prompt * batch, non_streaming_mode)
        engine = BatchedEngine(fast.model)
        got = _codes(fast.model, texts, prompt * batch, non_streaming_mode, engine)
        assert engine.last["rows"] == batch
        for want, have in zip(expected, got):
            assert torch.equal(want, have)

    def test_long_prompt_falls_back_to_upstream(self, fast, prompt):
        engine = BatchedEngine(fast.model, max_seq_len=600)  # the ICL prompt leaves < 512 positions
        expected = _codes(fast.model, TEXTS[:2], prompt * 2, False)
        got = _codes(fast.model, TEXTS[:2], prompt * 2, False, engine)
        assert engine.last == {}
        assert all(torch.equal(want, have) for want, have in zip(expected, got))

    @pytest.mark.parametrize("max_new_tokens", [2, 8])
    def test_budget_exhaustion_matches_upstream(self, fast, prompt, max_new_tokens):
        texts = TEXTS[1:3]
        expected = _codes(fast.model, texts, prompt * 2, False, max_new_tokens=max_new_tokens)
        engine = BatchedEngine(fast.model)
        got = _codes(fast.model, texts, prompt * 2, False, engine, max_new_tokens=max_new_tokens)
        assert all(len(c) == max_new_tokens - 1 for c in expected)
        assert all(torch.equal(want, have) for want, have in zip(expected, got))

    @pytest.mark.parametrize("ref_text", [REF_TEXT, [REF_TEXT, REF_TEXT]])
    def test_icl_dictionary_prompt(self, fast, prompt, ref_text):
        vcp = fast.model._prompt_items_to_voice_clone_prompt(prompt * 2)
        wavs, sr = fast.generate_voice_clone(text=TEXTS[:2], language="English",
                                             voice_clone_prompt=vcp, ref_text=ref_text, max_new_tokens=8)
        assert sr == fast.sample_rate
        assert len(wavs) == 2 and all(len(w) > 0 for w in wavs)

    def test_compact_xvector_dictionary_prompt(self, fast, prompt):
        vcp = {"ref_spk_embedding": [prompt[0].ref_spk_embedding] * 2}
        wavs, sr = fast.generate_voice_clone(text=TEXTS[:2], language="English",
                                             voice_clone_prompt=vcp, max_new_tokens=8)
        assert sr == fast.sample_rate
        assert len(wavs) == 2 and all(len(w) > 0 for w in wavs)

    def test_minimum_suppresses_early_eos(self, fast, prompt):
        engine = BatchedEngine(fast.model, min_new_tokens=50)
        codes = _codes(fast.model, TEXTS[:1], prompt, False, engine, max_new_tokens=80)
        assert len(codes[0]) >= 50
        assert not (codes[0][:50, 0] == engine.config.codec_eos_token_id).any()

    def test_generate_voice_clone_accepts_a_list(self, fast):
        wavs, sr = fast.generate_voice_clone(text=TEXTS, language="English", ref_audio=REF_AUDIO,
                                             ref_text=REF_TEXT, max_new_tokens=400)
        assert sr == fast.sample_rate
        assert len(wavs) == len(TEXTS) and all(len(w) > 0 for w in wavs)
        assert "generate" not in fast.model.model.talker.__dict__  # upstream decode restored afterwards
