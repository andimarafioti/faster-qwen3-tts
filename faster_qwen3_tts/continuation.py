from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import torch
from transformers.cache_utils import DynamicCache

CONTINUATION_STATE_VERSION = 1
DECODER_CONTEXT_FRAMES = 25
CONTINUATION_WARNING_MIN_TOKENS = 64

ReturnContinuationMode = Literal["none", "delta", "full"]


def normalize_return_continuation_state(value: bool | str | None) -> ReturnContinuationMode:
    if value in (None, False):
        return "none"
    if value is True:
        return "delta"
    if value not in {"delta", "full"}:
        raise ValueError(
            "return_continuation_state must be one of False, True, 'delta', or 'full'"
        )
    return value


def model_signature(*, num_layers: int, max_seq_len: int, hidden_size: int) -> Dict[str, int]:
    return {
        "num_layers": int(num_layers),
        "max_seq_len": int(max_seq_len),
        "hidden_size": int(hidden_size),
    }


def build_continuation_state_status(*, seq_len: int, max_seq_len: int) -> Dict[str, Any]:
    usable_seq_len = max(int(max_seq_len) - 1, 0)
    remaining_tokens = max(usable_seq_len - int(seq_len), 0)
    # Warn once the session is down to roughly the larger of 64 tokens or 12.5%
    # of the cache. That gives callers room to reset before they hard-fail.
    warning_threshold_tokens = max(CONTINUATION_WARNING_MIN_TOKENS, usable_seq_len // 8)
    should_reset = remaining_tokens <= warning_threshold_tokens

    status: Dict[str, Any] = {
        "seq_len": int(seq_len),
        "max_seq_len": int(max_seq_len),
        "usable_seq_len": usable_seq_len,
        "remaining_tokens": remaining_tokens,
        "warning_threshold_tokens": warning_threshold_tokens,
        "should_reset": should_reset,
    }
    if remaining_tokens == 0:
        status["warning"] = (
            "Continuation state has no remaining token budget. Start a fresh session "
            "before the next turn."
        )
    elif should_reset:
        status["warning"] = (
            f"Continuation state is close to max_seq_len "
            f"({remaining_tokens} tokens remaining). Start a fresh session soon."
        )
    return status


def validate_full_continuation_state(
    state: Optional[Dict[str, Any]],
    *,
    mode: str,
    expected_signature: Dict[str, int],
) -> None:
    if state is None:
        return
    if state.get("version") != CONTINUATION_STATE_VERSION:
        raise ValueError(
            f"Unsupported continuation_state version: {state.get('version')!r}"
        )
    if state.get("state_kind") != "full":
        raise ValueError("continuation_state must be a full state. Apply the delta first.")
    if state.get("mode") != mode:
        raise ValueError(
            f"continuation_state mode mismatch: expected {mode!r}, got {state.get('mode')!r}"
        )
    if state.get("model_signature") != expected_signature:
        raise ValueError("continuation_state does not match the loaded model configuration")


def continuation_state_to_dynamic_cache(
    state: Dict[str, Any],
    *,
    config,
    device: str,
) -> DynamicCache:
    cache_entries = []
    for layer in state["cache"]:
        key = layer["key"].to(device)
        value = layer["value"].to(device)
        cache_entries.append((key, value))
    return DynamicCache(ddp_cache_data=cache_entries, config=config)


def continuation_state_first_token_history(
    state: Optional[Dict[str, Any]],
    *,
    device: str,
) -> list[torch.Tensor]:
    if state is None:
        return []
    history = state.get("first_codebook_history")
    if history is None or history.numel() == 0:
        return []
    return [tok.detach() for tok in history.to(device)]


def continuation_state_decoder_context(
    state: Optional[Dict[str, Any]],
    *,
    device: str,
) -> Optional[torch.Tensor]:
    if state is None:
        return None
    codes = state.get("decoder_context_codes")
    if codes is None or codes.numel() == 0:
        return None
    return codes.to(device)


def export_static_cache_slice(
    cache_source,
    *,
    start_pos: int,
    end_pos: int,
    device: str,
) -> list[Dict[str, torch.Tensor]]:
    layers = []
    for layer in cache_source.layers:
        layers.append(
            {
                "key": layer.keys[:, :, start_pos:end_pos, :].clone().to(device),
                "value": layer.values[:, :, start_pos:end_pos, :].clone().to(device),
            }
        )
    return layers


def build_continuation_state_delta(
    *,
    cache_source,
    base_seq_len: int,
    seq_len: int,
    rope_deltas: Optional[torch.Tensor],
    first_codebook_history_delta: list[torch.Tensor],
    codec_ids_delta: list[torch.Tensor],
    mode: str,
    non_streaming_mode: bool,
    model_signature_dict: Dict[str, int],
    device: str,
) -> Dict[str, Any]:
    rope = (
        torch.zeros(1, 1, dtype=torch.float32, device=device)
        if rope_deltas is None
        else rope_deltas.detach().clone().to(device=device, dtype=torch.float32)
    )
    if first_codebook_history_delta:
        first_tokens = torch.stack(first_codebook_history_delta).to(device=device)
    else:
        first_tokens = torch.empty(0, dtype=torch.long, device=device)
    if codec_ids_delta:
        codec_delta = torch.stack(codec_ids_delta).to(device=device)
    else:
        codec_delta = torch.empty(0, 0, dtype=torch.long, device=device)

    return {
        "version": CONTINUATION_STATE_VERSION,
        "state_kind": "delta",
        "mode": mode,
        "non_streaming_mode": bool(non_streaming_mode),
        "model_signature": dict(model_signature_dict),
        "base_seq_len": int(base_seq_len),
        "seq_len": int(seq_len),
        "added_seq_len": int(seq_len - base_seq_len),
        "cache_delta": export_static_cache_slice(
            cache_source,
            start_pos=base_seq_len,
            end_pos=seq_len,
            device=device,
        ),
        "rope_deltas": rope,
        "first_codebook_history_delta": first_tokens,
        "codec_ids_delta": codec_delta,
    }


def _decoder_context_from_delta(
    base_context: Optional[torch.Tensor],
    delta_codes: torch.Tensor,
) -> Optional[torch.Tensor]:
    pieces = []
    if base_context is not None and base_context.numel() > 0:
        pieces.append(base_context)
    if delta_codes.numel() > 0:
        pieces.append(delta_codes)
    if not pieces:
        return None
    merged = torch.cat(pieces, dim=0)
    if merged.shape[0] > DECODER_CONTEXT_FRAMES:
        merged = merged[-DECODER_CONTEXT_FRAMES:]
    return merged


def prefill_with_continuation(
    *,
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    continuation_state: Optional[Dict[str, Any]],
    max_seq_len: int,
    device: torch.device | str,
) -> Tuple[Any, torch.Tensor, int]:
    """Run talker prefill with or without an existing continuation cache."""
    base_seq_len = 0 if continuation_state is None else int(continuation_state["seq_len"])
    full_attention_mask = attention_mask
    if continuation_state is None:
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
        return out, full_attention_mask, base_seq_len

    prefix_len = int(talker_input_embeds.shape[1])
    required_len = base_seq_len + prefix_len
    if required_len >= max_seq_len - 1:
        raise RuntimeError(
            "Continuation prefill exceeds max_seq_len. Reset the continuation state "
            "or increase max_seq_len."
        )
    full_attention_mask = torch.ones(
        attention_mask.shape[0],
        required_len,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    prefix_cache_position = torch.arange(base_seq_len, required_len, device=device)
    prefix_input_ids = torch.zeros(
        (talker_input_embeds.shape[0], prefix_len),
        dtype=torch.long,
        device=device,
    )
    talker.rope_deltas = continuation_state["rope_deltas"].to(
        device=talker_input_embeds.device,
        dtype=torch.float32,
    )
    past_key_values = continuation_state_to_dynamic_cache(
        continuation_state,
        config=talker.config,
        device=device,
    )
    out = talker.forward(
        input_ids=prefix_input_ids,
        inputs_embeds=talker_input_embeds,
        attention_mask=full_attention_mask,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
        trailing_text_hidden=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        generation_step=None,
        past_hidden=None,
        past_key_values=past_key_values,
        cache_position=prefix_cache_position,
    )
    return out, full_attention_mask, base_seq_len


def attach_continuation_result(
    *,
    timing: Dict[str, Any],
    continuation_return_mode: ReturnContinuationMode,
    running_state: Optional[Dict[str, Any]],
    cache_source,
    base_seq_len: int,
    seq_len: int,
    rope_deltas: Optional[torch.Tensor],
    first_codebook_history_delta: list[torch.Tensor],
    codec_ids_delta: list[torch.Tensor],
    mode: str,
    non_streaming_mode: bool,
    model_signature_dict: Dict[str, int],
    device: str,
    max_seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Attach delta/full continuation metadata to a timing dict."""
    delta = build_continuation_state_delta(
        cache_source=cache_source,
        base_seq_len=base_seq_len,
        seq_len=seq_len,
        rope_deltas=rope_deltas,
        first_codebook_history_delta=first_codebook_history_delta,
        codec_ids_delta=codec_ids_delta,
        mode=mode,
        non_streaming_mode=non_streaming_mode,
        model_signature_dict=model_signature_dict,
        device=device,
    )
    if continuation_return_mode == "delta":
        timing["continuation_state_delta"] = delta
    else:
        running_state = apply_continuation_state_delta(running_state, delta)
        timing["continuation_state"] = running_state
    timing["continuation_state_status"] = build_continuation_state_status(
        seq_len=seq_len,
        max_seq_len=max_seq_len,
    )
    return running_state


def apply_continuation_state_delta(
    state: Optional[Dict[str, Any]],
    delta: Dict[str, Any],
) -> Dict[str, Any]:
    if delta.get("version") != CONTINUATION_STATE_VERSION:
        raise ValueError(
            f"Unsupported continuation delta version: {delta.get('version')!r}"
        )
    if delta.get("state_kind") == "full":
        return delta
    if delta.get("state_kind") != "delta":
        raise ValueError("Unknown continuation state kind")

    if state is None:
        if int(delta["base_seq_len"]) != 0:
            raise ValueError(
                "Cannot build a full continuation state from a delta whose base_seq_len is non-zero"
            )
        base_context = None
        cache = [
            {
                "key": layer["key"].clone(),
                "value": layer["value"].clone(),
            }
            for layer in delta["cache_delta"]
        ]
        first_history = delta["first_codebook_history_delta"].clone()
        rope_deltas = delta["rope_deltas"].clone().to(dtype=torch.float32)
    else:
        validate_full_continuation_state(
            state,
            mode=delta["mode"],
            expected_signature=delta["model_signature"],
        )
        if state["seq_len"] != delta["base_seq_len"]:
            raise ValueError(
                "continuation_state seq_len does not match the supplied delta base_seq_len"
            )
        cache = []
        for existing, update in zip(state["cache"], delta["cache_delta"]):
            cache.append(
                {
                    "key": torch.cat([existing["key"], update["key"].to(existing["key"].device)], dim=2),
                    "value": torch.cat(
                        [existing["value"], update["value"].to(existing["value"].device)],
                        dim=2,
                    ),
                }
            )
        first_history = torch.cat(
            [
                state["first_codebook_history"],
                delta["first_codebook_history_delta"].to(state["first_codebook_history"].device),
            ],
            dim=0,
        )
        base_context = state.get("decoder_context_codes")
        rope_deltas = delta["rope_deltas"].to(
            device=state["rope_deltas"].device,
            dtype=state["rope_deltas"].dtype,
        ).clone()

    context_device = (
        base_context.device
        if base_context is not None
        else first_history.device
    )
    delta_codes = delta["codec_ids_delta"].to(context_device)
    decoder_context = _decoder_context_from_delta(base_context, delta_codes)

    return {
        "version": CONTINUATION_STATE_VERSION,
        "state_kind": "full",
        "mode": delta["mode"],
        "non_streaming_mode": bool(delta["non_streaming_mode"]),
        "model_signature": dict(delta["model_signature"]),
        "seq_len": int(delta["seq_len"]),
        "cache": cache,
        "rope_deltas": rope_deltas,
        "first_codebook_history": first_history,
        "decoder_context_codes": decoder_context,
    }
