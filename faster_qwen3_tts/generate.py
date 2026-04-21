#!/usr/bin/env python3
"""
Non-streaming generation loop using CUDA graphs for both predictor and talker.
"""
import time
from typing import Optional, Tuple

import torch

from .continuation import (
    attach_continuation_result,
    continuation_state_first_token_history,
    model_signature,
    normalize_return_continuation_state,
    prefill_with_continuation,
    validate_full_continuation_state,
)
from .predictor_graph import PredictorGraph
from .sampling import apply_repetition_penalty, build_suppress_mask, sample_logits
from .talker_graph import TalkerGraph


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
    continuation_state: Optional[dict] = None,
    return_continuation_state: bool | str = False,
    continuation_mode: str = "voice_clone",
    continuation_non_streaming_mode: bool = False,
    continuation_state_device: str = "cpu",
) -> Tuple[Optional[torch.Tensor], dict]:
    """
    Fast autoregressive generation with CUDA-graphed predictor and talker.
    """
    eos_id = config.codec_eos_token_id
    num_code_groups = config.num_code_groups
    vocab_size = config.vocab_size
    device = talker_input_embeds.device
    suppress_mask = build_suppress_mask(vocab_size, eos_id, device=device)

    continuation_return_mode = normalize_return_continuation_state(return_continuation_state)
    continuation_active = continuation_state is not None or continuation_return_mode != "none"
    signature = None
    if continuation_active:
        signature = model_signature(
            num_layers=talker_graph.num_layers,
            max_seq_len=talker_graph.max_seq_len,
            hidden_size=talker_graph.hidden_size,
        )
        validate_full_continuation_state(
            continuation_state,
            mode=continuation_mode,
            expected_signature=signature,
        )
        if (
            continuation_state is not None
            and continuation_state["non_streaming_mode"] != continuation_non_streaming_mode
        ):
            raise ValueError(
                "continuation_state non_streaming_mode does not match the current request"
            )
    base_seq_len = 0 if continuation_state is None else int(continuation_state["seq_len"])

    if parity_mode:
        suppress_tokens = torch.nonzero(suppress_mask, as_tuple=False).flatten().tolist()
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
    out, full_attention_mask, base_seq_len = prefill_with_continuation(
        talker=talker,
        talker_input_embeds=talker_input_embeds,
        attention_mask=attention_mask,
        trailing_text_hiddens=trailing_text_hiddens,
        tts_pad_embed=tts_pad_embed,
        continuation_state=continuation_state,
        max_seq_len=talker_graph.max_seq_len,
        device=device,
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
        suppress_tokens=[eos_id] if suppress_eos else None,
    )
    
    # Copy prefill KV cache into talker graph's static cache
    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    # Sync padding mask + rope deltas for decode parity
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(full_attention_mask, rope_deltas)
    
    torch.cuda.synchronize()
    t_prefill = time.time() - t_start
    
    # === DECODE LOOP ===
    t_decode_start = time.time()
    all_codec_ids = []
    all_first_tokens = continuation_state_first_token_history(
        continuation_state,
        device=device,
    )
    generated_first_tokens = []
    
    for step_idx in range(max_new_tokens):
        if token.item() == eos_id:
            break
        
        # --- CUDA-Graphed Code Predictor ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))  # [1, 1, H]
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)  # [1, 2, H]
        codebook_token_ids = predictor_graph.run(pred_input)  # [15] long tensor
        
        # Build full codec: [first_cb, cb1, ..., cb15]
        all_cb = torch.cat([token.view(1), codebook_token_ids])  # [16]
        all_codec_ids.append(all_cb.detach())
        all_first_tokens.append(token.detach())
        generated_first_tokens.append(token.detach())
        
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
        
        if repetition_penalty != 1.0 and all_first_tokens:
            history = torch.stack(all_first_tokens)
            logits = apply_repetition_penalty(logits, history, repetition_penalty)

        suppress_eos = len(all_codec_ids) < min_new_tokens
        token = sample_logits(
            logits.squeeze(0),
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=[eos_id] if suppress_eos else None,
        )
        past_hidden = hidden_states[:, -1:, :].clone()  # clone since it's the static buffer
        gen_step += 1
    
    torch.cuda.synchronize()
    t_decode = time.time() - t_decode_start
    
    n_steps = len(all_codec_ids)
    timing = {
        'prefill_ms': t_prefill * 1000,
        'decode_s': t_decode,
        'steps': n_steps,
        'ms_per_step': (t_decode / n_steps * 1000) if n_steps > 0 else 0,
        'steps_per_s': (n_steps / t_decode) if t_decode > 0 else 0,
    }
    if continuation_return_mode != "none":
        final_seq_len = prefill_len + n_steps
        attach_continuation_result(
            timing=timing,
            continuation_return_mode=continuation_return_mode,
            running_state=continuation_state,
            cache_source=talker_graph.static_cache,
            base_seq_len=base_seq_len,
            seq_len=final_seq_len,
            rope_deltas=talker_graph.rope_deltas,
            first_codebook_history_delta=generated_first_tokens,
            codec_ids_delta=all_codec_ids,
            mode=continuation_mode,
            non_streaming_mode=continuation_non_streaming_mode,
            model_signature_dict=signature,
            device=continuation_state_device,
            max_seq_len=talker_graph.max_seq_len,
        )
    
    if all_codec_ids:
        return torch.stack(all_codec_ids), timing
    return None, timing
