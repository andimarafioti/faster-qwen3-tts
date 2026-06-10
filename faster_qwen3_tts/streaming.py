#!/usr/bin/env python3
"""
Streaming generation with CUDA graphs for both predictor and talker.

Yields codec ID chunks during generation instead of collecting all at once.
CUDA graph usage is identical to non-streaming — same per-step performance.
"""
import time
from typing import Generator, Tuple

import torch
import torch.nn.functional as F

from .generate import get_eos_tracker, get_fused_codec_embeddings
from .predictor_graph import PredictorGraph
from .sampling import apply_repetition_penalty, build_suppress_mask, sample_logits
from .talker_graph import TalkerGraph


@torch.inference_mode()
def fast_generate_streaming(
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
    chunk_size: int = 12,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """
    Streaming autoregressive generation with CUDA-graphed predictor and talker.

    Yields (codec_chunk, timing_info) tuples every chunk_size steps.
    codec_chunk: [chunk_steps, 16] tensor of codec IDs.
    The final chunk may be shorter than chunk_size.
    """
    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    device = talker_input_embeds.device

    suppress_mask = build_suppress_mask(vocab_size, eos_id, device)
    eos_suppress_ids = torch.tensor([eos_id], dtype=torch.long, device=device)

    predictor = talker.code_predictor
    talker_codec_embed = talker.get_input_embeddings()
    talker_codec_head = talker.codec_head
    fused_codec_weights, fused_codec_offsets = get_fused_codec_embeddings(predictor)

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

    prefill_len = talker_graph.prefill_kv(talker_past_kv)
    rope_deltas = getattr(talker, "rope_deltas", None)
    talker_graph.set_generation_state(attention_mask, rope_deltas)

    # Deferred EOS detection (see fast_generate): tokens are copied to a pinned
    # host buffer asynchronously and checked one iteration late so the CPU never
    # blocks on the GPU mid-step. Slot k%2 holds the token consumed by iteration
    # k. Chunk flushes additionally validate their tail entry (after the sync
    # they already perform) so an EOS overshoot is never yielded downstream.
    token_cpu, token_events = get_eos_tracker(device)
    token_cpu[0:1].copy_(token, non_blocking=True)
    token_events[0].record()

    torch.cuda.synchronize()
    t_prefill = time.time() - t_start

    # === DECODE LOOP — yield chunks ===
    chunk_buffer = []
    # Preallocated first-codebook history for repetition penalty across chunks;
    # rebuilding it with torch.stack over a growing list is O(n) launches per step.
    rep_history = None
    if repetition_penalty != 1.0:
        rep_history = torch.empty(max_new_tokens, dtype=torch.long, device=device)
    total_steps = 0
    chunk_count = 0
    eos_found = False
    chunk_start = time.time()

    for step_idx in range(max_new_tokens):
        if step_idx > 0:
            prev_slot = (step_idx - 1) % 2
            token_events[prev_slot].synchronize()
            if int(token_cpu[prev_slot]) == eos_id:
                # Previous iteration consumed EOS — drop its output and stop.
                # The entry is always still in chunk_buffer: a flush validates
                # its tail before yielding, so an EOS entry never leaves it.
                chunk_buffer.pop()
                eos_found = True
                break
        cur_slot = step_idx % 2
        if token_events[cur_slot].query() and int(token_cpu[cur_slot]) == eos_id:
            # This iteration's own token is already visible and is EOS — stop
            # before doing any work (no overshoot).
            eos_found = True
            break

        # --- CUDA-Graphed Code Predictor ---
        last_id_hidden = talker_codec_embed(token.unsqueeze(1))
        pred_input = torch.cat((past_hidden, last_id_hidden), dim=1)
        codebook_token_ids = predictor_graph.run(pred_input)

        all_cb = torch.cat([token.view(1), codebook_token_ids])
        chunk_buffer.append(all_cb.detach())
        if rep_history is not None:
            rep_history[step_idx:step_idx + 1] = token

        # --- Build input embedding for talker ---
        # One fused gather over all 15 codebook tables; the cat+sum keeps the
        # exact reduction order of the previous per-table loop for parity.
        codebook_embeds = F.embedding(
            codebook_token_ids + fused_codec_offsets, fused_codec_weights
        ).unsqueeze(0)  # [1, 15, H]
        inputs_embeds = torch.cat((last_id_hidden, codebook_embeds), dim=1).sum(1, keepdim=True)

        if gen_step < trailing_text_hiddens.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hiddens[:, gen_step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + tts_pad_embed

        # --- CUDA-Graphed Talker decode step ---
        current_pos = prefill_len + step_idx
        if current_pos >= talker_graph.max_seq_len - 1:
            break

        hidden_states = talker_graph.run(inputs_embeds, position=current_pos)

        logits = talker_codec_head(hidden_states[:, -1, :]).unsqueeze(0)

        if rep_history is not None:
            logits = apply_repetition_penalty(
                logits, rep_history[:step_idx + 1], repetition_penalty
            )

        suppress_eos = step_idx + 1 < min_new_tokens
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
        past_hidden = hidden_states[:, -1:, :].clone()
        gen_step += 1

        # --- Yield chunk when buffer is full ---
        if len(chunk_buffer) >= chunk_size:
            torch.cuda.synchronize()
            # This chunk's tail entry hasn't been EOS-checked yet (that happens
            # at the top of the next iteration); validate it now that we're synced.
            if int(token_cpu[step_idx % 2]) == eos_id:
                chunk_buffer.pop()
                eos_found = True
                if not chunk_buffer:
                    break
            chunk_decode_time = time.time() - chunk_start
            total_steps += len(chunk_buffer)

            yield torch.stack(chunk_buffer), {
                'chunk_index': chunk_count,
                'chunk_steps': len(chunk_buffer),
                'prefill_ms': t_prefill * 1000 if chunk_count == 0 else 0,
                'decode_ms': chunk_decode_time * 1000,
                'total_steps_so_far': total_steps,
                'is_final': eos_found,
            }

            chunk_buffer = []
            if eos_found:
                break
            chunk_count += 1
            chunk_start = time.time()

    # --- Yield final partial chunk ---
    if chunk_buffer:
        torch.cuda.synchronize()
        if not eos_found:
            # Loop exited on budget/seq-len with the newest entry unchecked;
            # slots are keyed by global iteration index.
            tail_idx = total_steps + len(chunk_buffer) - 1
            if int(token_cpu[tail_idx % 2]) == eos_id:
                chunk_buffer.pop()

    if chunk_buffer:
        chunk_decode_time = time.time() - chunk_start
        total_steps += len(chunk_buffer)

        yield torch.stack(chunk_buffer), {
            'chunk_index': chunk_count,
            'chunk_steps': len(chunk_buffer),
            'prefill_ms': t_prefill * 1000 if chunk_count == 0 else 0,
            'decode_ms': chunk_decode_time * 1000,
            'total_steps_so_far': total_steps,
            'is_final': True,
        }


@torch.inference_mode()
def parity_generate_streaming(
    talker,
    talker_input_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    trailing_text_hiddens: torch.Tensor,
    tts_pad_embed: torch.Tensor,
    config,
    max_new_tokens: int = 2048,
    min_new_tokens: int = 2,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    repetition_penalty: float = 1.05,
    chunk_size: int = 12,
) -> Generator[Tuple[torch.Tensor, dict], None, None]:
    """
    Streaming generation without CUDA graphs (dynamic cache).

    Yields (codec_chunk, timing_info) tuples every chunk_size steps.
    """
    # NOTE: This function intentionally mirrors fast_generate_streaming. The core
    # decode loop is duplicated so we can swap CUDA graphs/static cache for the
    # dynamic-cache path while keeping sampling/chunking identical. If you edit
    # the fast path, check parity_generate_streaming for matching changes.
    eos_id = config.codec_eos_token_id
    vocab_size = config.vocab_size
    device = talker_input_embeds.device

    suppress_mask = build_suppress_mask(vocab_size, eos_id, device)
    eos_suppress_ids = torch.tensor([eos_id], dtype=torch.long, device=device)

    # === PREFILL ===
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

    if attention_mask is not None:
        attention_mask = attention_mask.clone()

    torch.cuda.synchronize()
    t_prefill = time.time() - t_start

    # === DECODE LOOP — yield chunks ===
    chunk_buffer = []
    all_first_tokens = []
    total_steps = 0
    chunk_count = 0
    chunk_start = time.time()

    for _ in range(max_new_tokens):
        if token.item() == eos_id:
            break

        cache_position = None
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=1,
            )
            cache_position = torch.tensor([attention_mask.shape[1] - 1], device=attention_mask.device)

        out = talker.forward(
            input_ids=token.view(1, 1),
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            generation_step=gen_step,
            past_hidden=past_hidden,
            past_key_values=talker_past_kv,
            subtalker_dosample=do_sample,
            subtalker_top_k=top_k,
            subtalker_top_p=top_p,
            subtalker_temperature=temperature,
            cache_position=cache_position,
        )

        codec_ids = out.hidden_states[1]
        if codec_ids is None:
            break

        chunk_buffer.append(codec_ids.squeeze(0).detach())
        all_first_tokens.append(token.detach())

        logits = out.logits[:, -1, :]
        if repetition_penalty != 1.0 and all_first_tokens:
            history = torch.stack(all_first_tokens)
            logits = apply_repetition_penalty(logits, history, repetition_penalty)

        suppress_eos = len(all_first_tokens) < min_new_tokens
        token = sample_logits(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            suppress_mask=suppress_mask,
            suppress_tokens=eos_suppress_ids if suppress_eos else None,
        )

        talker_past_kv = out.past_key_values
        past_hidden = out.past_hidden
        gen_step = out.generation_step

        if len(chunk_buffer) >= chunk_size:
            torch.cuda.synchronize()
            chunk_decode_time = time.time() - chunk_start
            total_steps += len(chunk_buffer)

            yield torch.stack(chunk_buffer), {
                'chunk_index': chunk_count,
                'chunk_steps': len(chunk_buffer),
                'prefill_ms': t_prefill * 1000 if chunk_count == 0 else 0,
                'decode_ms': chunk_decode_time * 1000,
                'total_steps_so_far': total_steps,
                'is_final': False,
            }

            chunk_buffer = []
            chunk_count += 1
            chunk_start = time.time()

    if chunk_buffer:
        torch.cuda.synchronize()
        chunk_decode_time = time.time() - chunk_start
        total_steps += len(chunk_buffer)

        yield torch.stack(chunk_buffer), {
            'chunk_index': chunk_count,
            'chunk_steps': len(chunk_buffer),
            'prefill_ms': t_prefill * 1000 if chunk_count == 0 else 0,
            'decode_ms': chunk_decode_time * 1000,
            'total_steps_so_far': total_steps,
            'is_final': True,
        }
