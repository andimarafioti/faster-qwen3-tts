"""Per-epoch diagnostics for Qwen3-TTS finetuning.

Runs on one fixed sample each epoch (no waiting for full runs):
  - codec0 teacher-forced argmax accuracy
  - sub-talker residual teacher-forced accuracy (per-position)
  - sub-talker residual AUTOREGRESSIVE accuracy (per-position)  <-- the key health metric
  - a no-KV-cache manual AR generation: codec0-match-vs-GT, frame count, EOS, and a wav

Uses speaker_encoder(ref_mels) for the speaker embedding (exactly what training injects),
so it works on the in-memory model before the checkpoint is saved.
"""
import torch


@torch.no_grad()
def run_epoch_eval(model, batch, config, out_dir, epoch, max_frames=200):
    was_training = model.training
    model.eval()
    talker = model.talker
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype

    input_ids = batch["input_ids"].to(dev)
    codec_ids = batch["codec_ids"].to(dev)
    ref_mels = batch["ref_mels"].to(dev).to(dt)
    text_embedding_mask = batch["text_embedding_mask"].to(dev)
    codec_embedding_mask = batch["codec_embedding_mask"].to(dev)
    attention_mask = batch["attention_mask"].to(dev)
    codec_0_labels = batch["codec_0_labels"].to(dev)
    codec_mask = batch["codec_mask"].to(dev)

    spk = model.speaker_encoder(ref_mels)[:1]                      # [1, D] (use sample 0)
    b0 = slice(0, 1)
    ite = talker.text_projection(talker.model.text_embedding(input_ids[b0, :, 0])) * text_embedding_mask[b0]
    ice = talker.model.codec_embedding(input_ids[b0, :, 1]) * codec_embedding_mask[b0]
    ice[:, 6, :] = spk
    emb = ite + ice
    for i in range(1, 16):
        emb = emb + talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[b0, :, i]) * codec_mask[b0].unsqueeze(-1)

    # ---- teacher-forced codec0 + residual accuracy ----
    out = talker(inputs_embeds=emb[:, :-1, :], attention_mask=attention_mask[b0, :-1], output_hidden_states=True)
    pred0 = out.logits.argmax(-1)[0]
    gt0 = codec_0_labels[0, 1:]
    valid = gt0 != -100
    codec0_tf = (pred0[valid] == gt0[valid]).float().mean().item()

    hidden = out.hidden_states[0][-1]
    # match inference: condition on the hidden that PREDICTED each frame's codec0 (one before the frame)
    pred_mask = torch.zeros_like(codec_mask)
    pred_mask[:, :-1] = codec_mask[:, 1:]
    frame_hidden = hidden[0][pred_mask[0, :-1]]                    # [C, D]
    gt = codec_ids[0, codec_mask[0]]                              # [C, 16]
    sub_logits, _ = talker.forward_sub_talker_finetune(gt, frame_hidden)
    res_tf_pos = (sub_logits.argmax(-1) == gt[:, 1:]).float().mean(0)   # [15]

    # ---- autoregressive residual accuracy (per position) ----
    C = frame_hidden.shape[0]
    res_ar_pos = torch.zeros(15)
    for k in range(C):
        c0_emb = talker.get_input_embeddings()(gt[k, 0].view(1, 1))
        pred = talker.code_predictor.generate(
            inputs_embeds=torch.cat([frame_hidden[k].view(1, 1, -1), c0_emb], dim=1),
            max_new_tokens=15, do_sample=False, return_dict_in_generate=True,
        )
        res_ar_pos += (pred.sequences[0].cpu() == gt[k, 1:].cpu()).float()
    res_ar_pos /= max(C, 1)

    # ---- THE reality check: short no-cache AR (capped, no decode/wav so it can't hang) ----
    # The smoking-gun metric: codec0_match was pinned at exactly 1/len (only frame 0 survived).
    # If it rises above that and grows with training, the fix works.
    EOS = config.talker_config.codec_eos_token_id
    first = int(torch.where(codec_mask[0])[0][0])
    gt_len = int(codec_mask[0].sum())
    cap = gt_len + 20
    seq = emb[:, :first, :].clone()
    pad_id = torch.tensor([[config.tts_pad_token_id]], device=dev)
    pad_text = talker.text_projection(talker.model.text_embedding(pad_id))
    gen0, eos_at = [], -1
    for stp in range(cap):
        o = talker(inputs_embeds=seq, attention_mask=torch.ones(seq.shape[:2], device=dev, dtype=torch.long), output_hidden_states=True)
        c0 = o.logits[0, -1].argmax()
        if int(c0) == EOS:
            eos_at = stp
            break
        h = o.hidden_states[0][-1][:, -1:, :]
        c0_emb = talker.get_input_embeddings()(c0.view(1, 1))
        pr = talker.code_predictor.generate(inputs_embeds=torch.cat([h, c0_emb], dim=1),
                                            max_new_tokens=15, do_sample=False, return_dict_in_generate=True)
        res = pr.sequences
        gen0.append(int(c0))
        fe = c0_emb + pad_text
        for i in range(1, 16):
            fe = fe + talker.code_predictor.get_input_embeddings()[i - 1](res[:, i - 1:i])
        seq = torch.cat([seq, fe], dim=1)
    gtc0 = codec_ids[0, codec_mask[0]][:, 0].cpu()
    gen0 = torch.tensor(gen0) if gen0 else torch.empty(0)
    n = min(len(gen0), len(gtc0))
    ar_match = (gen0[:n] == gtc0[:n]).float().mean().item() if n else 0.0

    def fmt(t):
        return " ".join(f"{x:.2f}" for x in t.tolist())

    print(f"\n[EVAL epoch {epoch}] codec0_tf={codec0_tf:.3f}  res_tf={res_tf_pos.mean():.3f}  res_ar={res_ar_pos.mean():.3f}")
    print(f"  res_tf per-pos : {fmt(res_tf_pos)}")
    print(f"  res_ar per-pos : {fmt(res_ar_pos)}")
    print(f"  >>> AR REALITY: frames={len(gen0)} (GT={gt_len})  eos@{eos_at}  codec0_match={ar_match:.3f}  (was pinned 0.013)")

    if was_training:
        model.train()
    return {"codec0_tf": codec0_tf, "res_tf": res_tf_pos.mean().item(),
            "res_ar": res_ar_pos.mean().item(), "ar_match": ar_match, "eos_at": eos_at}
