# faster-qwen3-tts 本机运行记录

## 已验证运行环境

- 不要优先使用仓库内 `.venv`。本项目当前真实可用环境是：
  - Python: `/home/ivan/.venvs/qwen3-tts-ray/bin/python`
  - CLI: `/home/ivan/.venvs/qwen3-tts-ray/bin/faster-qwen3-tts`
  - Hugging Face CLI: `/home/ivan/.venvs/qwen3-tts-ray/bin/hf`
- 当前默认后台服务使用 user systemd transient unit：`qwen3-tts-ray.service`。
- 2026-05-14 已切换为 1.7B CustomVoice 双 worker，端口仍为 `8091`，启动参数：
  ```bash
  systemd-run --user --unit="qwen3-tts-ray" \
    --property="Restart=on-failure" \
    --property="RestartSec=5" \
    --working-directory="/home/ivan/github/faster-qwen3-tts" \
    /usr/bin/env CUDA_VISIBLE_DEVICES=1,2 \
    /home/ivan/.venvs/qwen3-tts-ray/bin/python \
    /home/ivan/github/faster-qwen3-tts/examples/ray_dual_worker_server.py \
    --model /home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --host 0.0.0.0 \
    --port 8091 \
    --workers 2 \
    --attn sdpa \
    --dtype float16 \
    --language Chinese \
    --speaker Serena \
    --chunk-size 8 \
    --max-new-tokens 512
  ```
- Ray 服务常用接口：
  - `GET /health`
  - `GET /api/status`
  - `POST /api/tts/load`
  - `POST /api/tts/speak`
  - `POST /v1/audio/speech`
- 如果要替换服务参数：
  ```bash
  systemctl --user stop qwen3-tts-ray.service
  /home/ivan/.venvs/qwen3-tts-ray/bin/ray stop --force
  systemctl --user reset-failed qwen3-tts-ray.service
  ```
  然后重新执行上面的 `systemd-run --user` 命令。
- 一键启动脚本：
  ```bash
  ./start_qwen3_tts_ray.sh
  ```
  该脚本会停止同名 user service、清理 Ray 进程，然后以 transient user systemd service 方式启动 1.7B 双 worker。
- 安装为开机后 user systemd 常驻服务：
  ```bash
  ./install_qwen3_tts_ray_user_service.sh
  ```
  该脚本会写入 `~/.config/systemd/user/qwen3-tts-ray.service`，执行 `daemon-reload`，并 `enable --now`。

## 模型目录

- 0.6B CustomVoice 已验证模型：
  - `/home/ivan/models/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- 1.7B CustomVoice 已下载并验证模型：
  - `/home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - 目录大小约 `4.3G`
  - 本机默认优先使用该模型。

## 官方模型来源

- 官方 Qwen3-TTS GitHub: `https://github.com/QwenLM/Qwen3-TTS`
- Hugging Face:
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- ModelScope 对应模型存在：
  - `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - `https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`

官方 README 给出的 ModelScope 下载命令格式：

```bash
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## Hugging Face 下载经验

在本机代理环境下，`huggingface-cli download` 默认并发下载曾出现主权重连接卡住、临时文件不增长的情况。推荐使用同一真实环境里的 `hf`，关闭 Xet 并限制并发：

```bash
HF_HUB_DISABLE_XET=1 /home/ivan/.venvs/qwen3-tts-ray/bin/hf download \
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir /home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --max-workers 1
```

如果下载中断，保留 `.cache/huggingface/download/*.incomplete`，重复上述命令可续传。

## 1.7B CustomVoice 验证命令

V100 不支持 bf16，单独占用 V100 时用 `CUDA_VISIBLE_DEVICES=2` 加 `--dtype fp16`。命令：

```bash
CUDA_VISIBLE_DEVICES=2 /home/ivan/.venvs/qwen3-tts-ray/bin/faster-qwen3-tts \
  --device cuda \
  --dtype fp16 \
  custom \
  --model /home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --speaker Serena \
  --language Chinese \
  --text "这是大一点的一点七B语音模型运行测试。" \
  --output /tmp/qwen3_tts_1_7b_customvoice.wav \
  --max-new-tokens 128 \
  --greedy
```

已验证结果：

- 输出文件：`/tmp/qwen3_tts_1_7b_customvoice.wav`
- 音频时长：约 `3.36s`
- RTF：约 `0.41`
- 运行时出现 `sox: not found` 提示不影响本次 wav 生成。

## 2026-05-14 后台状态

- `/health` 已确认两个 worker 都加载 `/home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`。
- worker 使用 GPU `1` 和 `2`，`dtype=float16`，默认 speaker 为 `serena`。
- API wav 验证文件：`/tmp/qwen3_tts_1_7b_service_check.wav`。

## 2026-05-14 长文本卡顿/持续杂音排障经验

- 长文本客户端一般会先调用 `POST /api/tts/plan` 分段，再逐段调用 `POST /api/tts/speak`。如果日志出现短分段生成固定 `audio_s=40.960`，通常表示该段没有自然 EOS、直接打满 `max_new_tokens=512`，容易表现为持续杂音、啸叫或长拖尾。
- 典型日志形态：
  ```text
  [TTS] plan trace_id=... text_len=1629 chunks=29 max_chars=60 longest=60 truncated=False
  [TTS] worker=1 gpu=2 text_len=60 speaker=serena max_new_tokens=512 audio_s=40.960 ...
  ```
- 2026-05-14 已修复 `examples/ray_dual_worker_server.py`：`/api/tts/speak`、`/api/tts/speak_json`、`/v1/audio/speech` 支持可选 `max_new_tokens`；未传时按文本估算，不再让短分段默认放到 512；响应头和日志会包含 `hit_token_cap`、`suspicious_duration`。
- 修改代码后必须重启 `qwen3-tts-ray.service` 才会生效。确认是否加载新代码：
  ```bash
  systemctl --user status qwen3-tts-ray.service --no-pager
  journalctl --user -u qwen3-tts-ray.service -n 80 --no-pager
  ```
  新代码日志里应能看到 `hit_token_cap=False suspicious_duration=False` 这类字段。
- 检查最近是否还有异常段：
  ```bash
  journalctl --user -u qwen3-tts-ray.service --since "today" --no-pager \
    | rg "audio_s=40\\.960|hit_token_cap=True|suspicious_duration=True|\\[TTS\\] plan"
  ```
- 轻量接口验证：
  ```bash
  curl -sS -D - -o /tmp/qwen3_tts_restart_check.wav \
    -X POST http://127.0.0.1:8091/api/tts/speak \
    -H "Content-Type: application/json" \
    -d '{"text":"重启后的接口检查。","speaker":"Serena","max_new_tokens":96}'
  ```
  响应头应包含 `x-tts-hit-token-cap` 和 `x-tts-suspicious-duration`。

## 注意事项

- 当前后台 Ray 双 worker 服务的 GPU 占用和模型路径以 `GET /health` 返回为准。
- 如果明确要加载 1.7B，先检查后台服务和显存占用，避免抢占当前服务。
- 不要在没有用户要求时重建环境、安装依赖或切换分支；先从正在运行的进程确认解释器、模型路径和启动参数。
