# faster-qwen3-tts 本机运行记录

## 已验证运行环境

- 不要优先使用仓库内 `.venv`。本项目当前真实可用环境是：
  - Python: `/home/ivan/.venvs/qwen3-tts-ray/bin/python`
  - CLI: `/home/ivan/.venvs/qwen3-tts-ray/bin/faster-qwen3-tts`
  - Hugging Face CLI: `/home/ivan/.venvs/qwen3-tts-ray/bin/hf`
- 当前默认外部入口使用 user systemd transient unit：`qwen3-tts-gateway.service`，监听 `0.0.0.0:8091`。
- 主机 Ray TTS 建议作为内部 primary，监听 `127.0.0.1:8092`；如果主机 GPU 被其他服务占用，Gateway 会自动切到 Edge。
- 2026-05-14 之前的 Ray 直连启动参数如下；现在若要作为 Gateway primary，必须把 host/port 改成 `127.0.0.1:8092`：
  ```bash
  systemd-run --user --unit="qwen3-tts-ray" \
    --property="Restart=on-failure" \
    --property="RestartSec=5" \
    --working-directory="/home/ivan/github/faster-qwen3-tts" \
    /usr/bin/env CUDA_VISIBLE_DEVICES=1,2 \
    /home/ivan/.venvs/qwen3-tts-ray/bin/python \
    /home/ivan/github/faster-qwen3-tts/examples/ray_dual_worker_server.py \
    --model /home/ivan/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --host 127.0.0.1 \
    --port 8092 \
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
  QWEN_TTS_HOST=127.0.0.1 QWEN_TTS_PORT=8092 ./start_qwen3_tts_ray.sh
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

- 主机外部 `8091` 已改为 Gateway：`qwen3-tts-gateway.service`。
- Gateway primary：`http://127.0.0.1:8092`。
- Gateway backup-1：`http://100.101.54.115:8091`，即 Edge `edgexpert-4353`。
- Edge 单卡备机已验证运行：`qwen3-tts-single.service`，模型 `/home/admin/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`，GPU `NVIDIA GB10`，端口 `8091`。
- 2026-05-14 15:57 已在主机 GPU 被 `llama-server` / Ray VL 服务占用、primary `127.0.0.1:8092` 不可用时验证自动切到 Edge：
  ```text
  x-tts-backend: backup-1
  x-tts-backend-url: http://100.101.54.115:8091
  x-tts-failover: true
  ```
- Failover API wav 验证文件：`/tmp/qwen3_tts_failover_edge_check.wav`。

## 2026-05-14 Edge 备机与透明切换方案

- 新增单卡非 Ray 备机服务：`examples/single_gpu_custom_voice_server.py`。
  - 目标是 Edge 单 GPU 机器，不启动 Ray。
  - 默认模型路径：`/home/admin/models/Qwen3-TTS-12Hz-1.7B-CustomVoice`
  - 默认端口：`8091`
  - 默认 speaker：`Serena`
  - 对外接口兼容当前主服务的 `/health`、`/api/status`、`/api/tts/load`、`/api/tts/plan`、`/api/tts/speak`、`/api/tts/speak_json`、`/v1/audio/speech`。
- Edge 临时启动脚本：
  ```bash
  ./start_qwen3_tts_single.sh
  ```
- Edge 安装为 user systemd 常驻服务：
  ```bash
  ./install_qwen3_tts_single_user_service.sh
  ```
- 新增透明切换网关：`examples/tts_failover_gateway.py`。
  - 设计用途：客户端仍调用主机原来的 `:8091`，网关优先转发到本机主服务，主服务失败、5xx 或超时后自动切到 Edge。
  - 默认主服务地址：`http://127.0.0.1:8092`
  - 当前已验证 Edge 候选：`http://100.101.54.115:8091`
  - 旧 Edge 候选保留参考：`http://192.168.31.72:8091,http://192.168.31.74:8091,http://edge.taild500c8.ts.net:8091`
  - 响应诊断头：`X-TTS-Backend`、`X-TTS-Backend-Url`、`X-TTS-Failover`
- 主机临时启动网关：
  ```bash
  ./start_qwen3_tts_gateway.sh
  ```
- 主机安装为 user systemd 常驻网关：
  ```bash
  ./install_qwen3_tts_gateway_user_service.sh
  ```
- 正式切换端口时，主机 Ray TTS 应改为只监听内部端口，例如：
  ```bash
  QWEN_TTS_HOST=127.0.0.1 QWEN_TTS_PORT=8092 ./start_qwen3_tts_ray.sh
  ```
  然后让 gateway 监听外部 `8091`。停止现有 `qwen3-tts-ray.service`、安装 user service、切换端口都属于有运行影响的操作，需要用户明确确认后再执行。
- 当前 Edge 模型同步经验：
  - Spark 与 AMD 均未找到可直接复用的 `Qwen3-TTS-12Hz-1.7B-CustomVoice` safetensors 模型；Spark 只有 0.6B Base，AMD 只有 0.6B 和 GGUF TTS。
  - 通过 AMD 跳板到 Edge 的 `192.168.100.148` 实测只有几十到几百 KB/s，不适合传 4.3G 模型。
  - 已使用 Tailscale 地址 `100.101.54.115` 从本机 rsync 同步模型到 Edge，速度约 `6-7MB/s`。
  - Edge 仓库路径：`/home/admin/github/faster-qwen3-tts`；Edge 可用 Python：`/home/admin/github/faster-qwen3-tts/.venv/bin/python`。

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
