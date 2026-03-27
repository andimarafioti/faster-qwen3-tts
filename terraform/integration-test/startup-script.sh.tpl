#!/bin/bash
set -e

exec > >(tee -a /var/log/faster-qwen3-tts-startup.log)
exec 2>&1

echo "=== Faster Qwen3 TTS startup at $(date) ==="

# Wait for system to stabilize
sleep 30

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release jq git

# ---------------------------------------------------------------------------
# Wait for NVIDIA drivers (pre-installed in Deep Learning VM image)
# ---------------------------------------------------------------------------
echo "Waiting for NVIDIA drivers..."
for i in {1..30}; do
  if nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA drivers ready"
    nvidia-smi
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: NVIDIA drivers not responding after 2.5 minutes"
    exit 1
  fi
  echo "  attempt $i/30 ..."
  sleep 5
done

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------
if ! command -v docker &>/dev/null; then
  echo "Installing Docker..."
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list >/dev/null
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# ---------------------------------------------------------------------------
# NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
if ! command -v nvidia-ctk &>/dev/null; then
  echo "Installing NVIDIA Container Toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update
  apt-get install -y nvidia-container-toolkit
fi

nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "Waiting for Docker daemon..."
for i in {1..30}; do
  if docker info >/dev/null 2>&1; then
    echo "Docker daemon ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Docker daemon not ready"
    exit 1
  fi
  sleep 5
done

# ---------------------------------------------------------------------------
# Authenticate with Google Artifact Registry & run container
# ---------------------------------------------------------------------------
echo "Authenticating Docker with GAR..."
gcloud auth configure-docker us-docker.pkg.dev --quiet

# ---------------------------------------------------------------------------
# Server container
# ---------------------------------------------------------------------------
IMAGE="${image_registry}/faster-qwen3-tts:${image_tag}"
echo "Pulling $IMAGE ..."
docker pull "$IMAGE"

echo "Starting server container..."
docker run -d --name faster-qwen3-tts \
  --gpus all \
  -p 8000:8000 \
  -p 30800:30800 \
  -v /opt/model-cache:/root/.cache/huggingface \
  -e HF_TOKEN="${hf_token}" \
  -e CUDA_LAUNCH_BLOCKING=0 \
  "$IMAGE"

echo "=== Startup script finished at $(date) ==="
