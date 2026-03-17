#!/bin/bash
set -e

# Log all output to a file for debugging
exec > >(tee -a /var/log/faster-qwen3-tts-startup.log)
exec 2>&1

echo "Starting Faster Qwen3 TTS API deployment setup at $(date)"

# Wait for system to fully boot and stabilize
echo "Waiting for system initialization to complete..."
sleep 30

# Install prerequisites if not already installed
echo "Installing prerequisites..."
export DEBIAN_FRONTEND=noninteractive

# Update package list
apt-get update

# Install required packages
apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  jq

# Verify NVIDIA drivers (pre-installed in Deep Learning VM image)
echo "Verifying NVIDIA driver installation..."

# Wait for NVIDIA driver modules to be fully loaded
for i in {1..30}; do
  if nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA drivers are ready"
    nvidia-smi
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: NVIDIA drivers not responding after 2.5 minutes"
    exit 1
  fi
  echo "Waiting for NVIDIA drivers to be ready... (attempt $i/30)"
  sleep 5
done

# Install Docker if not already installed
if ! command -v docker &> /dev/null; then
  echo "Installing Docker..."

  # Clean apt cache completely to avoid stale package version issues
  echo "Cleaning apt cache..."
  apt-get clean
  rm -rf /var/lib/apt/lists/*
  rm -rf /var/cache/apt/archives/*

  # Add Docker's official GPG key
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc

  # Set up the Docker repository
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

  # Update package list with fresh cache
  apt-get update

  # Install specific version of containerd.io that we verified exists
  echo "Installing Docker packages with known-good versions..."

  # Install containerd.io first with a specific version that exists
  apt-get install -y containerd.io=1.7.29-1~ubuntu.24.04~noble || \
  apt-get install -y containerd.io=1.7.28-2~ubuntu.24.04~noble || \
  apt-get install -y containerd.io=2.2.0-2~ubuntu.24.04~noble

  # Then install the rest of Docker components
  apt-get install -y \
    docker-ce \
    docker-ce-cli \
    docker-buildx-plugin \
    docker-compose-plugin

  # Add ubuntu user to docker group
  usermod -aG docker ubuntu || true

  echo "Docker installed successfully"
else
  echo "Docker is already installed"
fi

# Install NVIDIA Container Toolkit
echo "Installing NVIDIA Container Toolkit..."
if ! command -v nvidia-ctk &> /dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  apt-get update
  apt-get install -y nvidia-container-toolkit

  echo "NVIDIA Container Toolkit installed"
else
  echo "NVIDIA Container Toolkit is already installed"
fi

# Configure Docker to use NVIDIA runtime
echo "Configuring Docker NVIDIA runtime..."
nvidia-ctk runtime configure --runtime=docker

# Set nvidia as the DEFAULT runtime for Docker
echo "Setting nvidia as default Docker runtime..."
if [ -f /etc/docker/daemon.json ]; then
  cat /etc/docker/daemon.json | jq '. + {"default-runtime": "nvidia"}' > /tmp/daemon.json
  mv /tmp/daemon.json /etc/docker/daemon.json
else
  echo '{"default-runtime": "nvidia"}' > /etc/docker/daemon.json
  nvidia-ctk runtime configure --runtime=docker
fi

# Restart Docker daemon
echo "Restarting Docker daemon..."
systemctl restart docker

# Wait for Docker daemon to be fully ready
echo "Waiting for Docker daemon to be ready..."
for i in {1..30}; do
  if docker info >/dev/null 2>&1; then
    echo "Docker daemon is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Docker daemon not ready after 2.5 minutes"
    exit 1
  fi
  echo "Waiting for Docker daemon... (attempt $i/30)"
  sleep 5
done

# Verify NVIDIA runtime is available in Docker
echo "Verifying Docker NVIDIA runtime..."
if ! docker info | grep -q nvidia; then
  echo "ERROR: NVIDIA runtime not available in Docker"
  exit 1
fi

echo "Docker daemon.json content:"
cat /etc/docker/daemon.json

# Test GPU access with Docker
echo "Testing GPU access with Docker..."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi || {
  echo "ERROR: GPU test with Docker failed"
  exit 1
}

# ============================================================
# Install k3s (lightweight Kubernetes)
# ============================================================
echo "Installing k3s..."

# Install k3s with Docker as container runtime and NVIDIA support
# Use short hostname as node name to avoid k8s 63-char label limit with GCP FQDNs
K3S_NODE_NAME=$(hostname -s)
curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--docker --node-name $K3S_NODE_NAME" sh -

# Wait for k3s to be ready
echo "Waiting for k3s to be ready..."
for i in {1..60}; do
  if kubectl get nodes >/dev/null 2>&1; then
    echo "k3s is ready"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "ERROR: k3s not ready after 5 minutes"
    exit 1
  fi
  echo "Waiting for k3s... (attempt $i/60)"
  sleep 5
done

# Make k3s.yaml readable by all users
chmod 644 /etc/rancher/k3s/k3s.yaml

# Set up kubectl for ubuntu user
mkdir -p /home/ubuntu/.kube
cp /etc/rancher/k3s/k3s.yaml /home/ubuntu/.kube/config
chown -R ubuntu:ubuntu /home/ubuntu/.kube
chmod 600 /home/ubuntu/.kube/config

# Also set KUBECONFIG for root
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml

# Set KUBECONFIG environment variable globally
echo 'export KUBECONFIG=/etc/rancher/k3s/k3s.yaml' >> /etc/environment

# Verify k3s node is ready
echo "Waiting for k3s node to be ready..."
for i in {1..30}; do
  if kubectl get nodes | grep -q " Ready"; then
    echo "k3s node is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: k3s node not ready after 2.5 minutes"
    exit 1
  fi
  echo "Waiting for node to be ready... (attempt $i/30)"
  sleep 5
done

# ============================================================
# Install Helm
# ============================================================
echo "Installing Helm..."
curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify Helm installation
helm version

# ============================================================
# Authenticate with Google Artifact Registry
# ============================================================
echo "Authenticating with Google Artifact Registry..."

# The key.json file is copied by Terraform to /home/ubuntu/key.json
KEY_FILE="/home/ubuntu/key.json"

# Wait for key file to be present (Terraform file provisioner may take a moment)
for i in {1..30}; do
  if [ -f "$KEY_FILE" ]; then
    echo "Service account key file found"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Service account key file not found after 2.5 minutes"
    exit 1
  fi
  echo "Waiting for service account key file... (attempt $i/30)"
  sleep 5
done

# Authenticate Docker with Google Artifact Registry
echo "Authenticating Docker with GAR..."
cat "$KEY_FILE" | docker login -u _json_key --password-stdin https://us-docker.pkg.dev
if [ $? -ne 0 ]; then
  echo "ERROR: Docker authentication with GAR failed"
  exit 1
fi

# Authenticate Helm with Google Artifact Registry
echo "Authenticating Helm with GAR..."
cat "$KEY_FILE" | helm registry login -u _json_key --password-stdin https://us-docker.pkg.dev
if [ $? -ne 0 ]; then
  echo "ERROR: Helm authentication with GAR failed"
  exit 1
fi

echo "Successfully authenticated with Google Artifact Registry"

# ============================================================
# Install the Faster Qwen3 TTS Helm chart from GAR
# ============================================================
echo "Installing Faster Qwen3 TTS Helm chart from Google Artifact Registry..."

# Create namespace
kubectl create namespace faster-qwen3-tts --dry-run=client -o yaml | kubectl apply -f -

# Create image pull secret for GAR
echo "Creating image pull secret for Google Artifact Registry..."
kubectl create secret docker-registry gar-docker-reg-secret \
  --docker-server=us-docker.pkg.dev \
  --docker-username=_json_key \
  --docker-password="$(cat $KEY_FILE)" \
  --namespace faster-qwen3-tts \
  --dry-run=client -o yaml | kubectl apply -f -

# Create HuggingFace token secret if token is provided
if [ -n "${huggingface_token}" ]; then
  echo "Creating HuggingFace token secret..."
  kubectl create secret generic huggingface-token \
    --from-literal=HF_TOKEN=${huggingface_token} \
    --namespace faster-qwen3-tts \
    --dry-run=client -o yaml | kubectl apply -f -
  echo "HuggingFace token secret created"
else
  echo "No HuggingFace token provided, skipping secret creation"
fi

# Install the Helm chart from OCI registry with custom values
HELM_ARGS="--namespace faster-qwen3-tts"
HELM_ARGS="$HELM_ARGS --set imagePullSecrets[0].name=gar-docker-reg-secret"
HELM_ARGS="$HELM_ARGS --set model.name=${tts_model}"
HELM_ARGS="$HELM_ARGS --set model.chunkSize=${chunk_size}"

# Set Docker registry if specified (otherwise uses values.yaml default)
if [ -n "${docker_registry}" ]; then
  HELM_ARGS="$HELM_ARGS --set image.repository=${docker_registry}"
  echo "Using Docker registry override: ${docker_registry}"
else
  echo "Using default Docker registry from Helm chart values.yaml"
fi

# Set Docker image tag if specified (otherwise uses values.yaml default)
if [ -n "${docker_image_tag}" ]; then
  HELM_ARGS="$HELM_ARGS --set image.tag=${docker_image_tag}"
  echo "Using Docker image tag override: ${docker_image_tag}"
else
  echo "Using default Docker image tag from Helm chart values.yaml"
fi

# Set HF token secret name if token is provided
if [ -n "${huggingface_token}" ]; then
  HELM_ARGS="$HELM_ARGS --set hfToken.secretName=huggingface-token"
fi

helm upgrade --install faster-qwen3-tts ${gar_helm_registry} \
  --version ${helm_chart_version} \
  $HELM_ARGS \
  --wait \
  --timeout 20m

echo "Helm chart installed successfully"

# ============================================================
# Create convenience scripts
# ============================================================

# Create tts-info.sh script
cat > /home/ubuntu/tts-info.sh << 'EOF'
#!/bin/bash
echo "Faster Qwen3 TTS System Information"
echo "===================================="
echo ""
echo "GPU Information:"
nvidia-smi
echo ""
echo "Kubernetes Nodes:"
kubectl get nodes -o wide
echo ""
echo "Faster Qwen3 TTS Pods:"
kubectl get pods -n faster-qwen3-tts -o wide
echo ""
echo "Faster Qwen3 TTS Services:"
kubectl get svc -n faster-qwen3-tts
echo ""
echo "Pod Logs (last 50 lines):"
kubectl logs -n faster-qwen3-tts -l app=faster-qwen3-tts --tail=50 2>/dev/null || echo "No logs available yet"
echo ""
echo "Faster Qwen3 TTS Health (via NodePort 30800):"
curl -s http://localhost:30800/health | jq '.' || echo "API not ready yet"
echo ""
EOF

chmod +x /home/ubuntu/tts-info.sh
chown ubuntu:ubuntu /home/ubuntu/tts-info.sh

# Create tts-logs.sh script
cat > /home/ubuntu/tts-logs.sh << 'EOF'
#!/bin/bash
kubectl logs -n faster-qwen3-tts -l app=faster-qwen3-tts -f
EOF
chmod +x /home/ubuntu/tts-logs.sh
chown ubuntu:ubuntu /home/ubuntu/tts-logs.sh

# Create tts-restart.sh script
cat > /home/ubuntu/tts-restart.sh << 'EOF'
#!/bin/bash
kubectl rollout restart deployment/faster-qwen3-tts -n faster-qwen3-tts
kubectl rollout status deployment/faster-qwen3-tts -n faster-qwen3-tts
EOF
chmod +x /home/ubuntu/tts-restart.sh
chown ubuntu:ubuntu /home/ubuntu/tts-restart.sh

# Create tts-test.sh script
cat > /home/ubuntu/tts-test.sh << 'EOF'
#!/bin/bash
echo "Testing Faster Qwen3 TTS API..."
echo ""
echo "Health Check:"
curl -s http://localhost:30800/health | jq '.'
echo ""
echo ""
echo "Listing Voices:"
curl -s http://localhost:30800/voices | jq '.'
echo ""
echo ""
echo "HTTP Streaming Test (saves to output.wav):"
echo "Note: Requires a voice to be configured in /app/voices"
curl -s -X POST http://localhost:30800/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world, this is a test", "language": "English"}' \
  --output output.wav
if [ -f output.wav ]; then
  echo "Saved to output.wav"
fi
echo ""
EOF
chmod +x /home/ubuntu/tts-test.sh
chown ubuntu:ubuntu /home/ubuntu/tts-test.sh

# Add kubectl completion and KUBECONFIG to ubuntu's bashrc
cat >> /home/ubuntu/.bashrc << 'EOF'

# Kubernetes configuration
export KUBECONFIG=/home/ubuntu/.kube/config
source <(kubectl completion bash)
alias k=kubectl
complete -o default -F __start_kubectl k

# Helm completion
source <(helm completion bash)
EOF

# Mark setup as complete
touch /home/ubuntu/.cloud-init-complete
echo "Setup complete! Faster Qwen3 TTS is deploying via Kubernetes/Helm."
echo "Use 'tts-info.sh' to check status."
echo "API will be available at http://<IP>:30800 once deployed."
echo "Test with 'tts-test.sh' script."
echo "Deployment finished at $(date)"
