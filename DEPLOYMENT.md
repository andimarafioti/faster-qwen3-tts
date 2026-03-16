# Faster Qwen3 TTS Deployment Guide

Deploy Faster Qwen3 TTS model on GCP with real-time streaming audio using CUDA graphs for 6-10x speedup.

## Features

- **CUDA Graph Acceleration**: 6-10x speedup over baseline using static cache and CUDA graphs
- **Real-time Streaming**: Low-latency audio generation with chunked output
- **Multiple API Endpoints**: WebSocket, HTTP streaming, and non-streaming endpoints
- **Voice Cloning**: Upload custom voices with reference audio + transcript
- **Production Ready**: Kubernetes deployment with health checks and auto-restart
- **Infrastructure as Code**: Terraform for GCP infrastructure, Helm for K8s deployment

## Architecture

- **Application**: FastAPI server with WebSocket and HTTP streaming
- **Model**: Faster Qwen3 TTS (CUDA graph optimized)
- **Base Model**: Qwen3-TTS-12Hz-1.7B-Base (or 0.6B)
- **Infrastructure**: GCP Compute Engine with L4 GPU
- **Container Orchestration**: K3s (lightweight Kubernetes)
- **Deployment**: Helm charts with configurable parameters

## Prerequisites

### Required
1. GCP account with GPU quota in target region
2. Service account with necessary permissions
3. Google Artifact Registry (GAR) repositories:
   - Docker images: `us-docker.pkg.dev/<project>/dockerimg`
   - Helm charts: `us-docker.pkg.dev/<project>/helm`
4. HuggingFace account (for model access)
5. Terraform >= 1.0
6. Docker (for building images)
7. Helm >= 3.0

### Optional
- Voice samples (.wav + .txt files) in `./voices/` directory
- SSH key pair for instance access

## Project Structure

```
faster-qwen3-tts/
├── app.py                          # FastAPI application
├── config.py                       # Configuration loader
├── config.yaml                     # Default configuration
├── Dockerfile                      # Docker image definition
├── requirements.txt                # Python dependencies
├── voices/                         # Voice samples (WAV + TXT files)
├── client/                         # Test clients
│   ├── test_client.py             # Unified streaming/non-streaming client
│   └── manage_voices.py           # Voice management utility
├── terraform/
│   └── gcp/
│       ├── main.tf                # GCP infrastructure
│       ├── variables.tf           # Terraform variables
│       ├── outputs.tf             # Terraform outputs
│       ├── terraform.tfvars       # Your configuration (gitignored)
│       ├── terraform.tfvars.example # Example configuration
│       └── startup-script.sh.tpl  # VM initialization script
└── helm/
    ├── Chart.yaml                 # Helm chart metadata
    ├── values.yaml                # Default configuration
    └── templates/
        └── deployment.yaml        # Kubernetes deployment

```

## Quick Start

### Step 1: Prepare Voice Samples

Create voice samples in the `voices/` directory:

```bash
# Each voice needs two files:
voices/
├── my_voice.wav          # Audio sample (≥3 seconds)
└── my_voice.txt          # Transcript of the audio
```

Example:
```bash
mkdir -p voices
# Copy your voice samples
cp ~/my_voice.wav voices/
echo "This is the exact transcript of the audio" > voices/my_voice.txt
```

### Step 2: Build Docker Image

**From Linux host:**
```bash
# Build the image
docker build -t faster-qwen3-tts:1.0.0 .

# Tag for GAR
docker tag faster-qwen3-tts:1.0.0 \
  us-docker.pkg.dev/<gcp_project>/dockerimg/faster-qwen3-tts:1.0.0

# Authenticate with GAR
cat /path/to/service-account-key.json | \
  docker login -u _json_key --password-stdin https://us-docker.pkg.dev

# Push to GAR
docker push us-docker.pkg.dev/<gcp_project>/dockerimg/faster-qwen3-tts:1.0.0
```

**From Mac (Apple Silicon):**
```bash
# Build for linux/amd64 platform
docker build --platform linux/amd64 -t faster-qwen3-tts:1.0.0 .

# Rest of the steps identical to Linux host
```

**Verify image in GCP Console:**
- Navigate to: Artifact Registry → dockerimg → faster-qwen3-tts
- Confirm version 1.0.0 is present

### Step 3: Package and Push Helm Chart

```bash
cd helm

# Package the chart
helm package .
# This creates: faster-qwen3-tts-1.0.0.tgz

# Authenticate Helm with GAR
cat /path/to/service-account-key.json | \
  helm registry login -u _json_key --password-stdin us-docker.pkg.dev

# Push to GAR
helm push faster-qwen3-tts-1.0.0.tgz oci://us-docker.pkg.dev/<gcp_project>/helm

# Return to project root
cd ..
```

**Verify chart in GCP Console:**
- Navigate to: Artifact Registry → helm → faster-qwen3-tts
- Confirm version 1.0.0 is present

### Step 4: Configure Terraform

```bash
cd terraform/gcp

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit terraform.tfvars with your settings
vim terraform.tfvars  # or nano, code, etc.
```

**Required settings to update:**
```hcl
# Your GCP project and credentials
gcp_project_id       = "your-gcp-project-id"
gcp_credentials_file = "/path/to/your-gcp-credentials.json"

# Your service account
service_account_email = "your-service-account@your-project.iam.gserviceaccount.com"
gar_service_account_key_path = "/path/to/gar-service-account-key.json"

# Your GAR registries (full paths)
gar_helm_registry = "oci://us-docker.pkg.dev/your-project/helm/faster-qwen3-tts"
docker_registry   = "us-docker.pkg.dev/your-project/dockerimg/faster-qwen3-tts"

# Your HuggingFace token
huggingface_token = "hf_..."
```

### Step 5: Deploy Infrastructure

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the configuration
terraform apply

# Type 'yes' when prompted
```

**Expected duration**: 10-15 minutes for complete deployment

**Save the outputs:**
```bash
# Note the public IP
terraform output public_ip

# Save SSH command
terraform output ssh_command
```

### Step 6: Monitor Deployment

```bash
# SSH into the instance (from terraform output)
ssh -i ~/.ssh/id_ed25519 ubuntu@<PUBLIC_IP>

# Check system status
./tts-info.sh

# Watch pod logs
./tts-logs.sh

# Wait for: "Faster Qwen3 TTS API Server Ready!"
```

### Step 7: Test the API

```bash
# From your local machine
PUBLIC_IP=<your-instance-ip>

# Health check
curl http://$PUBLIC_IP:30800/health

# List voices
curl http://$PUBLIC_IP:30800/voices

# Test streaming (requires a voice to be available)
curl -X POST http://$PUBLIC_IP:30800/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world, this is a test", "language": "English", "voice": "my_voice"}' \
  --output test_output.wav

# Play the audio
play test_output.wav  # or: aplay test_output.wav
```

## Voice Management

### Voices Baked into Docker Image

All voices in the `./voices/` directory are copied into the Docker image at build time and automatically loaded at startup. No manual loading required!

**To add default voices:**
1. Add `.wav` and `.txt` files to `./voices/` directory
2. Rebuild the Docker image
3. Push to GAR
4. Redeploy

### Upload Custom Voices at Runtime

Upload new voices without redeploying:

```bash
# Upload voice with transcript (required)
curl -X POST http://<PUBLIC_IP>:30800/voices/upload \
  -F "voice_name=custom_voice" \
  -F "wav_file=@sample.wav" \
  -F "txt_file=@transcript.txt"

# Upload user-specific voice
curl -X POST http://<PUBLIC_IP>:30800/voices/upload \
  -F "voice_name=my_voice" \
  -F "uid=user123" \
  -F "wav_file=@sample.wav" \
  -F "txt_file=@transcript.txt"
```

**Requirements:**
- WAV file: ≥3 seconds, mono or stereo, any sample rate
- TXT file: Exact transcript of the audio

**Voice is immediately available** after upload - no restart needed!

### Using the Voice Management Script

We provide a helper script for voice operations:

```bash
# List all voices
python client/manage_voices.py <PUBLIC_IP> list

# Upload a voice
python client/manage_voices.py <PUBLIC_IP> upload john_voice sample.wav --transcript transcript.txt

# Test a voice
python client/manage_voices.py <PUBLIC_IP> test my_voice "Hello world"
```

## API Endpoints

### GET /health
Health check endpoint
```bash
curl http://<PUBLIC_IP>:30800/health
```

### GET /voices
List available voices
```bash
curl http://<PUBLIC_IP>:30800/voices
```

### POST /voices/upload
Upload a new voice
```bash
curl -X POST http://<PUBLIC_IP>:30800/voices/upload \
  -F "voice_name=my_voice" \
  -F "wav_file=@voice.wav" \
  -F "txt_file=@transcript.txt"
```

### POST /tts/stream
HTTP streaming TTS (returns WAV file)
```bash
curl -X POST http://<PUBLIC_IP>:30800/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "English", "voice": "my_voice"}' \
  --output output.wav
```

### POST /tts
Non-streaming TTS (returns raw PCM float32)
```bash
curl -X POST http://<PUBLIC_IP>:30800/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "English", "voice": "my_voice"}'
```

### WS /tts/ws
WebSocket streaming TTS
```bash
wscat -c ws://<PUBLIC_IP>:30800/tts/ws
> {"text": "Hello world", "language": "English", "voice": "my_voice"}
```

## Testing with Python Clients

### Streaming Client (Recommended)

```bash
# Install dependencies
pip install websockets numpy sounddevice requests

# Test streaming with real-time playback
python client/test_client.py <PUBLIC_IP> "Hello world. This is a test." --voice my_voice

# Process entire file, paragraph by paragraph
python client/test_client.py <PUBLIC_IP> --file document.txt --mode paragraph --voice my_voice --output-dir output/

# No playback, just save WAVs
python client/test_client.py <PUBLIC_IP> --file document.txt --voice my_voice --no-play --output-dir output/
```

### Non-Streaming Client

```bash
# Use non-streaming mode
python client/test_client.py <PUBLIC_IP> "Hello world" --voice my_voice --no-streaming --save output.wav
```

See client help for all options:
```bash
python client/test_client.py --help
```

## Configuration

### Model Selection

Edit `terraform.tfvars`:
```hcl
tts_model = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"  # Default (better quality)
# or
tts_model = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"  # Faster, smaller
```

### Streaming Parameters

Edit `terraform.tfvars` or `helm/values.yaml`:
```yaml
chunk_size: 8  # Codec frames per chunk (8 frames ≈ 667ms of audio)
```

**Chunk size vs performance:**
- `chunk_size: 4` → ~333ms chunks (lower latency)
- `chunk_size: 8` → ~667ms chunks (balanced, default)
- `chunk_size: 12` → ~1000ms chunks (higher throughput)

### GPU Configuration

Edit `terraform.tfvars`:
```hcl
machine_type = "g2-standard-4"   # L4 GPU, 4 vCPUs, 16GB RAM
gpu_type     = "nvidia-l4"
gpu_count    = 1
```

**GPU Options:**

| Instance Type | GPU | VRAM | Cost/hr (approx) | RTF (1.7B) | Recommended For |
|---------------|-----|------|------------------|------------|-----------------|
| g2-standard-4 | L4  | 24GB | ~$0.70 | ~1.8-2.5x | Development, Testing |
| g2-standard-12 | L4  | 24GB | ~$1.30 | ~1.8-2.5x | Production |
| a2-highgpu-1g | A100 | 40GB | ~$3.00 | ~3.3x | High Performance |

**Note**: RTF values are approximate for 1.7B model. See [faster-qwen3-tts benchmarks](https://github.com/andimarafioti/faster-qwen3-tts#results) for detailed performance data.

### Using Reserved Instances

If you have a reserved instance:

```hcl
# terraform/gcp/terraform.tfvars
machine_type = "g2-standard-12"  # Match your reservation
use_reservation = true
```

The `use_reservation = true` setting uses `ANY_RESERVATION` affinity type to automatically consume matching reservations.

## Build and Deploy Checklist

### ✓ Pre-Deployment

- [ ] GCP account with GPU quota in target region
- [ ] Service account JSON key file available
- [ ] SSH key pair generated
- [ ] Docker installed and running
- [ ] Terraform >= 1.0 installed
- [ ] Helm >= 3.0 installed
- [ ] HuggingFace token obtained
- [ ] Voice samples prepared in `./voices/` (optional)

### ✓ Build Phase

#### 1. Build Docker Image

**From Linux:**
```bash
docker build -t faster-qwen3-tts:1.0.0 .
```

**From Mac (Apple Silicon):**
```bash
docker build --platform linux/amd64 -t faster-qwen3-tts:1.0.0 .
```

- [ ] Docker build completed successfully
- [ ] (Optional) Local test passed with `docker run --gpus all -p 8000:8000 faster-qwen3-tts:1.0.0`

#### 2. Push to Google Artifact Registry

```bash
# Replace <gcp_project> with your project ID
export GCP_PROJECT="your-gcp-project-id"
export SERVICE_ACCOUNT_KEY="/path/to/service-account-key.json"

# Tag for GAR
docker tag faster-qwen3-tts:1.0.0 \
  us-docker.pkg.dev/$GCP_PROJECT/dockerimg/faster-qwen3-tts:1.0.0

# Authenticate
cat $SERVICE_ACCOUNT_KEY | \
  docker login -u _json_key --password-stdin https://us-docker.pkg.dev

# Push
docker push us-docker.pkg.dev/$GCP_PROJECT/dockerimg/faster-qwen3-tts:1.0.0
```

- [ ] Docker authentication successful
- [ ] Image pushed to GAR
- [ ] Verify in GCP Console: Artifact Registry → dockerimg → faster-qwen3-tts

#### 3. Package and Push Helm Chart

```bash
cd helm

# Package the chart (creates faster-qwen3-tts-1.0.0.tgz)
helm package .

# Authenticate Helm with GAR
cat $SERVICE_ACCOUNT_KEY | \
  helm registry login -u _json_key --password-stdin us-docker.pkg.dev

# Push to GAR
helm push faster-qwen3-tts-1.0.0.tgz oci://us-docker.pkg.dev/$GCP_PROJECT/helm

cd ..
```

- [ ] Helm chart packaged
- [ ] Chart pushed to GAR
- [ ] Verify in GCP Console: Artifact Registry → helm → faster-qwen3-tts

### ✓ Infrastructure Deployment

#### 4. Configure Terraform

```bash
cd terraform/gcp

# Copy example configuration
cp terraform.tfvars.example terraform.tfvars

# Edit with your settings
vim terraform.tfvars
```

**Update these values:**
```hcl
gcp_project_id       = "your-gcp-project-id"
gcp_credentials_file = "/path/to/gcp-credentials.json"

service_account_email = "your-sa@your-project.iam.gserviceaccount.com"
gar_service_account_key_path = "/path/to/gar-key.json"

# Full paths to your GAR resources
gar_helm_registry = "oci://us-docker.pkg.dev/your-project/helm/faster-qwen3-tts"
docker_registry   = "us-docker.pkg.dev/your-project/dockerimg/faster-qwen3-tts"

huggingface_token = "hf_..."
```

- [ ] terraform.tfvars configured
- [ ] All paths verified
- [ ] Credentials accessible

#### 5. Deploy with Terraform

```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Review the plan
terraform plan

# Apply (creates all GCP resources)
terraform apply

# Type 'yes' when prompted
```

- [ ] Terraform init successful
- [ ] Plan reviewed
- [ ] Apply completed
- [ ] Public IP noted from outputs

**Expected duration**: 10-15 minutes for complete deployment

### ✓ Post-Deployment Verification

#### 6. Monitor Deployment

```bash
# Get public IP from Terraform outputs
terraform output public_ip

# SSH into instance
ssh -i ~/.ssh/id_ed25519 ubuntu@<PUBLIC_IP>

# Watch startup logs
sudo journalctl -u google-startup-scripts.service -f

# Wait for: "Setup complete! Faster Qwen3 TTS is deploying..."
```

- [ ] SSH connection successful
- [ ] Startup script completed
- [ ] k3s installed
- [ ] Helm chart deployed

#### 7. Verify Service Status

```bash
# On the instance
./tts-info.sh

# Check:
# - GPU detected (nvidia-smi output)
# - K8s node is Ready
# - Pod is Running
# - Service is available
```

- [ ] GPU detected and functional
- [ ] Pod status: Running
- [ ] Health check returns "ok"

#### 8. Check Application Logs

```bash
# View pod logs
./tts-logs.sh

# Look for:
# - "Model loaded in X.XXs"
# - "CUDA graphs captured and ready"
# - "Faster Qwen3 TTS API Server Ready!"
```

- [ ] Model loaded successfully
- [ ] CUDA graphs initialized
- [ ] No error messages

#### 9. Test API Endpoints

```bash
# From your local machine
export PUBLIC_IP=<your-instance-ip>

# Test 1: Health check
curl http://$PUBLIC_IP:30800/health

# Test 2: List voices
curl http://$PUBLIC_IP:30800/voices

# Test 3: HTTP streaming (if voices are available)
curl -X POST http://$PUBLIC_IP:30800/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world, this is a test", "language": "English", "voice": "my_voice"}' \
  --output test.wav

# Test 4: Verify and play audio
file test.wav
play test.wav
```

- [ ] Health check returns status "ok"
- [ ] Voices listed (if any in Docker image)
- [ ] HTTP streaming returns valid WAV
- [ ] Audio plays correctly

#### 10. Test with Python Client

```bash
# Install client dependencies
pip install websockets numpy sounddevice requests

# Test streaming
python client/test_client.py $PUBLIC_IP "Hello world. This is a test." --voice my_voice

# Test with file input
python client/test_client.py $PUBLIC_IP --file test.txt --voice my_voice --output-dir output/
```

- [ ] Python client works
- [ ] Audio streams and plays correctly
- [ ] WAV files saved successfully

## Performance Tuning

### Chunk Size

Lower chunk size = lower latency but more overhead:
```yaml
# helm/values.yaml
model:
  chunkSize: 4   # ~333ms per chunk (lower latency)
  chunkSize: 8   # ~667ms per chunk (balanced, default)
  chunkSize: 12  # ~1000ms per chunk (higher throughput)
```

### Expected Performance (RTX 4090, 1.7B model)

- **RTF**: ~4.2x (faster than real-time)
- **TTFA**: ~174ms (time to first audio chunk)
- **Throughput**: ~15ms per codec step

See [faster-qwen3-tts benchmarks](https://github.com/andimarafioti/faster-qwen3-tts#results) for comprehensive performance data across different GPUs.

## Troubleshooting

### Pod Not Starting

```bash
kubectl describe pod -n faster-qwen3-tts -l app=faster-qwen3-tts
kubectl logs -n faster-qwen3-tts -l app=faster-qwen3-tts
```

Common issues:
- Image pull errors: Check GAR authentication and image exists
- OOM errors: Increase shared memory or use larger GPU

### GPU Not Detected

```bash
# On the instance
nvidia-smi  # Should show GPU

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Check K8s pod GPU access
kubectl exec -it -n faster-qwen3-tts <pod-name> -- nvidia-smi
```

### Model Download Issues

```bash
# Check HF token secret
kubectl get secret huggingface-token -n faster-qwen3-tts

# Check pod environment
kubectl exec -it -n faster-qwen3-tts <pod-name> -- env | grep HF_TOKEN
```

### Voice Not Found

If you get "Voice not found" errors:
1. Check available voices: `curl http://<PUBLIC_IP>:30800/voices`
2. Upload the voice if missing
3. Verify voice files exist in container: `kubectl exec -it -n faster-qwen3-tts <pod-name> -- ls /app/voices/`

### CUDA Graph Capture Issues

CUDA graphs capture automatically on first generation. If you see warnings:
- Ensure PyTorch >= 2.5.1 is used
- Check GPU compute capability (requires CC 7.0+)
- Review pod logs for capture errors

## Updating Deployment

### Update Docker Image

```bash
# Build new version
docker build -t faster-qwen3-tts:1.0.1 .
docker tag faster-qwen3-tts:1.0.1 \
  us-docker.pkg.dev/$GCP_PROJECT/dockerimg/faster-qwen3-tts:1.0.1
docker push us-docker.pkg.dev/$GCP_PROJECT/dockerimg/faster-qwen3-tts:1.0.1

# Update deployment on instance
ssh ubuntu@<PUBLIC_IP>
kubectl set image deployment/faster-qwen3-tts \
  faster-qwen3-tts=us-docker.pkg.dev/$GCP_PROJECT/dockerimg/faster-qwen3-tts:1.0.1 \
  -n faster-qwen3-tts
```

### Update Helm Chart

```bash
# Update Chart.yaml version first
cd helm
# Edit Chart.yaml: version: 1.0.1

# Package and push
helm package .
helm push faster-qwen3-tts-1.0.1.tgz oci://us-docker.pkg.dev/$GCP_PROJECT/helm

# Upgrade on instance
ssh ubuntu@<PUBLIC_IP>
helm upgrade faster-qwen3-tts \
  oci://us-docker.pkg.dev/$GCP_PROJECT/helm/faster-qwen3-tts \
  --version 1.0.1 \
  --namespace faster-qwen3-tts
```

### Update Infrastructure

```bash
# Edit terraform.tfvars with new settings
cd terraform/gcp
vim terraform.tfvars

# Apply changes
terraform plan
terraform apply
```

## Cost Estimation

**Monthly cost for continuous operation:**

| Component | Cost |
|-----------|------|
| g2-standard-4 (L4 GPU) | ~$500/month |
| g2-standard-12 (L4 GPU) | ~$950/month |
| Network egress | ~$50-100/month |
| Persistent storage (20GB) | ~$2/month |

**Total**: ~$550-1,050/month depending on instance type

**Cost reduction strategies:**
- Stop instance when not in use
- Use reserved instances (significant discount)
- Scale down during off-hours
- Use preemptible instances (not recommended for production)

## Cleanup

```bash
cd terraform/gcp

# Destroy all resources
terraform destroy

# Type 'yes' when prompted
```

This removes:
- Compute instance
- Static IP
- VPC network and subnet
- Firewall rules

**Not removed** (manual cleanup if needed):
- Docker images in GAR
- Helm charts in GAR
- Service accounts

## Advanced Topics

### Multi-User Support

Use the `uid` parameter for user-specific voices:

```bash
# Upload user-specific voice
curl -X POST http://<PUBLIC_IP>:30800/voices/upload \
  -F "voice_name=my_voice" \
  -F "uid=user123" \
  -F "wav_file=@voice.wav" \
  -F "txt_file=@transcript.txt"

# Use in synthesis
curl -X POST http://<PUBLIC_IP>:30800/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "voice": "my_voice", "uid": "user123"}' \
  --output output.wav
```

### Language Support

Supported languages:
- English, Chinese, Japanese, Korean
- French, German, Spanish, Italian, Portuguese
- Russian, Arabic

Example:
```bash
curl -X POST http://<PUBLIC_IP>:30800/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour le monde", "language": "French", "voice": "my_voice"}' \
  --output french.wav
```

### Performance Monitoring

```bash
# On the instance, watch GPU utilization
watch -n 1 nvidia-smi

# Monitor logs for timing info
kubectl logs -n faster-qwen3-tts -l app=faster-qwen3-tts -f | grep "Generated"
```

## License

This deployment uses:
- faster-qwen3-tts (MIT)
- Qwen3-TTS (Apache 2.0)
- FastAPI (MIT)
- PyTorch (BSD-3-Clause)

## Support

For issues:
1. Check logs: `./tts-info.sh` and `./tts-logs.sh` on instance
2. Review Terraform outputs: `terraform output`
3. Check GCP console for instance status
4. Review startup script logs: `sudo journalctl -u google-startup-scripts.service`
5. Check faster-qwen3-tts issues: https://github.com/andimarafioti/faster-qwen3-tts/issues

## References

- [faster-qwen3-tts GitHub](https://github.com/andimarafioti/faster-qwen3-tts)
- [Qwen3-TTS Official](https://github.com/QwenLM/Qwen3-TTS)
- [GCP GPU Documentation](https://cloud.google.com/compute/docs/gpus)
- [Terraform GCP Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
