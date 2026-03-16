output "instance_id" {
  description = "ID of the Compute Engine instance"
  value       = google_compute_instance.tts.id
}

output "instance_name" {
  description = "Name of the Compute Engine instance"
  value       = google_compute_instance.tts.name
}

output "public_ip" {
  description = "Public IP address of the instance"
  value       = google_compute_address.tts_ip.address
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ${var.private_key_path} ${var.ssh_username}@${google_compute_address.tts_ip.address}"
}

output "api_endpoint" {
  description = "Faster Qwen3 TTS API endpoint"
  value       = "http://${google_compute_address.tts_ip.address}:8000"
}

output "api_endpoint_k8s" {
  description = "Faster Qwen3 TTS API endpoint via Kubernetes NodePort"
  value       = "http://${google_compute_address.tts_ip.address}:30800"
}

output "gpu_info" {
  description = "GPU configuration"
  value = {
    type  = var.gpu_type
    count = var.gpu_count
  }
}

output "test_commands" {
  description = "Commands to test the Faster Qwen3 TTS API"
  value = {
    health_check     = "curl http://${google_compute_address.tts_ip.address}:30800/health"
    list_voices      = "curl http://${google_compute_address.tts_ip.address}:30800/voices"
    websocket_stream = "wscat -c ws://${google_compute_address.tts_ip.address}:30800/tts/ws"
    http_stream      = "curl -X POST http://${google_compute_address.tts_ip.address}:30800/tts/stream -H 'Content-Type: application/json' -d '{\"text\": \"Hello world\", \"language\": \"English\", \"voice\": \"my_voice\"}' --output output.wav"
    non_streaming    = "curl -X POST http://${google_compute_address.tts_ip.address}:30800/tts -H 'Content-Type: application/json' -d '{\"text\": \"Hello world\", \"language\": \"English\", \"voice\": \"my_voice\"}'"
  }
}

output "model_info" {
  description = "TTS model configuration"
  value = {
    model      = var.tts_model
    chunk_size = var.chunk_size
  }
}

output "setup_notes" {
  description = "Post-deployment notes"
  value       = <<-EOT
    GCP Faster Qwen3 TTS API Deployment Complete!

    After instance is ready (may take 10-15 minutes for full setup):
    1. SSH into the instance: ssh -i ${var.private_key_path} ${var.ssh_username}@${google_compute_address.tts_ip.address}
    2. Run './tts-info.sh' to verify setup and GPU access
    3. Check startup logs: sudo journalctl -u google-startup-scripts.service -f
    4. Check service logs: kubectl logs -n faster-qwen3-tts -l app=faster-qwen3-tts -f

    IMPORTANT: Configure Voice Samples
    - Place voice samples (.wav + .txt) in /app/voices on the container
    - Or use the /voices/upload API endpoint
    - Voice cloning is REQUIRED for generation to work

    Faster Qwen3 TTS API Usage:
    - Health check: curl http://${google_compute_address.tts_ip.address}:30800/health
    - List voices: curl http://${google_compute_address.tts_ip.address}:30800/voices

    - WebSocket streaming (requires wscat):
      wscat -c ws://${google_compute_address.tts_ip.address}:30800/tts/ws
      > {"text": "Hello world", "language": "English", "voice": "my_voice"}

    - HTTP streaming (saves to file):
      curl -X POST http://${google_compute_address.tts_ip.address}:30800/tts/stream \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world", "language": "English", "voice": "my_voice"}' \
        --output output.wav

    - Non-streaming:
      curl -X POST http://${google_compute_address.tts_ip.address}:30800/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world", "language": "English", "voice": "my_voice"}'

    Streaming Parameters:
    - chunk_size=${var.chunk_size}: Number of codec frames per chunk (8 frames ≈ 667ms)

    Performance:
    - FasterQwen3TTS uses CUDA graphs for 6-10x speedup
    - CUDA graphs are captured on first generation
    - First request may be slower while graphs are captured

    Useful Commands:
    - SSH to instance: ssh -i ${var.private_key_path} ${var.ssh_username}@${google_compute_address.tts_ip.address}
    - View instance info: gcloud compute instances describe ${var.project_name}-instance --zone=${var.gcp_zone}
    - View serial console: gcloud compute instances get-serial-port-output ${var.project_name}-instance --zone=${var.gcp_zone}
  EOT
}
