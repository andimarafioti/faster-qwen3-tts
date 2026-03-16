variable "gcp_project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "gcp_credentials_file" {
  description = "Path to GCP service account JSON key file"
  type        = string
}

variable "gcp_region" {
  description = "GCP region to deploy resources"
  type        = string
  default     = "us-central1"
}

variable "gcp_zone" {
  description = "GCP zone to deploy resources"
  type        = string
  default     = "us-central1-a"
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "faster-qwen3-tts-api"
}

variable "machine_type" {
  description = "GCP machine type (must support GPUs)"
  type        = string
  default     = "g2-standard-4"
}

variable "gpu_type" {
  description = "GPU type to attach"
  type        = string
  default     = "nvidia-l4"
}

variable "gpu_count" {
  description = "Number of GPUs to attach"
  type        = number
  default     = 1
}

variable "image_family" {
  description = "Image family for the boot disk (Deep Learning VM with Ubuntu 24.04, CUDA 12.8, NVIDIA drivers)"
  type        = string
  default     = "deeplearning-platform-release/common-cu128-ubuntu-2404-nvidia-570"
}

variable "boot_disk_size" {
  description = "Size of boot disk in GB"
  type        = number
  default     = 100
}

variable "allowed_ssh_cidr_blocks" {
  description = "CIDR blocks allowed to SSH into the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ssh_username" {
  description = "SSH username for the instance"
  type        = string
  default     = "ubuntu"
}

variable "public_key_path" {
  description = "Path to public SSH key"
  type        = string
}

variable "private_key_path" {
  description = "Path to private SSH key (for provisioner connection)"
  type        = string
}

variable "service_account_email" {
  description = "Service account email to attach to the instance"
  type        = string
}

variable "gar_service_account_key_path" {
  description = "Path to Google Artifact Registry service account key JSON file"
  type        = string
}

variable "gar_helm_registry" {
  description = "Full OCI path to Helm chart (e.g., oci://us-docker.pkg.dev/project/helm/faster-qwen3-tts). Leave empty to use default from values.yaml"
  type        = string
  default     = ""
}

variable "docker_registry" {
  description = "Full path to Docker image (e.g., us-docker.pkg.dev/project/dockerimg/faster-qwen3-tts). Leave empty to use default from values.yaml"
  type        = string
  default     = ""
}

variable "docker_image_tag" {
  description = "Tag of the Docker image to deploy. Leave empty to use default from values.yaml"
  type        = string
  default     = ""
}

variable "helm_chart_version" {
  description = "Version of the Faster Qwen3 TTS Helm chart to deploy from Google Artifact Registry"
  type        = string
}

variable "tts_model" {
  description = "HuggingFace model name for Faster Qwen3 TTS"
  type        = string
  default     = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
}

variable "chunk_size" {
  description = "Number of codec frames per chunk for streaming (8 frames ≈ 667ms of audio)"
  type        = number
  default     = 8
}

variable "huggingface_token" {
  description = "HuggingFace API token for downloading private/gated models (optional)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "use_reservation" {
  description = "Whether to use reserved instance (set to true to use ANY_RESERVATION)"
  type        = bool
  default     = false
}
