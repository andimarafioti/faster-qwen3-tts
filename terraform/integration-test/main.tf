terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  required_version = ">= 1.0"

  backend "gcs" {}
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "qarl-studio"
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-east1"
}

variable "zone" {
  description = "GCP zone"
  type        = string
  default     = "us-east1-b"
}

variable "image_registry" {
  description = "Docker image registry path (e.g. us-docker.pkg.dev/project/repo)"
  type        = string
  default     = "us-docker.pkg.dev/qarlproductionregistries/dockerimg"
}

variable "image_tag" {
  description = "Faster Qwen3 TTS Docker image tag"
  type        = string
}

variable "deployment_name" {
  description = "Unique deployment name (used in resource names and state path)"
  type        = string
}

variable "github_token" {
  description = "GitHub PAT for cloning private repo"
  type        = string
  sensitive   = true
}

variable "hf_token" {
  description = "HuggingFace API token"
  type        = string
  sensitive   = true
  default     = ""
}

variable "service_account_email" {
  description = "Service account email to attach to the instance"
  type        = string
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------

resource "google_compute_network" "integration_test" {
  name                    = "faster-qwen3-tts-${var.deployment_name}-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "integration_test" {
  name          = "faster-qwen3-tts-${var.deployment_name}-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.region
  network       = google_compute_network.integration_test.id
}

resource "google_compute_firewall" "allow_api" {
  name    = "faster-qwen3-tts-${var.deployment_name}-allow-api"
  network = google_compute_network.integration_test.name

  allow {
    protocol = "tcp"
    ports    = ["22", "8000", "30800"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["faster-qwen3-tts-it"]
}

# ---------------------------------------------------------------------------
# Static IP
# ---------------------------------------------------------------------------

resource "google_compute_address" "integration_test" {
  name   = "faster-qwen3-tts-${var.deployment_name}-ip"
  region = var.region
}

# ---------------------------------------------------------------------------
# Compute Instance
# ---------------------------------------------------------------------------

resource "google_compute_instance" "integration_test" {
  name         = "faster-qwen3-tts-${var.deployment_name}"
  machine_type = "g2-standard-4"
  zone         = var.zone

  tags = ["faster-qwen3-tts-it"]

  boot_disk {
    initialize_params {
      image = "deeplearning-platform-release/common-cu128-ubuntu-2404-nvidia-570"
      size  = 100
      type  = "pd-balanced"
    }
  }

  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  network_interface {
    subnetwork = google_compute_subnetwork.integration_test.id
    access_config {
      nat_ip = google_compute_address.integration_test.address
    }
  }

  metadata_startup_script = templatefile("${path.module}/startup-script.sh.tpl", {
    image_registry = var.image_registry
    image_tag      = var.image_tag
    github_token   = var.github_token
    hf_token       = var.hf_token
  })

  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }

  allow_stopping_for_update = true
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_ip" {
  description = "Public IP of the integration-test VM"
  value       = google_compute_address.integration_test.address
}

output "instance_name" {
  description = "Name of the integration-test VM"
  value       = google_compute_instance.integration_test.name
}
