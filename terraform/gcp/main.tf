terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  credentials = file(var.gcp_credentials_file)
  project     = var.gcp_project_id
  region      = var.gcp_region
  zone        = var.gcp_zone
}

# Create VPC Network
resource "google_compute_network" "tts_network" {
  name                    = "${var.project_name}-network"
  auto_create_subnetworks = false
}

# Create Subnet
resource "google_compute_subnetwork" "tts_subnet" {
  name          = "${var.project_name}-subnet"
  ip_cidr_range = "10.0.1.0/24"
  region        = var.gcp_region
  network       = google_compute_network.tts_network.id
}

# Create Firewall Rule for SSH
resource "google_compute_firewall" "allow_ssh" {
  name    = "${var.project_name}-allow-ssh"
  network = google_compute_network.tts_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = var.allowed_ssh_cidr_blocks
  target_tags   = ["tts-server"]
}

# Create Firewall Rule for TTS API
resource "google_compute_firewall" "allow_tts_api" {
  name    = "${var.project_name}-allow-api"
  network = google_compute_network.tts_network.name

  allow {
    protocol = "tcp"
    ports    = ["8000", "30800"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["tts-server"]
}

# Create Firewall Rule for Kubernetes API
resource "google_compute_firewall" "allow_k8s_api" {
  name    = "${var.project_name}-allow-k8s"
  network = google_compute_network.tts_network.name

  allow {
    protocol = "tcp"
    ports    = ["6443"]
  }

  source_ranges = var.allowed_ssh_cidr_blocks
  target_tags   = ["tts-server"]
}

# Reserve Static External IP
resource "google_compute_address" "tts_ip" {
  name = "${var.project_name}-ip"
}

# Create Compute Instance with GPU
resource "google_compute_instance" "tts" {
  name         = "${var.project_name}-instance"
  machine_type = var.machine_type
  zone         = var.gcp_zone

  tags = ["tts-server"]

  boot_disk {
    initialize_params {
      image = var.image_family
      size  = var.boot_disk_size
      type  = "pd-balanced"
    }
  }

  # Attach GPU
  guest_accelerator {
    type  = var.gpu_type
    count = var.gpu_count
  }

  # GPU instances require specific scheduling settings
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
  }

  # Reservation affinity - use ANY_RESERVATION if enabled
  dynamic "reservation_affinity" {
    for_each = var.use_reservation ? [1] : []
    content {
      type = "ANY_RESERVATION"
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.tts_subnet.id
    access_config {
      nat_ip = google_compute_address.tts_ip.address
    }
  }

  metadata = {
    ssh-keys = "${var.ssh_username}:${file(var.public_key_path)}"
  }

  metadata_startup_script = templatefile("${path.module}/startup-script.sh.tpl", {
    gar_helm_registry      = var.gar_helm_registry
    docker_registry        = var.docker_registry
    docker_image_tag       = var.docker_image_tag
    helm_chart_version     = var.helm_chart_version
    tts_model              = var.tts_model
    chunk_size             = var.chunk_size
    huggingface_token      = var.huggingface_token
  })

  # Copy the GAR service account key file to the instance
  provisioner "file" {
    source      = var.gar_service_account_key_path
    destination = "/home/ubuntu/key.json"

    connection {
      type        = "ssh"
      user        = var.ssh_username
      private_key = file(var.private_key_path)
      host        = self.network_interface[0].access_config[0].nat_ip
    }
  }

  service_account {
    email  = var.service_account_email
    scopes = ["cloud-platform"]
  }

  # Allow Terraform to destroy the instance even if it has a GPU attached
  allow_stopping_for_update = true

  lifecycle {
    ignore_changes = [
      metadata_startup_script,
    ]
  }
}
