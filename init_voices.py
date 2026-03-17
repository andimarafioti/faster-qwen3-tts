"""Download voice files from GCS to local directory on startup."""

import os
import logging
from config import load_config

logger = logging.getLogger(__name__)


def init_voices():
    """Download voice files from GCS bucket to local voices directory.

    Reads configuration from config.yaml under voices.gcs:
        bucket: GCS bucket name
        prefix: Prefix/path within the bucket

    Skips files that already exist locally with the same size.
    Silently returns if GCS is not configured.
    """
    config = load_config()
    voices_config = config.get("voices", {})
    gcs_config = voices_config.get("gcs", {})

    bucket_name = os.environ.get("VOICE_CACHE_BUCKET", gcs_config.get("bucket", ""))
    prefix = os.environ.get("VOICE_CACHE_PREFIX", gcs_config.get("prefix", ""))
    voices_dir = voices_config.get("dir", "/app/voices")

    if not bucket_name or not prefix:
        logger.info("GCS not configured (voices.gcs.bucket/prefix not set in config), skipping download")
        return

    try:
        from google.cloud import storage
        client = storage.Client()
    except Exception as e:
        logger.error(f"Failed to create GCS client: {e}")
        return

    logger.info(f"Downloading voices from gs://{bucket_name}/{prefix}/ to {voices_dir}")
    os.makedirs(voices_dir, exist_ok=True)

    bucket = client.bucket(bucket_name)
    blobs = [b for b in bucket.list_blobs(prefix=f"{prefix}/") if not b.name.endswith("/")]

    if not blobs:
        logger.warning(f"No voice files found in gs://{bucket_name}/{prefix}/")
        return

    downloaded = 0
    for blob in blobs:
        filename = blob.name[len(prefix):].lstrip("/")
        if not filename:
            continue

        dest_path = os.path.join(voices_dir, filename)

        # Skip if already exists with same size
        if os.path.exists(dest_path) and os.path.getsize(dest_path) == blob.size:
            logger.info(f"  Skipping {filename} (already exists, same size)")
            continue

        logger.info(f"  Downloading {blob.name} ({blob.size} bytes) -> {dest_path}")
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        downloaded += 1

    logger.info(f"GCS download complete. Downloaded {downloaded} file(s), {len(blobs) - downloaded} skipped.")


if __name__ == "__main__":
    init_voices()
