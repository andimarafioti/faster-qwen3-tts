"""Download voice files to the local directory on startup.

Prefers a cloud-agnostic git repo (codeload tarball) when configured; otherwise
falls back to the legacy GCS download. This lets voices live in version control
(QarlAI/live-assets) for the GCP→Azure migration with no change to inference.
"""

import io
import os
import logging
import tarfile
import urllib.request
from config import load_config

logger = logging.getLogger(__name__)

_CODELOAD = "https://codeload.github.com"


def _download_voices_from_git(repo, branch, token, prefix, voices_dir):
    """Fetch the repo tarball from codeload and extract <prefix>/** into
    voices_dir (mirroring the GCS layout, prefix stripped). Returns file count."""
    url = f"{_CODELOAD}/{repo.strip('/')}/tar.gz/refs/heads/{branch}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    if token:
        req.add_header("Authorization", f"token {token}")

    logger.info(f"Downloading voices from git {repo}@{branch} (prefix '{prefix}') to {voices_dir}")
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - trusted codeload URL
        tar_bytes = resp.read()

    os.makedirs(voices_dir, exist_ok=True)
    voices_root = os.path.realpath(voices_dir)
    wanted = f"{prefix.strip('/')}/"
    count = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Strip the GitHub "<repo>-<sha>/" wrapper dir.
            parts = member.name.split("/", 1)
            if len(parts) != 2 or not parts[1].startswith(wanted):
                continue
            sub = parts[1][len(wanted):]
            if not sub:
                continue
            dest = os.path.realpath(os.path.join(voices_dir, sub))
            if not dest.startswith(voices_root + os.sep):
                logger.warning(f"  Skipping unsafe path: {member.name}")
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as out:
                out.write(f.read())
            count += 1
    logger.info(f"Git voice download complete. Wrote {count} file(s).")
    return count


def init_voices():
    """Download voice files into the local voices directory.

    Source selection (first configured wins):
        1. git repo — voices.git.repo/branch (or VOICE_CACHE_GIT_REPO/BRANCH env),
           token via VOICE_CACHE_GIT_TOKEN; prefix within the repo defaults to
           "voices/qwen3" (the flat layout this server reads).
        2. GCS bucket — voices.gcs.bucket/prefix (or VOICE_CACHE_BUCKET/PREFIX env).

    Skips files that already exist locally with the same size (GCS path).
    Silently returns if neither source is configured.
    """
    config = load_config()
    voices_config = config.get("voices", {})
    gcs_config = voices_config.get("gcs", {})
    git_config = voices_config.get("git", {})
    voices_dir = voices_config.get("dir", "/app/voices")

    # 1. Cloud-agnostic git source (preferred for the GCP→Azure migration).
    git_repo = os.environ.get("VOICE_CACHE_GIT_REPO", git_config.get("repo", ""))
    git_branch = os.environ.get("VOICE_CACHE_GIT_BRANCH", git_config.get("branch", ""))
    if git_repo and git_branch:
        git_token = os.environ.get("VOICE_CACHE_GIT_TOKEN", git_config.get("token", ""))
        git_prefix = os.environ.get("VOICE_CACHE_GIT_PREFIX", git_config.get("prefix", "voices/qwen3"))
        try:
            if _download_voices_from_git(git_repo, git_branch, git_token, git_prefix, voices_dir):
                return
            logger.warning("Git voice source yielded no files; falling back to GCS if configured")
        except Exception as e:
            logger.error(f"Git voice download failed ({e}); falling back to GCS if configured")

    # 2. Legacy GCS source (fallback).
    bucket_name = os.environ.get("VOICE_CACHE_BUCKET", gcs_config.get("bucket", ""))
    prefix = os.environ.get("VOICE_CACHE_PREFIX", gcs_config.get("prefix", ""))

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
