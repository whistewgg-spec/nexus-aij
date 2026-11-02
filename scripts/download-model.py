#!/usr/bin/env python3
"""Download the pinned public Qwen Coder GGUF and verify its known SHA-256."""

import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF"
REVISION = "7c137640ef0332dfedb229f2504c58d83ed4307a"
FILENAME = "Qwen2.5-Coder-3B-Instruct-Q4_K_M.gguf"
EXPECTED_SHA256 = "3da3afe6cf5c674ac195803ea0dd6fee7e1c228c2105c1ce8c66890d1d4ab460"
EXPECTED_BYTES = 1_929_903_360
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model:
        while chunk := model.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    destination = Path(
        os.environ.get(
            "NEXUS_MODEL_PATH",
            Path(__file__).resolve().parents[1] / "data" / "models" / FILENAME,
        )
    ).expanduser()
    if destination.is_file() and destination.stat().st_size == EXPECTED_BYTES:
        existing_hash = sha256_file(destination)
        if existing_hash == EXPECTED_SHA256:
            print(f"Verified model already present: {destination}")
            return
        print("Existing model failed SHA-256 verification; downloading a clean copy.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            remote_size = int(response.headers.get("Content-Length", "0"))
            download_url = response.geturl()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Could not check the model download source: {exc}") from exc
    if remote_size != EXPECTED_BYTES:
        raise SystemExit(
            f"Model size mismatch at pinned revision: expected {EXPECTED_BYTES}, got {remote_size}."
        )

    digest = hashlib.sha256()
    temporary_path = None
    try:
        with urllib.request.urlopen(download_url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, delete=False
            ) as output:
                temporary_path = Path(output.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        if total != EXPECTED_BYTES:
            raise RuntimeError(f"Model size mismatch: expected {EXPECTED_BYTES}, got {total}.")
        actual_hash = digest.hexdigest()
        if actual_hash.lower() != EXPECTED_SHA256:
            raise RuntimeError(
                f"Model SHA-256 mismatch: expected {EXPECTED_SHA256}, received {actual_hash}."
            )
        temporary_path.replace(destination)
        print(f"Model ready: {destination} ({total / 1024**2:.0f} MiB)")
        print(f"SHA-256 verified: {actual_hash}")
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
        if temporary_path:
            temporary_path.unlink(missing_ok=True)
        raise SystemExit(f"Model download/integrity check failed: {exc}") from exc


if __name__ == "__main__":
    main()
