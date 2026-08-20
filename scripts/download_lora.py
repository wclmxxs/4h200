#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    cache_root = Path(args.cache_root).resolve()
    hub_cache = cache_root / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo,
            revision=args.revision,
            filename=args.filename,
            cache_dir=hub_cache,
        )
    )
    actual = sha256_file(downloaded)
    if actual != args.sha256:
        raise SystemExit(
            f"LoRA SHA256 mismatch for {downloaded}: expected {args.sha256}, got {actual}"
        )
    relative = downloaded.relative_to(cache_root)
    print(Path("/cache/huggingface") / relative)


if __name__ == "__main__":
    main()
