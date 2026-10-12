"""Download the domain-adapted DINOv3 ViT-B/16 checkpoint from Hugging Face and verify its SHA-256.

    python scripts/download_weights.py                 # into $AQUAFM_WEIGHTS_ROOT (default ./weights)
    python scripts/download_weights.py --url <mirror>  # plain HTTPS mirror instead of the Hub

Meta's DINOv3 weights are gated and downloaded by hand; see README.md.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "<HF_REPO_ID>"
FILENAME = "dinov3_vitb16_subaqua_adapted.pth"
SHA256 = "5aef9a9bba55e32bdca00ef34f494c4c93f5b5718dd632249b3002289899c0b3"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", default=REPO_ID, help="Hugging Face model repo")
    parser.add_argument("--url", help="plain HTTPS mirror, used instead of the Hub")
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get("AQUAFM_WEIGHTS_ROOT", "weights")))
    args = parser.parse_args()

    target = args.out_dir / FILENAME
    if target.exists() and sha256_of(target) == SHA256:
        print(f"{target} already present and verified")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.url:
        downloaded = target.with_suffix(".part")
        print(f"Downloading {args.url} -> {target}")
        urllib.request.urlretrieve(args.url, downloaded)
    else:
        print(f"Downloading {args.repo_id}/{FILENAME} -> {target}")
        downloaded = Path(hf_hub_download(args.repo_id, FILENAME, local_dir=args.out_dir, force_download=True))
    actual = sha256_of(downloaded)
    if actual != SHA256:
        downloaded.unlink()
        sys.exit(f"SHA-256 mismatch: expected {SHA256}, got {actual}")
    downloaded.replace(target)
    print(f"Verified {target}")


if __name__ == "__main__":
    main()
