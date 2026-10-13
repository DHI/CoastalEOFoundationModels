"""Download the domain-adapted DINOv3 ViT-B/16 checkpoint from the GitHub release and verify its SHA-256.

    python scripts/download_weights.py                 # into $AQUAFM_WEIGHTS_ROOT (default ./weights)
    python scripts/download_weights.py --url <mirror>

Meta's DINOv3 weights are gated and downloaded by hand; see README.md.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/DHI/CoastalEOFoundationModels/releases/download/v0.1.0/dinov3_vitb16_subaqua_adapted.pth"
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
    parser.add_argument("--url", default=URL, help="download location")
    parser.add_argument("--out-dir", type=Path, default=Path(os.environ.get("AQUAFM_WEIGHTS_ROOT", "weights")))
    args = parser.parse_args()

    target = args.out_dir / FILENAME
    if target.exists() and sha256_of(target) == SHA256:
        print(f"{target} already present and verified")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".part")
    print(f"Downloading {args.url} -> {target}")
    urllib.request.urlretrieve(args.url, partial)
    actual = sha256_of(partial)
    if actual != SHA256:
        partial.unlink()
        sys.exit(f"SHA-256 mismatch: expected {SHA256}, got {actual}")
    partial.replace(target)
    print(f"Verified {target}")


if __name__ == "__main__":
    main()
