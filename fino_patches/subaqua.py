"""The unlabeled tropical coastal Sentinel-2 corpus as a FINO dataset.

Shim for `aquafm.adaptation.bundle_dataset`; `aquafm` must be installed in the environment
that runs the clone. Copy to `dinov3/data/datasets/subaqua.py` (see fino_patches/README.md).
"""

from aquafm.adaptation.bundle_dataset import SubAqua, _Metadata, _Split

__all__ = ["SubAqua", "_Metadata", "_Split"]
