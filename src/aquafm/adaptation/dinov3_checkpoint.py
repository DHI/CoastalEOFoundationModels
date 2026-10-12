"""Convert DINOv3 checkpoints between Meta's released (hub) layout and the FiNo/SSL layout.

Released: a bare backbone state dict. FiNo: `{"teacher": {"backbone.<key>": ...}}`, which
`init_fsdp_model_from_checkpoint` loads non-strictly, so an unconverted file loads nothing.

Usage:

    # released -> FiNo, for student.resume_from_teacher_chkpt
    python -m aquafm.adaptation.dinov3_checkpoint to-fino \
        --input  weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --output weights/fino/dinov3_vitb16_lvd1689m_teacher.pth

    # FiNo output -> released layout, for backbone_kwargs.ckpt_path
    python -m aquafm.adaptation.dinov3_checkpoint to-hub \
        --input  <run>/eval/<iter>/teacher_checkpoint.pth \
        --output weights/dinov3_vitb16_subaqua_adapted.pth
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

# `init_fsdp_model_from_checkpoint` indexes this key unconditionally.
TEACHER_KEY = "teacher"
# Prefix induced by the SSL student/teacher `nn.ModuleDict`.
BACKBONE_PREFIX = "backbone."
# Only in the SAT-493M ViT-L (untied global/local CLS norms).
UNTIED_NORM_KEYS = ("local_cls_norm.weight", "local_cls_norm.bias")


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _unwrap(obj: dict) -> dict:
    """Backbone state dict from either layout."""
    if TEACHER_KEY in obj:
        obj = obj[TEACHER_KEY]
    # Keep only backbone.* keys, dropping the SSL heads of a FiNo teacher.
    if any(k.startswith(BACKBONE_PREFIX) for k in obj):
        return {
            k[len(BACKBONE_PREFIX) :]: v for k, v in obj.items() if k.startswith(BACKBONE_PREFIX)
        }
    return dict(obj)


def to_fino(state: dict) -> dict:
    """Released layout -> `{"teacher": {"backbone.*": ...}}`."""
    return {TEACHER_KEY: {f"{BACKBONE_PREFIX}{k}": v for k, v in state.items()}}


def _report(state: dict) -> None:
    print(f"  backbone tensors: {len(state)}")
    if any(k in state for k in UNTIED_NORM_KEYS):
        print("  local_cls_norm.* present (SAT-493M ViT-L): set "
              "student/teacher.untie_global_and_local_cls_norm: true in the FiNo config")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("direction", choices=["to-fino", "to-hub"])
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    print(f"reading {args.input}")
    state = _unwrap(_load(args.input))
    _report(state)
    print(f"writing {args.direction[3:]} layout -> {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(to_fino(state) if args.direction == "to-fino" else state, args.output)

    if args.direction == "to-hub":
        # torch.hub's ViT-L loader parses an 8-character hash from a local filename.
        digest = hashlib.sha256(args.output.read_bytes()).hexdigest()[:8]
        print(f"  for a ViT-L, rename to {args.output.stem}-{digest}.pth")


if __name__ == "__main__":
    main()
