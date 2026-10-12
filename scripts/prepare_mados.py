#!/usr/bin/env python
"""Convert raw MADOS (Zenodo layout) into per-split image/label GeoTIFFs for TerraTorch.

    python scripts/prepare_mados.py --path $AQUAFM_DATA_ROOT/MADOS --out $AQUAFM_DATA_ROOT/MADOS_ml

Writes <out>/{train,val,test}/images/<roi>.tif (11 bands, float32, 240x240 at 10 m, NaN
replaced by the reference band mean), <out>/{train,val,test}/labels/<roi>.tif (int16,
-1 unannotated, 0..14 classes) and <out>/stats.json. Images and labels share basenames,
since TerraTorch pairs them by sort order.
"""

import argparse
import json
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

from aquafm.data.mados import (
    BANDS_MEAN,
    BANDS_STD,
    CLASS_DISTRIBUTION,
    CLASS_NAMES,
    GSD,
    IGNORE_INDEX,
    NUM_CLASSES,
    NUM_PATCHES,
    NUM_SCENES,
    PATCH_SIZE,
    PLATFORM_WAVELENGTHS,
    WAVELENGTHS,
)

# MADOS rasters have no geotransform; rasterio's `out_shape` read raises a numpy deprecation.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
warnings.filterwarnings("ignore", message="Setting the shape on a NumPy array")

PLATFORM_BY_WAVELENGTHS = {tuple(v): k for k, v in PLATFORM_WAVELENGTHS.items()}
BAND_PATTERN = re.compile(r"_L2R_rhorc_(\d+)_(.+)\.tif$")
MASK_PATTERN = re.compile(r"_L2R_cl_(.+)\.tif$")
SPLITS = ("train", "val", "test")


def read_splits(root: Path) -> dict[str, set[str]]:
    splits = {}
    for split in SPLITS:
        path = root / "splits" / f"{split}_X.txt"
        if not path.exists():
            raise SystemExit(f"missing split file {path}; --path must point at the unzipped MADOS root")
        splits[split] = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    return splits


def index_rois(root: Path) -> dict[str, dict]:
    """Map ROI id ("Scene_0_1") to its 10 m mask and {band centre: path}."""
    rois: dict[str, dict] = defaultdict(lambda: {"mask": None, "bands": {}})
    for scene_dir in sorted(root.glob("Scene_*")):
        for resolution_dir in sorted(scene_dir.glob("*")):
            if not resolution_dir.is_dir() or not resolution_dir.name.isdigit():
                continue
            for path in resolution_dir.glob("*.tif"):
                if match := BAND_PATTERN.search(path.name):
                    rois[f"{scene_dir.name}_{match.group(2)}"]["bands"][int(match.group(1))] = path
                elif (match := MASK_PATTERN.search(path.name)) and resolution_dir.name == str(GSD):
                    # The 20 m folder also has aggregated `_cl` masks; the label is the 10 m one.
                    rois[f"{scene_dir.name}_{match.group(1)}"]["mask"] = path
    return dict(rois)


def verify(root: Path, splits: dict[str, set[str]], rois: dict[str, dict], allow_subset: bool) -> dict[str, str]:
    """Exit on an incomplete or malformed download; return the ROI -> platform map."""
    problems = []
    listed = set().union(*splits.values())
    total_listed = sum(len(v) for v in splits.values())
    if not allow_subset:
        scenes = {p.name for p in root.glob("Scene_*") if p.is_dir()}
        if scenes != {f"Scene_{i}" for i in range(NUM_SCENES)}:
            problems.append(f"found {len(scenes)} scene folders, expected Scene_0..Scene_{NUM_SCENES - 1}")
        if len(rois) != NUM_PATCHES or total_listed != NUM_PATCHES:
            problems.append(f"found {len(rois)} patches and {total_listed} split entries, expected {NUM_PATCHES}")
    if total_listed != len(listed):
        problems.append("split files overlap")
    if unlisted := set(rois) - listed:
        problems.append(f"{len(unlisted)} patches are in no split, e.g. {sorted(unlisted)[:5]}")
    if unfound := listed - set(rois):
        problems.append(f"{len(unfound)} split entries have no files, e.g. {sorted(unfound)[:5]}")

    platforms = {}
    for roi, entry in sorted(rois.items()):
        if entry["mask"] is None:
            problems.append(f"{roi}: no 10 m _cl mask")
        if (platform := PLATFORM_BY_WAVELENGTHS.get(tuple(sorted(entry["bands"])))) is None:
            problems.append(f"{roi}: band centres {sorted(entry['bands'])} match neither S2A nor S2B")
        else:
            platforms[roi] = platform
    if problems:
        raise SystemExit("MADOS verification failed:\n  " + "\n  ".join(problems[:20]))
    return platforms


def load_image(entry: dict, resampling: Resampling) -> tuple[np.ndarray, float]:
    """Return the 11-band stack on the 10 m grid with NaN imputed, and its NaN fraction."""
    stack = np.empty((len(WAVELENGTHS), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    nan_count = 0
    # Ordered by ascending band centre, since S2B centres differ from the S2A names.
    for index, wavelength in enumerate(sorted(entry["bands"])):
        path = entry["bands"][wavelength]
        with rasterio.open(path) as src:
            expected = PATCH_SIZE // (int(path.parent.name) // GSD)
            if (src.height, src.width) != (expected, expected):
                raise SystemExit(f"{path}: expected {expected}x{expected} px, got {src.height}x{src.width}")
            band = src.read(1, out_shape=(PATCH_SIZE, PATCH_SIZE), resampling=resampling)
        # NaN becomes the band mean (0 after standardization), as in the reference loader.
        nan = np.isnan(band)
        nan_count += int(nan.sum())
        band[nan] = BANDS_MEAN[index]
        stack[index] = band
    return stack, nan_count / stack.size


def load_mask(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        raw = src.read(1)
        profile = src.profile
    if raw.shape != (PATCH_SIZE, PATCH_SIZE) or raw.min() < 0 or raw.max() > NUM_CLASSES:
        raise SystemExit(f"{path}: expected a {PATCH_SIZE}x{PATCH_SIZE} mask with values 0..{NUM_CLASSES}")
    # Cast before subtracting: the shipped dtype is unsigned.
    return raw.astype(np.int16) - 1, profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=Path, required=True, help="raw MADOS root (unzipped)")
    parser.add_argument("--out", type=Path, help="output root; not needed with --verify-only")
    parser.add_argument("--resampling", default="nearest", choices=["nearest", "bilinear"],
                        help="upsampling of the 20 m and 60 m bands (the reference uses nearest)")
    parser.add_argument("--limit", type=int, help="at most this many patches per split, for quick checks")
    parser.add_argument("--verify-only", action="store_true", help="check the download and exit")
    parser.add_argument("--overwrite", action="store_true", help="delete existing output tifs first")
    parser.add_argument("--allow-subset", action="store_true", help="skip the published scene/patch totals")
    args = parser.parse_args()

    splits = read_splits(args.path)
    rois = index_rois(args.path)
    platforms = verify(args.path, splits, rois, args.allow_subset)
    with rasterio.open(rois[min(rois)]["mask"]) as src:
        mask_dtype = src.dtypes[0]
    print(f"Verified {len(rois)} patches, masks {mask_dtype}, platforms {dict(sorted(Counter(platforms.values()).items()))}")
    if args.verify_only:
        return
    if args.out is None:
        parser.error("--out is required unless --verify-only")

    for split in SPLITS:
        for kind in ("images", "labels"):
            directory = args.out / split / kind
            directory.mkdir(parents=True, exist_ok=True)
            existing = list(directory.glob("*.tif"))
            if existing and not args.overwrite:
                raise SystemExit(f"{directory} is not empty; pass --overwrite or choose another --out")
            for path in existing:
                path.unlink()

    resampling = getattr(Resampling, args.resampling)
    band_sum = np.zeros(len(WAVELENGTHS), dtype=np.float64)
    band_sq_sum = np.zeros(len(WAVELENGTHS), dtype=np.float64)
    pixel_count = 0
    record = {}

    for split in SPLITS:
        selected = sorted(splits[split] & set(rois))[: args.limit]
        class_pixels = np.zeros(NUM_CLASSES, dtype=np.int64)
        nan_fractions, unannotated_patches = [], 0

        for roi in selected:
            mask, profile = load_mask(rois[roi]["mask"])
            image, nan_fraction = load_image(rois[roi], resampling)
            nan_fractions.append(nan_fraction)
            annotated = mask >= 0
            unannotated_patches += int(not annotated.any())
            class_pixels += np.bincount(mask[annotated], minlength=NUM_CLASSES)
            if split == "train":
                image64 = image.astype(np.float64)
                band_sum += image64.sum(axis=(1, 2))
                band_sq_sum += (image64**2).sum(axis=(1, 2))
                pixel_count += PATCH_SIZE * PATCH_SIZE

            common = {**profile, "driver": "GTiff", "compress": "deflate"}
            with rasterio.open(args.out / split / "images" / f"{roi}.tif", "w",
                               **{**common, "count": len(WAVELENGTHS), "dtype": "float32", "nodata": None}) as dst:
                dst.write(image)
                dst.update_tags(wavelengths_nm=",".join(map(str, WAVELENGTHS)))
                for index, wavelength in enumerate(WAVELENGTHS, 1):
                    dst.set_band_description(index, f"{wavelength}nm")
            with rasterio.open(args.out / split / "labels" / f"{roi}.tif", "w",
                               **{**common, "count": 1, "dtype": "int16", "nodata": IGNORE_INDEX}) as dst:
                dst.write(mask, 1)

        annotated_total = int(class_pixels.sum())
        majority = int(class_pixels.argmax())
        record[split] = {
            "patches": len(selected),
            "annotated_pixels": annotated_total,
            "annotated_fraction": annotated_total / (len(selected) * PATCH_SIZE**2),
            "patches_with_no_annotation": unannotated_patches,
            "mean_nan_fraction": float(np.mean(nan_fractions)),
            "max_nan_fraction": float(np.max(nan_fractions)),
            "class_pixels": dict(zip(CLASS_NAMES, class_pixels.tolist())),
            "majority_class": CLASS_NAMES[majority],
            "majority_class_baseline_accuracy": float(class_pixels[majority] / annotated_total),
            "majority_class_baseline_macro_miou": float(class_pixels[majority] / annotated_total / NUM_CLASSES),
            "platforms": dict(sorted(Counter(platforms[roi] for roi in selected).items())),
        }
        print(f"{split}: {len(selected)} patches, {record[split]['annotated_fraction']:.2%} annotated")

    means = band_sum / pixel_count
    stds = np.sqrt(np.clip(band_sq_sum / pixel_count - means**2, 1e-12, None))
    divergence = np.abs(means - np.array(BANDS_MEAN)) / np.array(BANDS_MEAN)
    stats = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(args.path),
        "resampling": args.resampling,
        "limit": args.limit,
        "published_totals_enforced": not args.allow_subset,
        # Read by MadosDataModule, which refuses incomplete preparations by default.
        "complete": args.limit is None and not args.allow_subset,
        "mask_dtype_as_shipped": mask_dtype,
        "patch_size": PATCH_SIZE,
        "canonical_wavelengths_nm": WAVELENGTHS,
        "platform_wavelengths_nm": PLATFORM_WAVELENGTHS,
        "splits": record,
        "train_band_means": means.tolist(),
        "train_band_stds": stds.tolist(),
        "reference_band_means": BANDS_MEAN,
        "reference_band_stds": BANDS_STD,
        "max_mean_divergence_from_reference": float(divergence.max()),
        "reference_class_distribution": dict(zip(CLASS_NAMES, CLASS_DISTRIBUTION)),
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Train band means differ from the reference by at most {divergence.max():.1%}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
