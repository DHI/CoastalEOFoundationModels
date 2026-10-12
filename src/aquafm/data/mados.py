"""MADOS constants and Lightning datamodule.

MADOS (Marine Debris and Oil Spill; Kikaki et al., ISPRS JPRS 2024) is a sparsely
annotated Sentinel-2 dataset. Reference implementation: https://github.com/gkakogeorgiou/mados;
data: https://doi.org/10.5281/zenodo.10664073 (CC-BY-4.0).
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from terratorch.datamodules import GenericNonGeoSegmentationDataModule

from aquafm.data.augmentation import SeededCompose
from aquafm.data.vscp import vscp_batch

# Band centres (nm) per platform: B1-B8, B8A, B11, B12 in ascending wavelength (ACOLITE
# `rhorc` has no B9/B10). This is the channel order written by `prepare_mados.py`.
# Patches come from S2A or S2B; no scene mixes them.
PLATFORM_WAVELENGTHS = {
    "S2A": [443, 492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202],
    "S2B": [442, 492, 559, 665, 704, 739, 780, 833, 864, 1610, 2186],
}
# Bands are named by their S2A centres (the majority platform; at most 16 nm off for S2B).
WAVELENGTHS = PLATFORM_WAVELENGTHS["S2A"]

# Sentinel-2A FWHM per band (nm), aligned to WAVELENGTHS.
BANDWIDTHS = [20, 65, 35, 30, 15, 15, 20, 115, 20, 90, 180]

# Per-band statistics from the reference implementation (`utils/dataset.py`).
BANDS_MEAN = [
    0.0582676, 0.05223386, 0.04381474, 0.0357083, 0.03412902, 0.03680401,
    0.03999107, 0.03566642, 0.03965081, 0.0267993, 0.01978944,
]
BANDS_STD = [
    0.03240627, 0.03432253, 0.0354812, 0.0375769, 0.03785412, 0.04992323,
    0.05884482, 0.05545856, 0.06423746, 0.04211187, 0.03019115,
]

# Classes 1..15 in the shipped masks, 0..14 after preparation.
CLASS_NAMES = [
    "Marine Debris",
    "Dense Sargassum",
    "Sparse Floating Algae",
    "Natural Organic Material",
    "Ship",
    "Oil Spill",
    "Marine Water",
    "Sediment-Laden Water",
    "Foam",
    "Turbid Water",
    "Shallow Water",
    "Waves & Wakes",
    "Oil Platform",
    "Jellyfish",
    "Sea snot",
]

# Share of annotated pixels per class, from the reference implementation (`utils/dataset.py`).
CLASS_DISTRIBUTION = [
    0.00336, 0.00241, 0.00336, 0.00142, 0.00775, 0.18452, 0.34775, 0.20638,
    0.00062, 0.1169, 0.09188, 0.01309, 0.00917, 0.00176, 0.00963,
]

NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = -1  # unannotated pixels
PATCH_SIZE = 240  # px
GSD = 10  # m, after upsampling the 20 m and 60 m bands

# Zenodo record totals.
NUM_SCENES = 174
NUM_PATCHES = 2803

# The constant c in the reference class weights 1 / log(c + p), as passed by
# `marinext/train.py --weight_param`.
REFERENCE_WEIGHT_PARAM = 1.03


def class_weights_from_distribution(smoothing: float = REFERENCE_WEIGHT_PARAM) -> list[float]:
    """Class weights 1 / log(c + p), as in the reference `gen_weights`."""
    return [1.0 / math.log(smoothing + p) for p in CLASS_DISTRIBUTION]


def bandwidths_for(output_bands: Sequence[int]) -> list[float]:
    """S2A FWHM for the bands with the given centres."""
    return [BANDWIDTHS[WAVELENGTHS.index(band)] for band in output_bands]


def statistics_for(output_bands: Sequence[int]) -> tuple[list[float], list[float]]:
    """Reference means and stds for the bands with the given centres."""
    positions = [WAVELENGTHS.index(band) for band in output_bands]
    return [BANDS_MEAN[i] for i in positions], [BANDS_STD[i] for i in positions]


class MadosDataModule(GenericNonGeoSegmentationDataModule):
    """MADOS as written by `scripts/prepare_mados.py`.

    `output_bands` selects bands by centre wavelength, in the order the model receives
    them. `means`/`stds` default to the reference statistics. `num_channels`,
    `band_wavelengths`, `band_bandwidths`, `class_names` and `class_weights` are read by
    the model configs and checked here against the tables above.
    """

    def __init__(
        self,
        data_root: str | Path,
        num_channels: int,
        output_bands: Sequence[int],
        band_bandwidths: Sequence[float],
        band_wavelengths: Sequence[int] | None = None,
        means: Sequence[float] | None = None,
        stds: Sequence[float] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        img_size: int = PATCH_SIZE,
        num_classes: int = NUM_CLASSES,
        augment: bool = True,
        dataset_dir: str = "MADOS_ml",
        drop_last: bool = True,
        pin_memory: bool = False,
        allow_incomplete: bool = False,
        class_names: Sequence[str] | None = None,
        class_weights: Sequence[float] | None = None,
        class_weight_smoothing: float = REFERENCE_WEIGHT_PARAM,
        vscp: bool = False,
    ):
        root = Path(data_root) / dataset_dir
        self._check_prepared(root, allow_incomplete)

        output_bands = list(output_bands)
        if unknown := [band for band in output_bands if band not in WAVELENGTHS]:
            raise ValueError(f"output_bands {unknown} are not MADOS band centres; use values from {WAVELENGTHS}")
        if len(set(output_bands)) != len(output_bands):
            raise ValueError(f"output_bands has duplicates: {output_bands}")
        if num_channels != len(output_bands):
            raise ValueError(f"num_channels={num_channels} must equal len(output_bands)={len(output_bands)}")
        if img_size != PATCH_SIZE:
            raise ValueError(f"img_size={img_size}, but MADOS patches are {PATCH_SIZE} px; set img_size={PATCH_SIZE}")
        if num_classes != NUM_CLASSES:
            raise ValueError(f"num_classes={num_classes}, but MADOS has {NUM_CLASSES}")
        band_wavelengths = output_bands if band_wavelengths is None else list(band_wavelengths)
        if band_wavelengths != output_bands:
            raise ValueError(f"band_wavelengths {band_wavelengths} must equal output_bands {output_bands}")
        if list(band_bandwidths) != bandwidths_for(output_bands):
            raise ValueError(f"band_bandwidths must be {bandwidths_for(output_bands)} for output_bands {output_bands}")
        if class_names is not None and list(class_names) != CLASS_NAMES:
            raise ValueError(f"class_names must be the MADOS classes in order: {CLASS_NAMES}")
        if class_weights is not None:
            expected = class_weights_from_distribution(class_weight_smoothing)
            if len(class_weights) != NUM_CLASSES or any(
                abs(a - b) >= 1e-3 for a, b in zip(class_weights, expected)
            ):
                raise ValueError(
                    f"class_weights do not match class_weights_from_distribution({class_weight_smoothing}); "
                    f"regenerate them or set class_weights=null"
                )

        default_means, default_stds = statistics_for(output_bands)
        means = default_means if means is None else list(means)
        stds = default_stds if stds is None else list(stds)
        if len(means) != num_channels or len(stds) != num_channels:
            raise ValueError(f"means and stds need {num_channels} entries, one per output band")

        # Train augmentation of the reference (`utils/dataset.py::join_transform`): one flip
        # drawn from [h, v, both, none, none], so P(flip) = 0.6, then a `np.rot90` that
        # RandomRotate90(p=0.8) matches.
        flips_and_rotations = [
            A.OneOf(
                [
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.Sequential([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)], p=1.0),
                ],
                p=0.6,
            ),
            A.RandomRotate90(p=0.8),
        ]
        to_tensor = [ToTensorV2()]

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            train_data_root=root / "train" / "images",
            train_label_data_root=root / "train" / "labels",
            val_data_root=root / "val" / "images",
            val_label_data_root=root / "val" / "labels",
            test_data_root=root / "test" / "images",
            test_label_data_root=root / "test" / "labels",
            img_grep="*.tif",
            label_grep="*.tif",
            dataset_bands=list(WAVELENGTHS),
            output_bands=output_bands,
            means=means,
            stds=stds,
            train_transform=SeededCompose((flips_and_rotations + to_tensor) if augment else list(to_tensor)),
            val_transform=list(to_tensor),
            test_transform=list(to_tensor),
            # Preparation already wrote -1 for unannotated pixels.
            reduce_zero_label=False,
            no_label_replace=IGNORE_INDEX,
            drop_last=drop_last,
            pin_memory=pin_memory,
        )
        self.root = root
        self.num_channels = num_channels
        self.band_wavelengths = band_wavelengths
        self.band_bandwidths = list(band_bandwidths)
        self.img_size = img_size
        self.augment = augment
        self.class_names = list(class_names) if class_names is not None else None
        self.class_weights = list(class_weights) if class_weights is not None else None
        self.class_weight_smoothing = class_weight_smoothing
        self.vscp = vscp

    @staticmethod
    def _check_prepared(root: Path, allow_incomplete: bool) -> None:
        """Raise if `root` is not a complete output of `prepare_mados.py`."""
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                if not (root / split / kind).is_dir():
                    raise FileNotFoundError(
                        f"{root / split / kind} not found; run "
                        f"`python scripts/prepare_mados.py --path <raw MADOS> --out {root}`"
                    )
        stats_path = root / "stats.json"
        if stats_path.exists() and not json.loads(stats_path.read_text()).get("complete", True) \
                and not allow_incomplete:
            raise ValueError(
                f"{root} was prepared from a subset (--limit or --allow-subset); "
                f"re-run prepare_mados.py on the full data or pass allow_incomplete=true"
            )

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        for name in ("train", "val", "test"):
            dataset = getattr(self, f"{name}_dataset", None)
            if dataset is not None and len(dataset) == 0:
                raise ValueError(f"{name} split under {self.root} is empty; re-run prepare_mados.py")

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int) -> dict:
        """Normalize, then apply VSCP to training batches (after normalization, as in the reference)."""
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        if self.vscp and self.trainer is not None and self.trainer.training:
            batch = vscp_batch(batch)
        return batch
