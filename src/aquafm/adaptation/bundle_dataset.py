"""`SubAqua`: the unlabeled Sentinel-2 corpus as a FiNo guided-SSL dataset.

Samples are `(image, (label, _Metadata))` with integer guide labels from the vocabularies in
`manifest_stats.json`. The dinov3 clone imports it through `fino_patches/subaqua.py`.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from aquafm.adaptation.bundle_corpus import (
    GUIDE_FIELDS,
    guide_vocabularies,
    read_manifest,
    read_stats,
)

# Coordinate normalization from FiNo's FMoW dataset.
_LAT_MIN, _LAT_MAX = -90.0, 90.0
_LON_MIN, _LON_MAX = -180.0, 180.0

# FiNo discards the class label but expects a `(label, metadata)` target.
_NO_LABEL = 0


@dataclass
class _Metadata:
    """Per-sample guide labels, named as the guides in the config; vocabulary indices."""

    tile: int  # index into vocabularies["tile"]
    year: int  # index into vocabularies["year"]
    season: int  # index into vocabularies["season"]
    month: int  # index into vocabularies["month"]
    coordinates: tuple[float, float]  # (lat, lon) normalized to [0, 1]


class _Split(Enum):
    """Only TRAIN exists."""

    TRAIN = "train"


def _standalone_base():
    """dinov3's `ExtendedVisionDataset` if the clone is importable, else the vendored copy."""
    try:
        from dinov3.data.datasets.extended import ExtendedVisionDataset
    except ModuleNotFoundError:
        from aquafm.adaptation.vision_dataset import ExtendedVisionDataset
    return ExtendedVisionDataset


class SubAqua(_standalone_base()):
    """The unlabeled Sentinel-2 patch corpus, indexed by `<root>/manifest.csv`.

    With `with_metadata=False`, targets are plain labels (for runs without guide heads).
    """

    Split = _Split

    def __init__(
        self,
        *,
        root: str,
        split: _Split = _Split.TRAIN,
        with_metadata: bool = True,
        **kwargs: Any,
    ) -> None:
        if not isinstance(split, _Split):
            raise TypeError(f"split must be a {_Split.__name__}, got {split!r}")
        self._split = split
        self._with_metadata = with_metadata
        super().__init__(root=str(root), **kwargs)

        patches = read_manifest(Path(root))
        if not patches:
            raise ValueError(f"{root}/manifest.csv is empty; rebuild it with "
                             "scripts/build_bundle_manifest.py")
        stats = read_stats(Path(root))
        vocabularies = guide_vocabularies(stats)

        # Columnar numpy arrays keep dataloader workers' copy-on-write memory shared.
        index_of = {field: {value: i for i, value in enumerate(vocabularies[field])}
                    for field in GUIDE_FIELDS}
        month_index_of = {value: i for i, value in enumerate(stats["vocabularies"]["month"])}
        self._guide_cardinalities = {field: len(vocabularies[field]) for field in GUIDE_FIELDS}

        self._paths = tuple(patch.path for patch in patches)
        self._tile = np.array([index_of["tile"][p.tile] for p in patches], dtype=np.int32)
        self._year = np.array([index_of["year"][p.year] for p in patches], dtype=np.int32)
        self._season = np.array([index_of["season"][p.season] for p in patches], dtype=np.int32)
        self._month = np.array([month_index_of[p.month] for p in patches], dtype=np.int32)
        self._coordinates = np.array(
            [
                (
                    (p.lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN),
                    (p.lon - _LON_MIN) / (_LON_MAX - _LON_MIN),
                )
                for p in patches
            ],
            dtype=np.float32,
        )

    @property
    def split(self) -> _Split:
        return self._split

    @property
    def guide_cardinalities(self) -> dict[str, int]:
        """The `n_outputs` each guide head in the training config needs."""
        return dict(self._guide_cardinalities)

    def get_image_data(self, index: int) -> bytes:
        """Raw file bytes; dinov3's `ImageDataDecoder` opens the GeoTIFFs with PIL."""
        with open(os.path.join(self.root, self._paths[index]), "rb") as handle:
            return handle.read()

    def get_target(self, index: int) -> Any:
        if not self._with_metadata:
            return _NO_LABEL
        # Python scalars, so collated dtypes match FiNo's FMoW dataset.
        return _NO_LABEL, _Metadata(
            tile=int(self._tile[index]),
            year=int(self._year[index]),
            season=int(self._season[index]),
            month=int(self._month[index]),
            coordinates=(float(self._coordinates[index][0]), float(self._coordinates[index][1])),
        )

    def __len__(self) -> int:
        return len(self._paths)
