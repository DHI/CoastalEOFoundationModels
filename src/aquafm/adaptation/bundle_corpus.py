"""Layout, quality filters and metadata of the unlabeled Sentinel-2 RGB patch corpus.

224x224 uint8 RGB patches, one directory per MGRS tile, indexed by the manifest files that
`scripts/build_bundle_manifest.py` writes into the corpus root.
"""

import csv
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

PATCH_SIZE = 224
NUM_CHANNELS = 3
CHANNEL_NAMES = ("red", "green", "blue")  # B4, B3, B2
GSD = 10  # m

# Patches are 8-bit contrast-stretched quicklooks: 0 (deep water in red) and 255 (cloud)
# are valid values.
SATURATION_VALUE = 255

# `{TILE}/{TILE}_{YYYYMMDD}_{location_index}.tif`
PATCH_NAME_RE = re.compile(r"^(?P<tile>\d{2}[A-Z]{3})_(?P<date>\d{8})_(?P<index>\d+)$")

MANIFEST_NAME = "manifest.csv"
REJECTED_NAME = "manifest_rejected.csv"
STATS_NAME = "manifest_stats.json"
FRAGMENT_NAME = "config_fragment.yaml"

MANIFEST_COLUMNS = (
    "path",  # relative to the corpus root
    "tile",
    "date",  # YYYYMMDD
    "year",
    "month",
    "season",
    "lon",  # patch centre, EPSG:4326
    "lat",
    "location_index",
    "zero_frac",
    "band_zero_frac",
    "saturated_frac",
    "band_std",
    "weakest_band_std",
    "mean_red",
    "mean_green",
    "mean_blue",
)

REJECTED_COLUMNS = ("path", "tile", "date", "location_index", "reason", "zero_frac",
                    "band_zero_frac", "saturated_frac", "band_std", "weakest_band_std")

# Each is a `_Metadata` field and a guide name in the training config; FiNo reads guide
# labels with `getattr(metadata, guide.name)`.
GUIDE_FIELDS = ("tile", "year", "season")

# Quality filters. A patch is rejected if any of these hold.
# Fraction of pixels zero in all bands (no data). Deep water alone zeroes red and blue only.
MAX_ZERO_FRAC = 0.5
# Largest fraction of zeros in any single band (a zero-filled channel).
MAX_BAND_ZERO_FRAC = 0.99
# Fraction of pixels 255 in all bands (cloud).
MAX_SATURATED_FRAC = 0.5
# Std over all bands and pixels (featureless patch).
MIN_BAND_STD = 5.0
# Std of the weakest band (a channel crushed to a few DN), applied only when another band's
# mean exceeds DEAD_BAND_SCENE_MEAN, since over dark water a dim red band is real signal.
MIN_WEAKEST_BAND_STD = 3.0
DEAD_BAND_SCENE_MEAN = 30.0

SEASONS = ("winter", "spring", "summer", "autumn")


def parse_patch_name(name: str) -> tuple[str, str, int]:
    """`"15MYV_20180112_7"` -> `("15MYV", "20180112", 7)`."""
    if (match := PATCH_NAME_RE.match(name)) is None:
        raise ValueError(f"{name!r} is not {{TILE}}_{{YYYYMMDD}}_{{index}}")
    return match["tile"], match["date"], int(match["index"])


def season_of(month: int, lat: float) -> str:
    """Meteorological season, flipped for the southern hemisphere."""
    northern = SEASONS[((month % 12) // 3)]
    if lat >= 0:
        return northern
    return SEASONS[(SEASONS.index(northern) + 2) % 4]


@dataclass(frozen=True)
class Patch:
    """One manifest row, typed. `path` is relative to the corpus root."""

    path: str
    tile: str
    date: str
    year: int
    month: int
    season: str
    lon: float
    lat: float
    location_index: int
    zero_frac: float
    band_zero_frac: float
    saturated_frac: float
    band_std: float
    weakest_band_std: float
    mean_red: float
    mean_green: float
    mean_blue: float


def read_manifest(root: Path) -> list[Patch]:
    """Load `<root>/manifest.csv`, which lists the patches that passed the quality filters."""
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {root}; build it with "
                                f"`python scripts/build_bundle_manifest.py --root {root}`.")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and tuple(rows[0]) != MANIFEST_COLUMNS:
        raise ValueError(f"{path} has columns {tuple(rows[0])}, expected {MANIFEST_COLUMNS}; "
                         "rebuild it with scripts/build_bundle_manifest.py.")
    types = {f.name: f.type for f in fields(Patch)}
    return [Patch(**{key: types[key](value) for key, value in row.items()}) for row in rows]


def read_stats(root: Path) -> dict:
    """Load `<root>/manifest_stats.json`."""
    path = Path(root) / STATS_NAME
    if not path.exists():
        raise FileNotFoundError(f"No {STATS_NAME} in {root}; build it with "
                                f"`python scripts/build_bundle_manifest.py --root {root}`.")
    return json.loads(path.read_text())


def guide_vocabularies(stats: dict) -> dict[str, list]:
    """Guide vocabularies; a value's list position is its integer label."""
    return {field: stats["vocabularies"][field] for field in GUIDE_FIELDS}
