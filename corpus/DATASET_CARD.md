# Tropical coastal Sentinel-2 corpus (unlabeled)

Unlabeled Sentinel-2 RGB patches over tropical reefs and shallow coastal water, built for
continued self-supervised training of vision foundation models. It is the corpus used to train
the domain-adapted DINOv3 ViT-B/16 in this repository (see [`../MODEL_CARD.md`](../MODEL_CARD.md)).

| | |
|---|---|
| Download | [Zenodo](https://zenodo.org/records/23313478) |
| License | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (contains modified Copernicus Sentinel data, see Attribution) |
| Size | ~31 GB extracted, 65 tile folders |
| Patches | 188,857 kept of 200,530 scanned (11,673 rejected by the quality filters) |
| Locations | 19,768 distinct patch locations, each imaged on 1–14 dates (mean 9.6) |
| Tiles | 65 MGRS tiles, 30 UTM zones, latitudes 24.2°S to 25.3°N |
| Regions | Belize and the Caribbean, the Red Sea, Southeast Asia, the Great Barrier Reef, and other tropical reef areas |
| Years | 2017–2025 |
| Source | Sentinel-2 Level-2A, bands B4/B3/B2 (true-colour R, G, B) |
| Format | GeoTIFF, 224×224 px, 3 bands, uint8 (8-bit stretched), 10 m GSD (2,240 m per patch) |

## Layout

```
S2_unlabeled/
  <TILE>/<TILE>_<YYYYMMDD>_<location_index>.tif   one patch; same location_index = same footprint
  manifest.csv              one row per kept patch (the training index)
  manifest_rejected.csv     one row per rejected patch, with the reason
  manifest_stats.json       vocabularies, guide cardinalities, normalisation, filter settings
  config_fragment.yaml      training-config values derived from the corpus
```

All four manifest files are in the Zenodo record and go in the corpus root as shown. This
folder of the repository has copies of `manifest_stats.json` and `config_fragment.yaml` only;
the CSVs are distributed with the data. To rebuild them (for example with different filters):

```bash
python scripts/build_bundle_manifest.py --extract <downloads>/*.tar --root $AQUAFM_DATA_ROOT/S2_unlabeled
# or, if already extracted:
python scripts/build_bundle_manifest.py --root $AQUAFM_DATA_ROOT/S2_unlabeled --jobs 12
```

`manifest.csv` columns: `path` (relative to the corpus root), `tile`, `date`, `year`, `month`,
`season` (hemisphere-aware meteorological season), `lon`, `lat` (patch centre, WGS84),
`location_index`, the quality statistics used by the filters (`zero_frac`, `band_zero_frac`,
`saturated_frac`, `band_std`, `weakest_band_std`) and the per-band means (`mean_red`,
`mean_green`, `mean_blue`, in DN).

Normalisation over the kept patches (0–1 units, R, G, B): mean `[0.2135, 0.4032, 0.4288]`,
std `[0.2423, 0.2787, 0.2729]`.

## Quality filters

A patch is rejected if any of the following holds (definitions in
`src/aquafm/adaptation/bundle_corpus.py`):

| Filter | Threshold | Catches |
|---|---|---|
| `zero_frac` | > 0.5 | mostly no-data (all bands 0) |
| `band_zero_frac` | > 0.99 | one band almost entirely zero |
| `saturated_frac` | > 0.5 | mostly saturated (all bands 255) |
| `band_std` | < 5.0 DN | featureless patch |
| `weakest_band_std` | < 3.0 DN, where another band's mean > 30 DN | one band crushed to near-constant by the 8-bit stretch |

The two band-level filters account for 11,145 of the 11,673 rejections. A band crushed by the
8-bit stretch has few exact zeros and the other bands keep the pooled standard deviation high,
so neither the zero-fraction nor the pooled-std test detects it; the weakest band's standard
deviation does. The test applies only when another band is bright, because a dim red band over
dark water is real signal (water absorbs red strongly).

## Intended use and limitations

Self-supervised pretraining or adaptation of encoders for coastal and shallow-water remote
sensing. The corpus has no labels; the metadata (tile, date, location) supports
metadata-aware objectives and geographic splits.

Patches are 8-bit stretched RGB, not calibrated reflectance, and coverage is limited to
tropical latitudes. Many locations repeat across dates, so evaluation splits on the corpus are
made by `location_index` or tile.

## Attribution

Contains modified Copernicus Sentinel data (2017–2025).

## Per-tile counts

Per-tile counts (kept, rejected, locations, dates) are under `per_tile` in
[`manifest_stats.json`](manifest_stats.json).
