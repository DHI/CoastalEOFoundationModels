import json

import hydra
import numpy as np
import pytest
import rasterio
import torch
from conftest import REPO
from omegaconf import OmegaConf
from rasterio.transform import Affine

from aquafm.data.mados import (
    BANDS_MEAN,
    GSD,
    NUM_CLASSES,
    PATCH_SIZE,
    PLATFORM_WAVELENGTHS,
    WAVELENGTHS,
    class_weights_from_distribution,
)

pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")

# Band positions (ascending wavelength) per resolution folder, as in the Zenodo layout.
POSITIONS_BY_RESOLUTION = {60: [0], 10: [1, 2, 3, 7], 20: [4, 5, 6, 8, 9, 10]}
SCENES, CROPS = ("Scene_0", "Scene_1"), ("1", "2")


def fill(position: int) -> float:
    return 0.01 * (position + 1)


def write_tif(path, array):
    # Real MADOS files declare EPSG:4326 but carry no geotransform.
    with rasterio.open(path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1], count=1,
                       dtype=array.dtype, crs="EPSG:4326", transform=Affine.identity()) as dst:
        dst.write(array, 1)


@pytest.fixture
def raw_mados(tmp_path):
    root = tmp_path / "MADOS"
    for scene in SCENES:
        for resolution, positions in POSITIONS_BY_RESOLUTION.items():
            (root / scene / str(resolution)).mkdir(parents=True)
            size = PATCH_SIZE * GSD // resolution
            for position in positions:
                for crop in CROPS:
                    band = np.full((size, size), fill(position), dtype="float32")
                    band[0, 0] = np.nan
                    name = f"{scene}_L2R_rhorc_{PLATFORM_WAVELENGTHS['S2A'][position]}_{crop}.tif"
                    write_tif(root / scene / str(resolution) / name, band)
        for crop in CROPS:
            # 0 = unannotated; blocks of the first and last class.
            mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype="uint8")
            mask[10:30, 10:30] = 1
            mask[40:60, 40:60] = NUM_CLASSES
            write_tif(root / scene / str(GSD) / f"{scene}_L2R_cl_{crop}.tif", mask)
    rois = [f"{scene}_{crop}" for scene in SCENES for crop in CROPS]
    (root / "splits").mkdir()
    for split, entries in {"train": rois[:2], "val": rois[2:3], "test": rois[3:]}.items():
        (root / "splits" / f"{split}_X.txt").write_text("\n".join(entries) + "\n")
    return root


def test_prepare_and_load(run_script, raw_mados, tmp_path):
    out = tmp_path / "data" / "MADOS_ml"
    run_script("prepare_mados.py", "--path", raw_mados, "--out", out, "--allow-subset")

    for split, count in {"train": 2, "val": 1, "test": 1}.items():
        assert len(list((out / split / "images").glob("*.tif"))) == count
        assert len(list((out / split / "labels").glob("*.tif"))) == count
    with rasterio.open(out / "train" / "images" / "Scene_0_1.tif") as src:
        image = src.read()
    assert image.shape == (len(WAVELENGTHS), PATCH_SIZE, PATCH_SIZE) and not np.isnan(image).any()
    assert image[3, 0, 0] == pytest.approx(BANDS_MEAN[3]) and image[3, 1, 1] == pytest.approx(fill(3))
    stats = json.loads((out / "stats.json").read_text())
    assert not stats["complete"] and stats["splits"]["train"]["patches"] == 2
    assert stats["splits"]["train"]["class_pixels"]["Marine Debris"] == 2 * 400
    assert stats["train_band_means"][0] == pytest.approx(fill(0), abs=1e-3)

    cfg = OmegaConf.load(REPO / "configs" / "data" / "mados_rgb.yaml")
    cfg.merge_with({"data_root": str(out.parent), "batch_size": 2, "num_workers": 0,
                    "means": [0.0] * 3, "stds": [1.0] * 3, "allow_incomplete": True})
    cfg = OmegaConf.create({"data": cfg}).data  # resolves ${data.output_bands}
    datamodule = hydra.utils.instantiate(cfg, _convert_="all")
    datamodule.setup("fit")
    batch = datamodule.on_after_batch_transfer(next(iter(datamodule.val_dataloader())), 0)
    assert batch["image"].shape == (1, 3, PATCH_SIZE, PATCH_SIZE)
    # R, G, B = 665, 560, 492 nm = positions 3, 2, 1.
    assert batch["image"][0, :, 1, 1].tolist() == pytest.approx([fill(3), fill(2), fill(1)])
    assert set(torch.unique(batch["mask"]).tolist()) == {-1, 0, NUM_CLASSES - 1}


def test_class_weights_match_config():
    cfg = OmegaConf.load(REPO / "configs" / "data" / "mados_rgb.yaml")
    assert class_weights_from_distribution(cfg.class_weight_smoothing) == pytest.approx(cfg.class_weights, abs=1e-5)
    assert cfg.class_weight_smoothing == 1.03
