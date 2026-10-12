import csv
import json

import numpy as np
import rasterio
import torch
from rasterio.transform import Affine

from aquafm.adaptation import dinov3_checkpoint
from aquafm.adaptation.bundle_corpus import MANIFEST_NAME, PATCH_SIZE, STATS_NAME
from aquafm.adaptation.bundle_dataset import SubAqua


def test_checkpoint_round_trip(tmp_path):
    hub = {"cls_token": torch.randn(1, 1, 8), "patch_embed.proj.weight": torch.randn(8, 3, 16, 16),
           "blocks.0.attn.qkv.weight": torch.randn(24, 8), "norm.weight": torch.randn(8)}
    torch.save(hub, tmp_path / "hub.pth")
    dinov3_checkpoint.main(["to-fino", "--input", str(tmp_path / "hub.pth"), "--output", str(tmp_path / "fino.pth")])
    fino = torch.load(tmp_path / "fino.pth", weights_only=True)
    assert set(fino) == {"teacher"} and set(fino["teacher"]) == {f"backbone.{k}" for k in hub}
    # A FiNo teacher also carries SSL heads, which to-hub drops.
    fino["teacher"]["dino_head.mlp.0.weight"] = torch.randn(4, 8)
    torch.save(fino, tmp_path / "fino.pth")
    dinov3_checkpoint.main(["to-hub", "--input", str(tmp_path / "fino.pth"), "--output", str(tmp_path / "back.pth")])
    back = torch.load(tmp_path / "back.pth", weights_only=True)
    assert back.keys() == hub.keys() and all(torch.equal(back[k], hub[k]) for k in hub)


def write_patch(path, array, epsg, origin):
    with rasterio.open(path, "w", driver="GTiff", height=PATCH_SIZE, width=PATCH_SIZE, count=3, dtype="uint8",
                       crs=f"EPSG:{epsg}", transform=Affine(10, 0, origin[0], 0, -10, origin[1])) as dst:
        dst.write(array)


def test_manifest_and_dataset(run_script, tmp_path):
    root = tmp_path / "S2_unlabeled"
    rng = np.random.default_rng(0)
    # (tile, UTM EPSG, origin): one southern and one northern tile.
    for tile, epsg, origin in [("15MYV", 32715, (500000, 9900000)), ("50NKL", 32650, (300000, 500000))]:
        (root / tile).mkdir(parents=True)
        for date in ("20190112", "20210605"):
            for index in range(2):
                array = rng.integers(20, 120, (3, PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                write_patch(root / tile / f"{tile}_{date}_{index}.tif", array, epsg, origin)
    write_patch(root / "50NKL" / "50NKL_20210605_9.tif", np.zeros((3, PATCH_SIZE, PATCH_SIZE), np.uint8),
                32650, (300000, 500000))

    run_script("build_bundle_manifest.py", "--root", root, "--expected-bundles", 2, "--jobs", 1)
    with (root / MANIFEST_NAME).open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8 and not any(r["path"].endswith("_9.tif") for r in rows)
    stats = json.loads((root / STATS_NAME).read_text())
    assert stats["complete"] and stats["patches_rejected"] == 1
    assert stats["guide_cardinalities"] == {"tile": 2, "year": 2, "season": 2}
    assert 20 / 255 < min(stats["rgb_mean"]) < max(stats["rgb_mean"]) < 120 / 255

    dataset = SubAqua(root=str(root), with_metadata=True)
    assert len(dataset) == 8
    image, (_, metadata) = dataset[0]
    assert image.size == (PATCH_SIZE, PATCH_SIZE) and image.mode == "RGB"
    assert stats["vocabularies"]["tile"][metadata.tile] == rows[0]["tile"]
    assert stats["vocabularies"]["year"][metadata.year] == int(rows[0]["year"])
