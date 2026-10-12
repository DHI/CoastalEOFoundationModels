import json

import pytest
from conftest import REPO
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

EXPERIMENTS = sorted(p.stem for p in (REPO / "configs" / "experiment").glob("*.yaml"))
# Tiny but complete run: CSV logger, checkpoints, callbacks.
TINY = ["experiment=dummy_baseline_unet_scratch", "trainer.accelerator=cpu", "trainer.devices=1",
        "trainer.max_epochs=1", "data.img_size=64", "data.batch_size=2",
        "data.num_train_samples=4", "data.num_val_samples=2", "data.num_test_samples=2"]


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_composes(experiment):
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base=None):
        cfg = compose("config", overrides=[f"experiment={experiment}", "protocol=mados_reference"],
                      return_hydra_config=True)
    HydraConfig.instance().set_config(cfg)
    del cfg.hydra
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["model"]["tta"] and resolved["data"]["vscp"] and resolved["trainer"]["max_epochs"] == 80


@pytest.mark.parametrize("experiment", ["dummy_baseline_unet_scratch", "dummy_dinov3_upernet_frozen"])
def test_smoke_train(run_script, tmp_path, experiment):
    run_script("train.py", f"experiment={experiment}", "trainer=smoke", f"hydra.run.dir={tmp_path}")
    assert (tmp_path / "results_fit.json").exists()


def test_train_test_aggregate(run_script, tmp_path):
    run_script("train.py", *TINY, f"hydra.run.dir={tmp_path / 'fit'}")
    ckpt = tmp_path / "fit" / "checkpoints" / "last.ckpt"
    assert ckpt.exists()
    run_script("test.py", *TINY, f"ckpt_path={ckpt}", f"hydra.run.dir={tmp_path / 'test'}")
    record = json.loads((tmp_path / "test" / "results_test.json").read_text())
    assert record["experiment"] == "dummy_baseline_unet_scratch"
    assert 0.0 <= record["metrics"]["test/mIoU"] <= 1.0
    table = run_script("aggregate_results.py", tmp_path, "--metrics", "test/mIoU")
    assert "dummy_baseline_unet_scratch" in table and "test/mIoU" in table
