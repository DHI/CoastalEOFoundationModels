"""Evaluate a checkpoint: `python scripts/test.py experiment=<name> [protocol=...] ckpt_path=<ckpt>`."""

import hydra
import lightning.pytorch as pl
import terratorch  # noqa: F401  (registers TerraTorch backbones, necks and decoders)
from omegaconf import DictConfig

import aquafm.backbones.copernicus_fm  # noqa: F401  (registers copernicusfm_vit)
import aquafm.backbones.prithvi  # noqa: F401  (registers prithvi_vit)
from aquafm.utils.results import write_record
from aquafm.utils.runtime import build_trainer, close_loggers


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt_path is None:
        raise ValueError("ckpt_path is required; pass ckpt_path=<checkpoint> with the experiment it was trained with")
    pl.seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.data, _convert_="all")
    task = hydra.utils.instantiate(cfg.model, _convert_="all")
    trainer = build_trainer(cfg, with_callbacks=False)
    # weights_only=False: the checkpoint's hyperparameters hold OmegaConf objects.
    trainer.test(model=task, datamodule=datamodule, ckpt_path=cfg.ckpt_path, weights_only=False)
    print(f"Wrote {write_record(cfg, trainer, 'test')}")
    close_loggers(trainer)


if __name__ == "__main__":
    main()
