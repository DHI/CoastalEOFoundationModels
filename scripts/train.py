"""Train a segmentation model: `python scripts/train.py experiment=<name> [protocol=mados_reference]`."""

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
    pl.seed_everything(cfg.seed, workers=True)
    # _convert_="all": torchmetrics' ClasswiseWrapper rejects ListConfig class names.
    datamodule = hydra.utils.instantiate(cfg.data, _convert_="all")
    task = hydra.utils.instantiate(cfg.model, _convert_="all")
    trainer = build_trainer(cfg)
    trainer.fit(model=task, datamodule=datamodule)
    print(f"Wrote {write_record(cfg, trainer, 'fit')}")
    close_loggers(trainer)


if __name__ == "__main__":
    main()
