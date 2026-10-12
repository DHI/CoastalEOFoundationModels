"""Trainer construction shared by scripts/train.py and scripts/test.py."""

import hydra
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig


def build_trainer(cfg: DictConfig, with_callbacks: bool = True) -> pl.Trainer:
    """Instantiate `cfg.trainer` with the configured logger and callbacks."""
    kwargs = {}
    logging = cfg.trainer.get("logger") is not False
    if logging:
        kwargs["logger"] = hydra.utils.instantiate(cfg.logger)
    if with_callbacks and cfg.get("callbacks"):
        callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks.values()]
        # Lightning rejects these callbacks when checkpointing or logging is off.
        if cfg.trainer.get("enable_checkpointing") is False or cfg.trainer.get("fast_dev_run"):
            callbacks = [c for c in callbacks if not isinstance(c, ModelCheckpoint)]
        if not logging:
            callbacks = [c for c in callbacks if not isinstance(c, LearningRateMonitor)]
        kwargs["callbacks"] = callbacks
    return hydra.utils.instantiate(cfg.trainer, **kwargs)


def close_loggers(trainer: pl.Trainer) -> None:
    """Finish wandb runs, so the next job of a multirun starts a new one."""
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            logger.experiment.finish()
