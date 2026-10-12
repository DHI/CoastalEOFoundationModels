"""Per-run result records (`results_fit.json`, `results_test.json`)."""

import json
from pathlib import Path

import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf


def write_record(cfg: DictConfig, trainer: pl.Trainer, stage: str) -> Path:
    """Write the run's config and final metrics to `results_<stage>.json` in the Hydra run dir."""
    hydra_cfg = HydraConfig.get()
    checkpoint = next((c for c in trainer.callbacks if isinstance(c, ModelCheckpoint)), None)
    best_score = checkpoint.best_model_score if checkpoint else None
    record = {
        "experiment": hydra_cfg.runtime.choices.get("experiment"),
        "protocol": hydra_cfg.runtime.choices.get("protocol"),
        "seed": cfg.seed,
        "stage": stage,
        "metrics": {k: v.item() for k, v in sorted(trainer.callback_metrics.items())},
        "best_score": best_score.item() if best_score is not None else None,
        "best_ckpt_path": (checkpoint.best_model_path or None) if checkpoint else None,
        "ckpt_path": cfg.ckpt_path,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    path = Path(hydra_cfg.runtime.output_dir) / f"results_{stage}.json"
    path.write_text(json.dumps(record, indent=2))
    return path
