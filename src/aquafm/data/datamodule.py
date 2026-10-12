"""Synthetic RGB segmentation data for running the pipeline without a real dataset."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from aquafm.data.transforms import Compose, Normalize, RandomFlip, RandomRot90

# Fixed seeds, so the band response and the label thresholds are shared by all splits.
_GAIN_SEED = 123_456_789
_CALIBRATION_SEED = 987_654_321
_CALIBRATION_SAMPLES = 16


class SyntheticSegmentationDataset(Dataset):
    """Smooth random scenes labelled by thresholding the (G - R) / (G + R) index.

    Thresholds are quantiles of the index, so classes are balanced and the task is
    learnable. Images add per-band gain/offset, noise and a DN scale, so normalization
    matters. Structures are 32 px, coarser than a ViT/16 patch.
    """

    smooth_scale = 32
    noise_std = 0.05
    value_scale = 10_000.0
    index_bands = (1, 0)

    def __init__(self, num_samples: int, num_channels: int, img_size: int, num_classes: int,
                 seed: int = 0, transform=None):
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.img_size = img_size
        self.num_classes = num_classes
        self.seed = seed
        self.transform = transform

        generator = torch.Generator().manual_seed(_GAIN_SEED)
        self.band_gain = (0.5 + torch.rand(num_channels, generator=generator)).view(-1, 1, 1)
        self.band_offset = (0.1 * torch.rand(num_channels, generator=generator)).view(-1, 1, 1)

        pooled = torch.cat([
            self._index(self._reflectance(torch.Generator().manual_seed(_CALIBRATION_SEED + i))).flatten()
            for i in range(_CALIBRATION_SAMPLES)
        ])
        quantiles = torch.arange(1, num_classes, dtype=torch.float32) / num_classes
        self.bin_edges = torch.quantile(pooled, quantiles).contiguous()

    def _reflectance(self, generator: torch.Generator) -> torch.Tensor:
        low_res = self.img_size // self.smooth_scale
        grid = torch.rand(self.num_channels, low_res, low_res, generator=generator)
        return F.interpolate(grid.unsqueeze(0), size=(self.img_size, self.img_size),
                             mode="bilinear", align_corners=False).squeeze(0)

    def _index(self, reflectance: torch.Tensor) -> torch.Tensor:
        a, b = reflectance[self.index_bands[0]], reflectance[self.index_bands[1]]
        return (a - b) / (a + b + 1e-6)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        generator = torch.Generator().manual_seed(self.seed + idx)
        reflectance = self._reflectance(generator)
        mask = torch.bucketize(self._index(reflectance), self.bin_edges)
        noise = self.noise_std * torch.randn(self.num_channels, self.img_size, self.img_size,
                                             generator=generator)
        image = (reflectance * self.band_gain + self.band_offset + noise) * self.value_scale
        sample = {"image": image.float(), "mask": mask.long()}
        return self.transform(sample) if self.transform is not None else sample


class DummySegmentationDataModule(pl.LightningDataModule):
    """Train/val/test splits of `SyntheticSegmentationDataset`, normalized with train statistics."""

    def __init__(self, num_channels: int, img_size: int, num_classes: int, batch_size: int = 4,
                 num_workers: int = 0, num_train_samples: int = 64, num_val_samples: int = 16,
                 num_test_samples: int = 16, augment: bool = True, seed: int = 0):
        super().__init__()
        self.save_hyperparameters()

    def _dataset(self, num_samples: int, seed: int) -> SyntheticSegmentationDataset:
        h = self.hparams
        return SyntheticSegmentationDataset(num_samples, h.num_channels, h.img_size, h.num_classes, seed)

    def setup(self, stage: str | None = None) -> None:
        h = self.hparams
        # Per-sample seeds are `split seed + idx`; splits must stay under 10k samples.
        self.train_dataset = self._dataset(h.num_train_samples, h.seed)
        self.val_dataset = self._dataset(h.num_val_samples, h.seed + 10_000)
        self.test_dataset = self._dataset(h.num_test_samples, h.seed + 20_000)

        total = torch.zeros(h.num_channels, dtype=torch.float64)
        total_sq = torch.zeros(h.num_channels, dtype=torch.float64)
        for i in range(len(self.train_dataset)):
            image = self.train_dataset[i]["image"].to(torch.float64)
            total += image.sum(dim=(1, 2))
            total_sq += (image**2).sum(dim=(1, 2))
        pixels = len(self.train_dataset) * h.img_size**2
        means = total / pixels
        stds = (total_sq / pixels - means**2).clamp_min(1e-12).sqrt()
        self.means, self.stds = means.float().tolist(), stds.float().tolist()

        normalize = Normalize(self.means, self.stds)
        self.train_dataset.transform = (
            Compose([RandomFlip(), RandomRot90(), normalize]) if h.augment else normalize
        )
        self.val_dataset.transform = normalize
        self.test_dataset.transform = normalize

    def _dataloader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=shuffle,
                          num_workers=self.hparams.num_workers)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, shuffle=False)
