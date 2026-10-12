"""Transforms on {"image": [C, H, W], "mask": [H, W]} dicts for the synthetic data."""

from collections.abc import Callable, Sequence

import torch


class Compose:
    def __init__(self, transforms: Sequence[Callable[[dict], dict]]):
        self.transforms = list(transforms)

    def __call__(self, sample: dict) -> dict:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class Normalize:
    """Per-band (x - mean) / std on the image."""

    def __init__(self, means: Sequence[float], stds: Sequence[float]):
        self.means = torch.as_tensor(list(means), dtype=torch.float32).view(-1, 1, 1)
        self.stds = torch.as_tensor(list(stds), dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, sample: dict) -> dict:
        sample["image"] = (sample["image"] - self.means) / self.stds
        return sample


class RandomFlip:
    """Independent horizontal and vertical flips, each with probability `p`."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict) -> dict:
        for dim in (-1, -2):
            if torch.rand(()) < self.p:
                sample["image"] = torch.flip(sample["image"], dims=[dim])
                sample["mask"] = torch.flip(sample["mask"], dims=[dim])
        return sample


class RandomRot90:
    """Rotation by a random multiple of 90 degrees."""

    def __call__(self, sample: dict) -> dict:
        k = int(torch.randint(0, 4, ()))
        if k:
            sample["image"] = torch.rot90(sample["image"], k, dims=(-2, -1))
            sample["mask"] = torch.rot90(sample["mask"], k, dims=(-2, -1))
        return sample
