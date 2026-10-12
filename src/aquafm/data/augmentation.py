"""Albumentations composition seeded from PyTorch's RNG."""

import albumentations as A
import torch


class SeededCompose(A.Compose):
    """`A.Compose` that draws its per-call seed from `torch.randint`.

    Albumentations 2.x seeds itself from OS entropy, so `pl.seed_everything` alone does not
    make it reproducible.
    """

    def __init__(self, transforms, **kwargs):
        kwargs.setdefault("is_check_shapes", False)
        super().__init__(transforms, **kwargs)

    def __call__(self, *args, **data):
        self.set_random_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        return super().__call__(*args, **data)
