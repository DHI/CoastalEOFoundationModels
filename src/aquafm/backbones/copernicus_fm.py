"""CopernicusFM (torchgeo) as a TerraTorch backbone returning multi-layer patch tokens."""

import torch
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from torch import nn
from torchgeo.models.copernicusfm import CopernicusFM_Base_Weights, copernicusfm_base


class CopernicusFMWrapper(nn.Module):
    """torchgeo's CopernicusFM ViT-B, returning [B, N + 1, C] tokens from `out_indices`.

    CopernicusFM's forward returns a pooled vector, so forward hooks on the blocks collect
    the tokens. `wavelengths` and `bandwidths` (nm, one per input channel) drive its
    spectral patch embedding. No geolocation is passed, so CopernicusFM uses its learned
    placeholder tokens. `apply_norm` applies the final LayerNorm (`fc_norm`) to every
    returned layer, as DINOv3's features are normalized.
    """

    def __init__(
        self,
        wavelengths: list,
        bandwidths: list,
        pretrained: bool = False,
        out_indices: list | None = None,
        apply_norm: bool = True,
        **model_kwargs,
    ):
        super().__init__()
        if len(wavelengths) != len(bandwidths):
            raise ValueError(f"wavelengths ({len(wavelengths)}) and bandwidths ({len(bandwidths)}) must have one entry per band")
        weights = CopernicusFM_Base_Weights.CopernicusFM_ViT if pretrained else None
        self.model = copernicusfm_base(weights=weights, **model_kwargs)
        self.wavelengths = list(wavelengths)
        self.bandwidths = list(bandwidths)
        self.apply_norm = apply_norm

        depth = len(self.model.blocks)
        self.out_indices = out_indices or [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        embed_dim = self.model.cls_token.shape[-1]
        self.out_channels = [embed_dim] * len(self.out_indices)

        self._features = {}
        for i in self.out_indices:
            self.model.blocks[i].register_forward_hook(self._make_hook(i))

    def _make_hook(self, index: int):
        def hook(module, inputs, output):
            self._features[index] = output

        return hook

    def forward(self, x: torch.Tensor, **_) -> list:
        metadata = torch.full((x.shape[0], 4), float("nan"), device=x.device, dtype=x.dtype)
        self.model.forward_features(
            x, metadata, wavelengths=self.wavelengths, bandwidths=self.bandwidths, input_mode="spectral"
        )
        features = [self._features.pop(i) for i in self.out_indices]
        if self.apply_norm:
            features = [self.model.fc_norm(f) for f in features]
        return features


@TERRATORCH_BACKBONE_REGISTRY.register
def copernicusfm_vit(**kwargs) -> CopernicusFMWrapper:
    """CopernicusFM ViT-B backbone, registered as "copernicusfm_vit"."""
    return CopernicusFMWrapper(**kwargs)
