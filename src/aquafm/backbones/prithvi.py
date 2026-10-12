"""Prithvi-EO as a TerraTorch backbone with per-layer `out_channels` and LayerNorm.

Wraps TerraTorch's Prithvi encoder, which reports one `out_channels` entry per block and
normalizes only the last block. 240 px MADOS inputs work through TerraTorch's
position-embedding interpolation.
"""

import torch
from terratorch.datasets import HLSBands
from terratorch.registry import BACKBONE_REGISTRY, TERRATORCH_BACKBONE_REGISTRY
from torch import nn

# Sentinel-2A band centres (nm) of the HLS bands, to check `bands` against the data.
HLS_BAND_CENTRES_NM = {
    HLSBands.COASTAL_AEROSOL: 443,
    HLSBands.BLUE: 492,
    HLSBands.GREEN: 560,
    HLSBands.RED: 665,
    HLSBands.RED_EDGE_1: 704,
    HLSBands.RED_EDGE_2: 740,
    HLSBands.RED_EDGE_3: 783,
    HLSBands.NIR_BROAD: 833,
    HLSBands.NIR_NARROW: 865,
    HLSBands.WATER_VAPOR: 945,
    HLSBands.CIRRUS: 1374,
    HLSBands.SWIR_1: 1614,
    HLSBands.SWIR_2: 2202,
}
# Covers S2B centres (up to 16 nm off); neighbouring S2 bands are at least 45 nm apart.
BAND_MATCH_TOLERANCE_NM = 20

VARIANT_DEPTHS = {
    "prithvi_eo_v1_100": 12,
    "prithvi_eo_v2_tiny_tl": 12,
    "prithvi_eo_v2_100_tl": 12,
    "prithvi_eo_v2_300": 24,
    "prithvi_eo_v2_300_tl": 24,
    "prithvi_eo_v2_600": 32,
    "prithvi_eo_v2_600_tl": 32,
}


class PrithviWrapper(nn.Module):
    """TerraTorch's Prithvi-EO encoder returning tokens from `out_indices`.

    `bands` are HLS band names in the order the data emits them; TerraTorch selects the
    pretrained patch-embedding weights by name, so the order matters. `wavelengths` (nm)
    from the data config are checked against them. `out_indices` must end at the final
    block. `apply_norm` applies the final LayerNorm to every returned layer, as DINOv3's
    features are normalized.
    """

    def __init__(
        self,
        bands: list,
        variant: str = "prithvi_eo_v2_100_tl",
        pretrained: bool = False,
        wavelengths: list | None = None,
        out_indices: list | None = None,
        apply_norm: bool = True,
        **model_kwargs,
    ):
        super().__init__()
        if variant not in VARIANT_DEPTHS:
            raise ValueError(f"unknown Prithvi variant {variant!r}; use one of {sorted(VARIANT_DEPTHS)}")
        bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in bands]
        if wavelengths is not None:
            self._check_bands(bands, wavelengths)

        depth = VARIANT_DEPTHS[variant]
        default_indices = [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]
        self.out_indices = list(out_indices) if out_indices else default_indices
        if max(self.out_indices) != depth - 1 or min(self.out_indices) < 0:
            raise ValueError(
                f"out_indices {self.out_indices} must lie in 0..{depth - 1} and end at the final block "
                f"of {variant}, e.g. {default_indices}"
            )

        self.model = BACKBONE_REGISTRY.build(
            f"terratorch_{variant}", pretrained=pretrained, bands=bands, out_indices=self.out_indices,
            **model_kwargs,
        )
        self.out_channels = [self.model.embed_dim] * len(self.out_indices)
        self.apply_norm = apply_norm
        self._final_block = depth - 1

    @staticmethod
    def _check_bands(bands: list, wavelengths: list) -> None:
        if len(bands) != len(wavelengths):
            raise ValueError(f"bands ({len(bands)}) and data wavelengths ({len(wavelengths)}) must have one entry per channel")
        for position, (band, nm) in enumerate(zip(bands, wavelengths)):
            if abs(HLS_BAND_CENTRES_NM[band] - nm) > BAND_MATCH_TOLERANCE_NM:
                raise ValueError(
                    f"channel {position}: bands lists {band} ({HLS_BAND_CENTRES_NM[band]} nm) but the data "
                    f"emits {nm} nm; reorder `bands` to match data.output_bands"
                )

    def forward(self, x: torch.Tensor, **_) -> list:
        features = self.model(x)
        if not self.apply_norm:
            return features
        # TerraTorch already normalized the final block's output.
        return [
            f if index == self._final_block else self.model.norm(f)
            for index, f in zip(self.out_indices, features)
        ]


@TERRATORCH_BACKBONE_REGISTRY.register
def prithvi_vit(**kwargs) -> PrithviWrapper:
    """Prithvi-EO backbone, registered as "prithvi_vit"; `variant` picks the checkpoint."""
    return PrithviWrapper(**kwargs)
