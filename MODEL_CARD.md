# Model card: DINOv3 ViT-B/16 adapted to tropical coastal water

`dinov3_vitb16_subaqua_adapted.pth` is a DINOv3 ViT-B/16 encoder adapted to shallow coastal
water by continued self-supervised training on the
[tropical coastal Sentinel-2 corpus](corpus/DATASET_CARD.md). It has the same architecture and
key layout as Meta's `dinov3_vitb16_pretrain_lvd1689m` checkpoint and loads the same way through
`torch.hub` or TerraTorch's `dinov3_vitb16`.

| | |
|---|---|
| Developed by | Spyros Kondylatos, Nicklas Simonsen (DHI) |
| Paper | Land-Pretrained Foundation Models Do Not Transfer to Shallow Water (REO-2 at NeurIPS 2026) |
| Download | [GitHub release v0.1.0](https://github.com/DHI/CoastalEOFoundationModels/releases/tag/v0.1.0), or `python scripts/download_weights.py` |
| SHA-256 | `5aef9a9bba55e32bdca00ef34f494c4c93f5b5718dd632249b3002289899c0b3` |
| Size | 343 MB |
| Architecture | ViT-B/16, 86 M parameters, 4 register tokens, RoPE, `mask_k_bias=True` |
| Initialised from | DINOv3 ViT-B/16 LVD-1689M (`dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`) |
| Input | RGB, 224×224 training crops, corpus normalisation (below) |
| License | [DINOv3 License](LICENSE-DINOv3.md) (derivative of the DINOv3 Materials) |

## Training

- Recipe: self-distillation from Meta's `FINO` branch of `facebookresearch/dinov3` (DINO, iBOT
  and SigReg, plus two metadata heads: `tile` as a prototypical guide and `year` as an
  adversarial guide). Config: `fino_patches/subaqua_vitb16_guided_b768_ref.yaml`.
- Schedule: 25,000 iterations at a global batch of 1,536 (2 × 768) on 2× RTX PRO 6000
  Blackwell GPUs, about 26 h. The backbone learning rate is zero for the first 10,000
  iterations (heads only), followed by a 1,000-iteration ramp and full-backbone training.
- Export: teacher EMA weights at iteration 24,999, converted with
  `python -m aquafm.adaptation.dinov3_checkpoint to-hub`. Relative L2 distance to the stock
  checkpoint is 13.3 %.
- Normalisation (0–1 units): mean `[0.2135, 0.4032, 0.4288]`, std `[0.2423, 0.2787, 0.2729]`.
  The downstream benchmarks use each dataset's own statistics.

Image-only objectives account for 98.9 % of the training loss and no control without the
metadata guides was run, so we describe the model as domain-adapted (continued SSL on in-domain
imagery) and make no claim about the effect of metadata guidance.

## Evaluation

Test mIoU with RGB input and a UperNet decoder, single training seed. Run-to-run spread is
about 0.005–0.02 mIoU depending on the dataset. MADOS is reproducible with this repository; the
two SAV datasets are not public.

| Benchmark | Stock, frozen | Adapted, frozen | Stock, fine-tuned | Adapted, fine-tuned |
|---|---|---|---|---|
| MADOS (15 classes, global marine) | 0.712 | 0.717 | 0.770 | 0.805 |
| SAV Denmark (3 classes, temperate) | 0.696 | 0.714 | 0.732 | 0.739 |
| SAV Malaysia (5 classes, tropical reef) | 0.557 | 0.664 | 0.630 | 0.642 |

The frozen gain is largest where the downstream domain matches the corpus (SAV Malaysia,
+0.107) and close to zero on MADOS. With fine-tuning, the adapted encoder improves on MADOS and
Denmark, and stock DINOv3 mostly closes the gap on Malaysia. Most of the fine-tuned MADOS gain
comes from one rare class.

## Intended use and limitations

Semantic segmentation and other dense tasks on Sentinel-2-like RGB imagery of coastal and
shallow water, with the encoder frozen or fine-tuned. Training data is 8-bit RGB at 10 m GSD;
multispectral input, other resolutions and calibrated reflectance are untested. The adaptation
data is tropical, and the benefit outside that domain is small.

## Usage

```python
import torch
model = torch.hub.load("facebookresearch/dinov3", "dinov3_vitb16",
                       weights="weights/dinov3_vitb16_subaqua_adapted.pth")
```

In this repository, `experiment=mados_dinov3b_upernet_rgb_adapted_frozen` and
`experiment=mados_dinov3b_upernet_rgb_adapted` load the file from `$AQUAFM_WEIGHTS_ROOT`.

## License

The weights are a derivative of Meta's DINOv3 weights and are released under the
[DINOv3 License](LICENSE-DINOv3.md), which also applies to redistribution. Built with
DINOv3 (Siméoni et al., arXiv:2508.10104, 2025).
