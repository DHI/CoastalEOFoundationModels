# Coastal DINOv3: MADOS benchmark and in-domain adaptation

Code for "On the Transferability of Earth Observation Foundation Models to Coastal Environments" (REO-2 Workshop at NeurIPS 2026).
The repository contains a segmentation benchmark on [MADOS](https://marine-pollution.github.io/)
(Sentinel-2, 15 marine classes) comparing DINOv3 (web and satellite weights), CopernicusFM,
Prithvi-EO-2.0 and supervised UNet / ViT baselines, all with an UperNet decoder and RGB input;
and the domain adaptation of DINOv3 ViT-B/16 by continued self-supervised training on an
unlabeled corpus of tropical coastal Sentinel-2 imagery, using Meta's `FINO` training branch.
We release the adapted encoder ([model card](MODEL_CARD.md)) and the corpus
([dataset card](corpus/DATASET_CARD.md)). The code builds on
[TerraTorch](https://github.com/IBM/terratorch) and [TorchGeo](https://github.com/torchgeo/torchgeo)
with Hydra configs.

Authors: Spyros Kondylatos, Nicklas Simonsen (DHI).

| Resource | Link |
|---|---|
| Paper | `<PAPER_URL>` |
| Adapted DINOv3 ViT-B/16 weights | [Hugging Face](https://huggingface.co/<HF_REPO_ID>) ([model card](MODEL_CARD.md)) |
| Unlabeled tropical coastal corpus | [Zenodo](https://zenodo.org/records/23313478) ([dataset card](corpus/DATASET_CARD.md)) |
| MADOS | [Zenodo](https://zenodo.org/records/10664073) |

> **License.** The code is released under [Apache-2.0](LICENSE). The adapted weights,
> `fino_patches/` and `src/aquafm/adaptation/vision_dataset.py` are derivatives of Meta's
> DINOv3 and are released under the [DINOv3 License](LICENSE-DINOv3.md). Built with DINOv3. 
> Use and redistribution of these components must comply with the DINOv3 License, including
> its trade-control and use restrictions. The corpus is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

## Installation

```bash
pip install -e ".[dev]"                       # add ,wandb for logger=wandb
pytest                                        # CPU smoke tests, no downloads

export AQUAFM_DATA_ROOT=/path/to/data         # MADOS/, MADOS_ml/, S2_unlabeled/
export AQUAFM_WEIGHTS_ROOT=/path/to/weights   # *.pth checkpoints
```

A pipeline check on synthetic data runs on CPU:

```bash
python scripts/train.py experiment=dummy_baseline_unet_scratch
python scripts/train.py experiment=dummy_dinov3_upernet_frozen trainer=smoke
```

## Data

MADOS:

```bash
cd "$AQUAFM_DATA_ROOT"
curl -L -C - --retry 5 -o MADOS.zip "https://zenodo.org/api/records/10664073/files/MADOS.zip/content"
echo "1076b25d2797be3095d82a61105c7380  MADOS.zip" | md5sum -c -
unzip -q MADOS.zip
cd -
python scripts/prepare_mados.py --path $AQUAFM_DATA_ROOT/MADOS --out $AQUAFM_DATA_ROOT/MADOS_ml
```

Tropical coastal corpus (188,857 patches, 65 MGRS tiles): download from
[Zenodo](https://zenodo.org/records/23313478) and extract into `$AQUAFM_DATA_ROOT/S2_unlabeled`.
The Zenodo record also holds the manifests (`manifest.csv`, `manifest_rejected.csv`,
`manifest_stats.json`, `config_fragment.yaml`); place them in the root of `S2_unlabeled/`.
Training reads `manifest.csv` and `manifest_stats.json` from there. To rebuild them instead,
run `python scripts/build_bundle_manifest.py --root $AQUAFM_DATA_ROOT/S2_unlabeled`.

## Weights

- DINOv3 (gated): accept the license on the
  [DINOv3 downloads page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/)
  and place `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`,
  `dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` and
  `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth` in `$AQUAFM_WEIGHTS_ROOT` under Meta's filenames.
- Adapted DINOv3 ViT-B/16: `python scripts/download_weights.py` downloads the checkpoint
  from [Hugging Face](https://huggingface.co/<HF_REPO_ID>) and verifies its SHA-256.
- CopernicusFM and Prithvi are downloaded on first use.

## Benchmark

```bash
python scripts/train.py experiment=mados_dinov3b_upernet_rgb_adapted_frozen protocol=mados_reference
python scripts/test.py  experiment=mados_dinov3b_upernet_rgb_adapted_frozen protocol=mados_reference \
    ckpt_path=<run dir>/checkpoints/best-<...>.ckpt
python scripts/aggregate_results.py outputs/
```

`test.py` takes the same overrides as the training run. Each run writes `results_fit.json` and
`results_test.json`; the best checkpoint path is recorded in `results_fit.json`.

All paper runs use `protocol=mados_reference`, which follows the MADOS reference
implementation ([gkakogeorgiou/mados](https://github.com/gkakogeorgiou/mados)): Adam (lr 2e-4),
MultiStepLR at epochs 45 and 65, 80 epochs, EMA 0.999, VSCP copy-paste augmentation, 8-view
test-time augmentation with majority vote, batch size 15, fp32. We report the test mIoU of the
checkpoint with the best validation mIoU, from a single seed per arm; seed-to-seed variation on
MADOS is large because rare classes alternate between zero and high IoU.

| Arm | Experiment | Test mIoU |
|---|---|---|
| U-Net, random init | `mados_baseline_unet` | 0.383 |
| U-Net, ImageNet | `mados_baseline_unet_imagenet` | 0.619 |
| ViT-B/16, ImageNet | `mados_baseline_vitb_upernet_rgb` | 0.547 |
| CopernicusFM ViT-B | `mados_copernicusfm_upernet_rgb_frozen` | 0.673 |
| Prithvi-EO-2.0 100M | `mados_prithvi_upernet_rgb_frozen` | 0.571 |
| DINOv3 ViT-L/16 SAT-493M | `mados_dinov3l_sat_upernet_rgb_frozen` | 0.715 |
| DINOv3 ViT-L/16 LVD-1689M | `mados_dinov3l_web_upernet_rgb_frozen` | 0.730 |
| DINOv3 ViT-B/16 LVD-1689M | `mados_dinov3b_upernet_rgb_frozen` | 0.712 |
| DINOv3 ViT-B/16, adapted | `mados_dinov3b_upernet_rgb_adapted_frozen` | 0.717 |
| DINOv3 ViT-B/16 LVD-1689M, fine-tuned | `mados_dinov3b_upernet_rgb` | 0.770 |
| DINOv3 ViT-B/16, adapted, fine-tuned | `mados_dinov3b_upernet_rgb_adapted` | 0.805 |

Encoders are frozen unless marked fine-tuned.

## Domain adaptation

The released checkpoint is used by the `mados_dinov3b_upernet_rgb_adapted*` experiments after
`python scripts/download_weights.py`. To reproduce it from the corpus:

```bash
# Meta's FINO branch with our patches (see fino_patches/README.md)
REPO=$PWD
git clone --branch FINO https://github.com/facebookresearch/dinov3.git ../dinov3-fino
(cd ../dinov3-fino && git checkout 81aae9a \
  && cp $REPO/fino_patches/subaqua.py dinov3/data/datasets/ \
  && git apply $REPO/fino_patches/register_subaqua.patch $REPO/fino_patches/guide_gradnorm_sync.patch \
  && cp $REPO/fino_patches/subaqua_vitb16_guided_b768_ref.yaml dinov3/configs/train/)

# Warm start from Meta's ViT-B/16, converted to the training layout
python -m aquafm.adaptation.dinov3_checkpoint to-fino \
  --input  $AQUAFM_WEIGHTS_ROOT/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --output $AQUAFM_WEIGHTS_ROOT/fino/dinov3_vitb16_lvd1689m_teacher.pth

# 25k iterations on 2 GPUs with >= 80 GB each; the script restarts after crashes
scripts/train_adaptation.sh ../dinov3-fino outputs/adaptation \
  student.resume_from_teacher_chkpt=$AQUAFM_WEIGHTS_ROOT/fino/dinov3_vitb16_lvd1689m_teacher.pth

# Export the teacher to the torch.hub layout
python -m aquafm.adaptation.dinov3_checkpoint to-hub \
  --input  outputs/adaptation/eval/24999/teacher_checkpoint.pth \
  --output $AQUAFM_WEIGHTS_ROOT/dinov3_vitb16_subaqua_adapted.pth
```

On other hardware, the global batch (`train.batch_size_per_gpu` × GPUs) and the
iteration-indexed schedule change together; the config header lists the coupled values.
Image-only objectives account for 98.9 % of the training loss and we ran no control without the
metadata guides, so we describe the result as a domain-adapted encoder.

## Citation

```bibtex
@inproceedings{kondylatos2026coastal,
  title     = {On the Transferability of Earth Observation Foundation Models to Coastal Environments},
  author    = {Kondylatos, Spyros and Simonsen, Nicklas},
  booktitle = {2nd Workshop on Advances in Representation Learning for Earth Observation (REO-2) at NeurIPS},
  year      = {2026}
}

```

This work builds on DINOv3 (Siméoni et al., arXiv:2508.10104, 2025) and the `FINO` adaptation
recipe ("Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already
Have", arXiv:2606.05107).

## License

Code is released under [Apache-2.0](LICENSE). `fino_patches/`,
`src/aquafm/adaptation/vision_dataset.py` and the adapted checkpoint are under the
[DINOv3 License](LICENSE-DINOv3.md); see [NOTICE](NOTICE). The corpus is under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); MADOS carries its own license, given on its page.
