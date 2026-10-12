# Patches for Meta's dinov3 `FINO` branch

The adaptation is trained with the `FINO` branch of
[facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) and the changes in this
folder. All files here are derivatives of the DINOv3 Materials and are distributed under the
[DINOv3 License](../LICENSE-DINOv3.md).

| File | Content |
|---|---|
| `subaqua.py` | Dataset shim; re-exports `aquafm.adaptation.bundle_dataset.SubAqua`. |
| `register_subaqua.patch` | Registers `SubAqua` in `dinov3/data/datasets/__init__.py` and in `_parse_dataset_str` (`dinov3/data/loaders.py`), following the existing `FMoW` entry. |
| `guide_gradnorm_sync.patch` | Fix for a multi-GPU hang (below). |
| `subaqua_vitb16_guided_b768_ref.yaml` | Training config of the released checkpoint. |

## Apply

```bash
git clone --branch FINO https://github.com/facebookresearch/dinov3.git dinov3-fino
cd dinov3-fino
git checkout 81aae9a
cp <repo>/fino_patches/subaqua.py dinov3/data/datasets/subaqua.py
git apply <repo>/fino_patches/register_subaqua.patch
git apply <repo>/fino_patches/guide_gradnorm_sync.patch
cp <repo>/fino_patches/subaqua_vitb16_guided_b768_ref.yaml dinov3/configs/train/
pip install -e <repo>        # the shim imports aquafm
```

The config's dataset string is
`SubAqua:root=$AQUAFM_DATA_ROOT/S2_unlabeled:split=TRAIN:with_metadata=true`; `SubAqua` reads
`manifest.csv` and `manifest_stats.json` from `root`. To check the registration:

```bash
cd dinov3-fino
PYTHONPATH=$PWD python -c "
import os
from dinov3.data.loaders import _parse_dataset_str
cls, kw = _parse_dataset_str(f'SubAqua:root={os.environ[\"AQUAFM_DATA_ROOT\"]}/S2_unlabeled:split=TRAIN:with_metadata=true')
ds = cls(**kw); image, (label, meta) = ds[0]
print(len(ds), image.mode, image.size, meta)
"
```

## `guide_gradnorm_sync.patch`

`GuidedSSLMetaArch._apply_grad_norm_normalization` compares each guide's gradient norm with
`eps = 1e-6` per rank, and the outcome determines which keys enter `loss_dict`, which
`train.py` stacks and all-reduces. When ranks disagree, `all_reduce` receives tensors of
different sizes and NCCL hangs. The patch averages the norms across ranks before the comparison
and writes the same keys on both branches. The released run sets
`optim.grad_norm_normalization: false`, so this code path was not exercised in training.
