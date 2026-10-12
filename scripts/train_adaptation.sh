#!/usr/bin/env bash
# Continued SSL training of DINOv3 ViT-B/16 on the unlabeled corpus, with automatic resume.
#
# Usage: scripts/train_adaptation.sh <dinov3-fino clone> <output dir> [extra key=value opts...]
#
# The clone must have fino_patches/ applied (see fino_patches/README.md), and
# AQUAFM_DATA_ROOT must point at the directory holding S2_unlabeled/.
#
# Dataloader worker memory grows by ~3 MB per iteration, so on a ~200 GB host the job is
# OOM-killed every ~3-4k iterations. train.py resumes from <output>/ckpt (every 250
# iterations), so this script restarts it until TARGET_ITERS is reached.
set -uo pipefail

CLONE=$(realpath "${1:?usage: $0 <clone> <output dir> [opts...]}")
OUT=$(realpath -m "${2:?usage: $0 <clone> <output dir> [opts...]}")
shift 2
CONFIG=dinov3/configs/train/subaqua_vitb16_guided_b768_ref.yaml
TARGET_ITERS=${TARGET_ITERS:-25000}
NPROC=${NPROC:-2}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}

if ! grep -q "Average the norms across ranks" "$CLONE/dinov3/train/guided_ssl_meta_arch.py"; then
    echo "guide_gradnorm_sync.patch is not applied to $CLONE; see fino_patches/README.md" >&2
    exit 1
fi
mkdir -p "$OUT"

last_iter() {
    python - "$OUT/training_metrics.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
best = 0
for line in (p.read_text().splitlines() if p.exists() else []):
    try:
        best = max(best, int(json.loads(line).get("iteration", 0)))
    except Exception:
        pass
print(best)
PY
}

# Progress is counted in new metric lines: a resume replays from the last checkpoint, so the
# max iteration can stay flat while training progresses.
metric_lines() {
    local f="$OUT/training_metrics.json"
    [ -f "$f" ] && grep -c . "$f" || echo 0
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    start=$(last_iter)
    lines=$(metric_lines)
    if [ "$start" -ge $((TARGET_ITERS - 1)) ]; then
        echo "done: reached iteration $start"; exit 0
    fi
    echo "attempt $attempt: starting from iteration $start"
    ( cd "$CLONE" && PYTHONPATH="$CLONE" torchrun --standalone --nproc_per_node="$NPROC" \
        dinov3/train/train.py --config-file "$CONFIG" --output-dir "$OUT" "$@" ) \
        >> "$OUT/train.log" 2>&1
    rc=$?
    end=$(last_iter)
    echo "attempt $attempt exited with code $rc at iteration $end"
    if [ "$end" -ge $((TARGET_ITERS - 1)) ]; then
        echo "done: reached iteration $end"; exit 0
    fi
    if [ "$(metric_lines)" -le "$lines" ]; then
        echo "no progress since the last attempt; see $OUT/train.log" >&2
        exit 1
    fi
    pkill -f "dinov3/train/train.py" 2>/dev/null; sleep 30
done
echo "gave up after $MAX_ATTEMPTS attempts" >&2; exit 1
