#!/bin/bash
# Wait for the 8-view refetch download to finish, then rebuild the 8-view SFT data
# and eval manifest. Does NOT retrain (left for an attended step). Idempotent-ish.
set -u
ROOT=/home/etri/Jeongbin/VLA_nuReasoning_mini-project
cd "$ROOT" || exit 1

echo "[monitor] waiting for download_clips.py to finish ..."
while pgrep -f "download_clips.py" >/dev/null; do sleep 60; done
echo "[monitor] download process exited at $(date '+%H:%M:%S')"

N8=$(ls -d /home/etri/DATASET/nureasoning/clips/*/cameras/CAM_M_B 2>/dev/null | wc -l)
NCLIP=$(ls -d /home/etri/DATASET/nureasoning/clips/*/metadata.json 2>/dev/null | wc -l)
echo "[monitor] clips on disk: $NCLIP ; with 8th view (CAM_M_B): $N8"

# Drop the now-redundant probe extraction (the probe clip is consolidated into clips/
# with 8 views; a leftover front-only copy would win manifest dedup and corrupt it).
PROBE=data/raw/_probe_extract/2023.07.31.23.53.58_KMHKM4AEXM1P98033_169ca87cd5ca5fdb912f770aa679edd7
if [ -d /home/etri/DATASET/nureasoning/clips/2023.07.31.23.53.58_KMHKM4AEXM1P98033_169ca87cd5ca5fdb912f770aa679edd7/cameras/CAM_M_B ]; then
  rm -rf data/raw/_probe_extract data/raw/_probe_dl
  echo "[monitor] removed redundant _probe_extract / _probe_dl"
fi

# Archive the old front-only eval results before the 8-view rebuild overwrites the pipeline.
mkdir -p results/archive_frontonly
cp -n results/zeroshot_metrics.json     results/archive_frontonly/ 2>/dev/null
cp -n results/finetuned_metrics.json    results/archive_frontonly/ 2>/dev/null
cp -n results/zeroshot_predictions.jsonl  results/archive_frontonly/ 2>/dev/null
cp -n results/finetuned_predictions.jsonl results/archive_frontonly/ 2>/dev/null
cp -n data/sft/train.jsonl results/archive_frontonly/train_frontonly.jsonl 2>/dev/null
cp -n data/sft/val.jsonl   results/archive_frontonly/val_frontonly.jsonl 2>/dev/null
echo "[monitor] archived front-only results -> results/archive_frontonly/"

cd "$ROOT/src" || exit 1
echo "[monitor] rebuilding 8-view eval manifest ..."
python -m evaluation.eval_set "$ROOT/data/raw" --out "$ROOT/data/eval/zeroshot_manifest.jsonl" 2>&1 | tail -20
echo "[monitor] rebuilding 8-view SFT train/val ..."
python -m sft_data.build_sft --out "$ROOT/data/sft" 2>&1 | tail -25

echo "[monitor] REBUILD DONE at $(date '+%H:%M:%S')"
