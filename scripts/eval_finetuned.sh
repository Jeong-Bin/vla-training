#!/usr/bin/env bash
# 파인튜닝 어댑터를 '동일 하니스'로 val 재평가 후 zero-shot 대비 비교표 생성.
set -euo pipefail
cd "$(dirname "$0")/../src"

ADAPTER="${ADAPTER:-../models/text_sft}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
python -m evaluation.run_zeroshot --adapter "$ADAPTER" --tag finetuned

python -m evaluation.report --tag finetuned
python -m evaluation.compare --baseline zeroshot --model finetuned
