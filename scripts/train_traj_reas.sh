#!/usr/bin/env bash
# 궤적 + reasoning 공동 학습(objective=trajectory) — torchrun DDP 멀티-GPU 런처.
# 각 rank가 자기 GPU에 VLM+DiT를 두고, 데이터를 rank별로 샤딩, grad는 매 step all-reduce 동기화한다.
# GPU0 결함 → 기본 0,1,2,3 사용(DECISIONS.md). 결과는 models/trajectory/에 저장.
#
# 사용:
# GPUS=0,1,2,3,4,5,6,7 scripts/train_traj_reas.sh --vlm-mode lora --epochs 10 --batch-size 1 --grad-accum 8 --reasoning-types spatial,decision,counterfactual
set -euo pipefail
cd "$(dirname "$0")/../src"          # -m training.* 가 패키지로 잡히도록 src에서 실행

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
# 프로세스 수 = GPUS의 쉼표 구분 개수(자동). NPROC를 직접 주면 그 값으로 오버라이드.
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"

CUDA_VISIBLE_DEVICES="$GPUS" \
torchrun --standalone --nproc_per_node="$NPROC" \
  -m training.train_traj_reas \
  --vlm-mode "${VLM_MODE:-full}" \
  --reas-weight "${REAS_WEIGHT:-0.5}" \
  --epochs "${EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-1}" \
  "$@"

# 단일 학습
# CUDA_VISIBLE_DEVICES=0 python -m training.train_traj_reas --vlm-mode full --reas-weight 0.5 --epochs 5 --batch-size 1