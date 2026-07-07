#!/usr/bin/env bash
# 텍스트 SFT 학습(objective=text_sft, QLoRA) — torchrun DDP 멀티-GPU 런처.
# Trainer가 torchrun 환경(LOCAL_RANK/WORLD_SIZE)을 인식해 GPU당 1프로세스 DDP로 학습한다.
# GPU0 결함 → 기본 0,1,2,3 사용(DECISIONS.md). 어댑터는 models/text_sft/에 저장.
#
# 사용:
#   scripts/train_text_sft.sh                         # 기본: GPUS=0,1,2,3 → 4 GPU DDP
#   GPUS=2,3 scripts/train_text_sft.sh                # GPUS 개수만큼 프로세스 수 자동(=2)
#   scripts/train_text_sft.sh --epochs 3 --limit 50   # 추가 인자는 그대로 전달
#   NPROC=2 GPUS=0,1,2,3 scripts/train_text_sft.sh    # (드묾) GPU<프로세스 수동 오버라이드
set -euo pipefail
cd "$(dirname "$0")/../src"          # -m training.* 가 패키지로 잡히도록 src에서 실행

GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
# 프로세스 수 = GPUS의 쉼표 구분 개수(자동). NPROC를 직접 주면 그 값으로 오버라이드.
NPROC="${NPROC:-$(awk -F, '{print NF}' <<<"$GPUS")}"

CUDA_VISIBLE_DEVICES="$GPUS" \
torchrun --standalone --nproc_per_node="$NPROC" \
  -m training.train_text_sft \
  --epochs "${EPOCHS:-2}" \
  --lora-rank "${LORA_RANK:-16}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --grad-accum "${GRAD_ACCUM:-8}" \
  "$@"

# 단일 학습
# CUDA_VISIBLE_DEVICES=0 python -m training.train_text_sft --epochs 2 --lora-rank 16 --batch-size 1 --grad-accum 8
