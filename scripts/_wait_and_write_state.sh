#!/bin/bash
# Wait for the spatial (1Hz) refetch download to finish, then write a completion message
# + timestamp to downlode_state.txt. Run as a FILE (cmdline = this script's path, which
# does NOT contain "download_clips.py") so the pgrep wait can't match itself.
set -u
ROOT=/home/etri/Jeongbin/VLA_nuReasoning_mini-project
cd "$ROOT" || exit 1

while pgrep -f "download_clips.py" >/dev/null 2>&1; do sleep 60; done

TS=$(date '+%Y-%m-%d %H:%M:%S')
NCLIP=$(ls -d /home/etri/DATASET/nureasoning/clips/*/metadata.json 2>/dev/null | wc -l)

# 1Hz(spatial) 프레임이 채워진 클립 수: front 카메라에 이미지가 10장 이상(키프레임 3장보다 많음)이면 spatial 추출 완료로 간주(≈13 기대).
SPDONE=0
for f in /home/etri/DATASET/nureasoning/clips/*/cameras/CAM_M_F; do
  n=$(ls "$f" 2>/dev/null | wc -l)
  [ "$n" -ge 10 ] && SPDONE=$((SPDONE + 1))
done
DISK=$(du -sh /home/etri/DATASET/nureasoning/clips 2>/dev/null | cut -f1)

{
  echo "160개 클립 다운로드 완료"
  echo "완료 시간: $TS"
  echo ""
  echo "[방식] A — spatial 1Hz 주석 프레임 × 8뷰 (증분 다운로드; 기존 키프레임 이미지 보존)"
  echo "[검증] 전체 클립 $NCLIP/160 | 1Hz 프레임 완성 클립 $SPDONE | clips 용량 $DISK"
} > downlode_state.txt

echo "[wait] state file written at $TS (clips=$NCLIP, spatial-complete=$SPDONE)"
