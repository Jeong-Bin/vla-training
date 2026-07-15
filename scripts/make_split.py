#!/usr/bin/env python3
"""clips-root의 전체 clip을 seed 기반으로 train/val 90/10 분할해 val manifest를 새로 쓴다.

split은 build_sft가 `zeroshot_manifest.jsonl`의 **clip_token 집합**으로 정한다(manifest에 있으면 val,
없으면 train). 따라서 이 스크립트는 **val clip의 clip_token만** manifest에 기록하면 충분하다.
각 clip의 clip_token은 metadata.json에서 읽는다.

⚠️ clips-root의 **현재 clip 전체**를 기준으로 나눈다 → part_2/3까지 모두 받은 **뒤** 실행해야
최종 90/10이 맞다. 재현성: 같은 --seed면 같은 분할(정렬 후 셔플).

사용:
  # dry-run: 분할 통계만(manifest 안 씀)
  python scripts/make_split.py --clips-root /home/etri/DATASET/nureasoning/clips
  # 실제로 manifest 갱신(기존 백업 후 교체)
  python scripts/make_split.py --clips-root /home/etri/DATASET/nureasoning/clips --apply
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "data" / "eval" / "zeroshot_manifest.jsonl"


def clip_tokens(clips_root: str):
    """clips-root의 각 clip 디렉터리 metadata.json에서 clip_token 수집 → 정렬 리스트(재현성)."""
    toks = []
    for name in sorted(os.listdir(clips_root)):
        mp = os.path.join(clips_root, name, "metadata.json")
        if os.path.isfile(mp):
            try:
                with open(mp) as fh:
                    toks.append(json.load(fh)["clip_token"])
            except Exception as e:
                print(f"  ⚠️ metadata 읽기 실패 {name}: {e}")
    return sorted(set(toks))                              # 정렬 → 셔플 전 결정적 순서


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", required=True)
    ap.add_argument("--val-frac", type=float, default=0.10, help="val 비율(기본 0.10=10%)")
    ap.add_argument("--seed", type=int, default=12345, help="셔플 시드(재현성). vlm.SEED와 맞춤.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="val manifest 출력 경로")
    ap.add_argument("--apply", action="store_true", help="실제로 manifest 갱신(미지정=dry-run)")
    args = ap.parse_args()

    toks = clip_tokens(args.clips_root)
    n = len(toks)
    if n == 0:
        sys.exit("clip_token을 하나도 못 읽음 — clips-root 확인.")
    order = list(toks)
    random.Random(args.seed).shuffle(order)              # 정렬된 리스트를 seed로 셔플(재현 가능)
    n_val = max(1, round(n * args.val_frac))
    val_toks = sorted(order[:n_val])
    n_train = n - n_val

    print(f"전체 clip {n} | seed={args.seed} val_frac={args.val_frac}")
    print(f"  → train {n_train} ({100*n_train/n:.1f}%) / val {n_val} ({100*n_val/n:.1f}%)")
    print(f"  val 예시 token: {val_toks[:3]}")

    if not args.apply:
        print("\n  dry-run: manifest 미갱신. --apply로 실제 기록.")
        return

    mpath = Path(args.manifest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    if mpath.exists():                                   # 기존 manifest 백업(덮어쓰기 전)
        bak = mpath.with_suffix(mpath.suffix + f".bak.{int(time.time())}")
        mpath.replace(bak)
        print(f"  기존 manifest 백업 → {bak.name}")
    with open(mpath, "w") as fh:                          # build_sft는 clip_token만 참조
        for t in val_toks:
            fh.write(json.dumps({"clip_token": t}) + "\n")
    print(f"  val manifest 기록: {mpath} ({n_val}줄)")
    print("  → 이제 build_sft 재빌드하면 이 split이 반영됨:")
    print("     python -m sft_data.build_sft --clips-root", args.clips_root)


if __name__ == "__main__":
    main()
