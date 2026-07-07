"""Build a fixed evaluation manifest from extracted clips (shared by zero-shot and fine-tuned eval).

Walks the loader's ``driving`` keyframes (frames with a populated GT decision) that
also have a front camera image, and writes one JSONL record per sample. Freezing the
set to a manifest lets zero-shot and the fine-tuned model be
scored on exactly the same frames.

Record fields:
  id, clip_token, frame_index, clip_dir (rel repo), images [{view, path(rel clip_dir)}],
  front_image (rel clip_dir, kept for back-compat), mission, gt_longitudinal, gt_lateral,
  gt_long_canon, gt_lat_canon, gt_reasoning, gt_counterfactual

val 크기 고정(--n-clips-val): val은 여러 학습 실험을 공정 비교하는 **고정 잣대**라, 한 번 정해
얼려둔다. --n-clips-val N이면 root 클립을 clip_token 정렬→seed 셔플→앞 N개만 val로 뽑고(클립 단위
= 누수 방지), 그 클립들의 driving 프레임만 manifest에 쓴다. 나머지 클립은 build_sft가 train으로
가른다. N=0(기본)이면 root 전체를 val로(레거시: 소규모 후보를 통째로 얼릴 때).

Run:  python -m evaluation.eval_set [ROOT] [--out PATH] [--n-clips-val N] [--seed S]
      (ROOT defaults to data/raw; PATH to data/eval/zeroshot_manifest.jsonl)
      예) python -m evaluation.eval_set /home/etri/DATASET/nureasoning/clips --n-clips-val 100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[2]   # 레포 루트(이 파일 기준 2단계 위)
sys.path.insert(0, str(REPO / "src"))

from nureasoning import CAMERA_VIEWS, Frame, iter_clips, select_keyframes  # noqa: E402
from evaluation import taxonomy as T  # noqa: E402

DEFAULT_ROOT = Path("/home/etri/DATASET/nureasoning/clips")  # 대용량 10Hz 클립 루트(REPO 밖 큰 디스크)
DEFAULT_OUT = REPO / "data" / "eval" / "zeroshot_manifest.jsonl"  # 고정 평가셋 출력 경로(레포 내 산출물)


# val_clip_tokens: root 아래 클립을 clip_token 기준으로 정렬 → seed로 셔플 → 앞 n_clips개를 val로 선정.
#   val을 **클립 단위**로 고르는 이유: 한 클립의 프레임이 train↔val에 걸치면 누수 → 클립째로 가른다.
#   n_clips<=0 이면 전체 클립을 val로(레거시: 얼려둔 소규모 후보 전체). 반환 = val로 쓸 clip_token 집합.
def val_clip_tokens(root: Path, n_clips: int, seed: int) -> set[str]:
    import random
    tokens = sorted({c.clip_token for c in iter_clips(root)})   # 정렬로 플랫폼 무관 결정성 확보
    if n_clips <= 0 or n_clips >= len(tokens):
        return set(tokens)
    rng = random.Random(seed)                                  # seed 고정 → 어떤 실행이든 같은 val
    rng.shuffle(tokens)
    return set(tokens[:n_clips])


# iter_eval_frames: root 아래 클립 중 **val로 선정된 clip_token**만 돌며, GT 주행결정이 채워진
# 'driving' 키프레임 중 front 이미지가 있는 프레임만 내보냄(평가 후보). val_tokens=None이면 전체.
def iter_eval_frames(root: Path, val_tokens: "set[str] | None" = None) -> Iterator[Frame]:
    """Yield driving keyframes (GT decision + front image) from the selected val clips."""
    for clip in iter_clips(root):
        if val_tokens is not None and clip.clip_token not in val_tokens:
            continue
        for f in select_keyframes(clip, policy="driving"):
            if f.has_camera("front"):
                yield f


# frame_to_record: 한 프레임(Frame)을 manifest 한 줄(JSON dict)로 직렬화.
# id/경로/mission/GT 원문(longitudinal·lateral·reasoning)과 표준 매핑값(_canon)을 함께 저장.
# gt_counterfactual: nuReasoning이 라벨링한 '대안 행동 + GT 위험등급'.
# Counterfactual의 Alternative/Top safety-critical actions를 표준 액션으로 매핑하고
# Risk level(Safe/Suboptimal/Unsafe)·Reason과 함께 저장 → 리포트에서 "모델이 택한 행동이
# GT가 Unsafe로 분류한 행동인가"를 판정하는 근거(데이터 출처가 있는 정렬 기준).
def _counterfactual_actions(f: Frame) -> list[dict]:
    cf = f.counterfactual() or {}
    out = []
    for key in ("Alternative actions", "Top safety-critical actions"):
        for a in cf.get(key, []) or []:
            if not isinstance(a, dict):
                continue
            out.append({
                "long": T.map_longitudinal(a.get("Longitudinal", "")),
                "lat": T.map_lateral(a.get("Lateral", "")),
                "risk": a.get("Risk level"),         # Safe | Suboptimal | Unsafe
                "reason": a.get("Reason", ""),
                "source": key,
            })
    return out


def frame_to_record(f: Frame) -> dict:
    dec = f.driving_decision() or {}
    lon_raw, lat_raw = dec.get("Longitudinal", ""), dec.get("Lateral", "")
    front = f.camera_path("front")
    # 디스크에 실제로 존재하는 뷰만 CAMERA_VIEWS 순서로(front는 iter_eval_frames에서 보장).
    images = [{"view": v, "path": str(f.camera_path(v).relative_to(f.clip_dir))}
              for v in CAMERA_VIEWS if f.has_camera(v)]
    return {
        "id": f.sample_id,
        "clip_token": f.clip_token,
        "frame_index": f.frame_index,
        "clip_dir": str(f.clip_dir),                   # 절대경로(REPO 밖 대용량 클립 지원)
        "images": images,                              # [{view, path(clip_dir 기준 상대경로)}]
        "front_image": str(front.relative_to(f.clip_dir)),  # 하위호환용(레거시 단일뷰 경로)
        "mission": f.mission_command,
        "gt_longitudinal": lon_raw,
        "gt_lateral": lat_raw,
        "gt_long_canon": T.map_longitudinal(lon_raw),
        "gt_lat_canon": T.map_lateral(lat_raw),
        "gt_reasoning": f.reasoning_trace() or "",
        "gt_counterfactual": _counterfactual_actions(f),
    }


# build_manifest: 평가 후보 프레임을 모아 JSONL manifest로 저장하고 통계 dict 반환.
# 같은 클립이 두 추출 폴더(_probe_extract, clips)에 겹칠 수 있어 id로 중복 제거.
# 반환 통계 = 샘플수·클립수·GT 라벨분포·GT 매핑실패수.
def build_manifest(root: Path, out_path: Path, n_clips_val: int = 0, seed: int = 1234) -> dict:
    # val 클립 선정(클립 단위, seed 재현). n_clips_val<=0이면 전체 클립을 val로(레거시 동작).
    val_tokens = val_clip_tokens(root, n_clips_val, seed)
    records, seen = [], set()                 # records: 최종 레코드, seen: 본 id 집합(중복 방지)
    for f in iter_eval_frames(root, val_tokens):  # dedup by id (same clip may sit in two extract dirs)
        if f.sample_id in seen:
            continue
        seen.add(f.sample_id)
        records.append(frame_to_record(f))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    lon_dist = Counter(r["gt_long_canon"] for r in records)  # 종방향 GT 라벨 분포
    lat_dist = Counter(r["gt_lat_canon"] for r in records)   # 횡방향 GT 라벨 분포
    views_dist = Counter(len(r["images"]) for r in records)  # 샘플당 뷰 수 분포(8뷰 정상)
    n = len(records)
    stats = {
        "n_samples": n,
        "n_clips": len({r["clip_token"] for r in records}),
        "n_clips_val_requested": n_clips_val,          # 요청한 val 클립 수(0=전체)
        "seed": seed,                                  # val 선정 시드(재현용)
        "views_per_sample": dict(views_dist.most_common()),
        "long_dist": dict(lon_dist.most_common()),
        "lat_dist": dict(lat_dist.most_common()),
        "gt_unmapped_long": lon_dist.get(T.UNMAPPED, 0),
        "gt_unmapped_lat": lat_dist.get(T.UNMAPPED, 0),
        "out": str(out_path.relative_to(REPO)),
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-clips-val", type=int, default=0,
                    help="val로 고정할 클립 수(클립 단위, 누수 방지). 0=root 아래 전체 클립을 val로"
                         "(레거시). 예: 100 → 셔플 후 앞 100클립만 val, 나머지는 build_sft에서 train.")
    ap.add_argument("--seed", type=int, default=1234,
                    help="val 클립 선정 시드(재현용). 같은 seed·root면 항상 같은 val 집합.")
    args = ap.parse_args()

    stats = build_manifest(Path(args.root), Path(args.out), args.n_clips_val, args.seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["n_samples"] == 0:
        print("\n⚠️  no eval samples found — extract clips first (download pipeline).")


if __name__ == "__main__":
    main()
