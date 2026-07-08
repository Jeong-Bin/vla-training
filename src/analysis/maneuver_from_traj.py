"""궤적 유도 기동 라벨(설계 C)의 타당성 검증.

목표: trajectory_future의 횡방향 변위로 기동(직진/좌·우)을 판정하는 규칙이, 키프레임의
Driving decision.Lateral(사람/모델이 단 텍스트 라벨)과 얼마나 일치하는가?
→ 일치하면 이 궤적 유도 규칙을 전 planning 프레임(30~150)에 적용해 baseline·selective를
   동일 프레임에서 공정 비교할 수 있다(Lateral의 키프레임 한정 문제 해소).

기동 유도 규칙(ego-frame 미래 5초 궤적, left>0=좌):
  - |max lateral| < THR_STRAIGHT           → straight
  - lane change 급의 큰 횡변위(> THR_LANE) → lane_change_{left|right}
  - 중간(회전/회피)                          → turn_or_nudge_{left|right}
임계값은 Lateral 라벨과의 일치를 최대화하도록 스윕해 본다.

Run:  python -m analysis.maneuver_from_traj [--clips-root <dir>] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nureasoning.geometry import future_waypoints_ego   # noqa: E402  (build와 동일 ego-frame 변환)

DEFAULT_CLIPS = Path("/home/etri/DATASET/nureasoning/clips")
RESULTS = REPO / "results" / "analysis"


def load_pkl(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)


# lateral_class: Driving decision.Lateral 텍스트 → {straight, left, right} 3분류(방향만; 검증 단순화).
def lateral_dir(lateral: str) -> "str | None":
    s = (lateral or "").strip().lower()
    if not s:
        return None
    if "no lateral" in s:
        return "straight"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return None


# traj_dir: ego-frame 미래궤적의 최종 lateral 변위 부호로 {straight, left, right} 판정(임계 thr).
def traj_dir(wp, thr: float) -> str:
    # wp: (T,3) [fwd,left,θ]. 최종 시점 lateral(=left 성분)로 방향, |·|<thr면 직진.
    lat = float(wp[-1][1])
    if abs(lat) < thr:
        return "straight"
    return "left" if lat > 0 else "right"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", default=str(DEFAULT_CLIPS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    clips_root = Path(args.clips_root)
    clip_dirs = sorted(c for c in clips_root.iterdir() if c.is_dir())
    if args.limit:
        clip_dirs = clip_dirs[: args.limit]

    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"maneuver_from_traj_{ts}.log"
    lines: list = []

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        lines.append(msg)

    # 키프레임에서 (Lateral 방향 라벨, 미래궤적 wp)를 모은다.
    pairs = []          # (label_dir, final_lateral, max_abs_lateral, final_theta_deg)
    n_kf = n_used = 0
    for clip in clip_dirs:
        ann_ts = sorted(int(p.stem) for p in (clip / "annotations").glob("*.pkl"))
        for rf in sorted((clip / "reasoning").glob("*.json")):
            try:
                rj = json.loads(rf.read_text())
            except Exception:
                continue
            dv = rj.get("Driving")
            if not isinstance(dv, dict):
                continue
            dd = dv.get("Driving decision")
            lat_txt = dd.get("Lateral") if isinstance(dd, dict) else None
            ld = lateral_dir(lat_txt) if lat_txt else None
            if ld is None:
                continue
            n_kf += 1
            # 이 키프레임의 ego_state → 미래궤적
            fi = rj.get("frame_index")
            if fi is None or fi >= len(ann_ts):
                continue
            try:
                e = load_pkl(clip / "ego_state" / f"{ann_ts[fi]}.pkl")
                wp = future_waypoints_ego(e, n_points=10, stride=5, with_heading=True)
            except Exception:
                wp = None
            if wp is None:
                continue
            n_used += 1
            final_lat = float(wp[-1][1])
            max_lat = max(abs(float(r[1])) for r in wp)
            theta = float(wp[-1][2]) * 57.2958
            pairs.append((ld, final_lat, max_lat, theta))

    log("=" * 70)
    log("궤적 유도 기동 라벨(설계 C) 타당성 검증")
    log(f"  clips-root: {clips_root}  |  clips={len(clip_dirs)}")
    log("=" * 70)
    log(f"\n키프레임 {n_kf}개 중 궤적 확보 {n_used}개로 검증\n")

    # 라벨 방향 분포
    lab = Counter(p[0] for p in pairs)
    log("Lateral 텍스트 라벨 방향 분포: " + str(dict(lab)))

    # 라벨별 최종 lateral 변위 통계(궤적이 라벨을 뒷받침하는가)
    log("\n라벨별 최종 lateral 변위(m) 통계:")
    for d in ["straight", "left", "right"]:
        vals = [p[1] for p in pairs if p[0] == d]
        if not vals:
            continue
        vals_sorted = sorted(vals)
        med = vals_sorted[len(vals) // 2]
        mean = sum(vals) / len(vals)
        log(f"  {d:9s}: n={len(vals):5d}  mean={mean:7.2f}  median={med:7.2f}  "
            f"min={min(vals):7.2f}  max={max(vals):7.2f}")

    # 임계값 스윕: |final_lateral|<thr → straight, 아니면 부호. 라벨과 3분류 일치율.
    log("\n임계값(thr) 스윕 — 궤적 유도 vs Lateral 텍스트 3분류(straight/left/right) 일치율:")
    log(f"  {'thr(m)':>7s}  {'전체정확도':>10s}  {'straight재현':>12s}  {'left재현':>10s}  {'right재현':>10s}")
    best = None
    for thr in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        correct = 0
        per_recall = defaultdict(lambda: [0, 0])   # dir → [correct, total]
        for ld, final_lat, max_lat, theta in pairs:
            pred = "straight" if abs(final_lat) < thr else ("left" if final_lat > 0 else "right")
            per_recall[ld][1] += 1
            if pred == ld:
                correct += 1
                per_recall[ld][0] += 1
        acc = correct / max(1, len(pairs))

        def rec(d):
            c, t = per_recall[d]
            return 100 * c / t if t else 0.0
        log(f"  {thr:7.2f}  {100*acc:9.1f}%  {rec('straight'):11.1f}%  "
            f"{rec('left'):9.1f}%  {rec('right'):9.1f}%")
        if best is None or acc > best[1]:
            best = (thr, acc)

    log(f"\n최적 thr={best[0]}m (전체 정확도 {100*best[1]:.1f}%)")
    log("→ 정확도가 높으면(예: >80%) 궤적 유도 라벨을 전 planning 프레임(30~150)에 적용해도 신뢰 가능(설계 C).")

    log_path.write_text("\n".join(lines))
    print(f"\nwrote {log_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
