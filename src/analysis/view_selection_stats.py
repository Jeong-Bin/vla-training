"""selective-view 프로젝트 필요성 검증 통계.

아이디어(maneuver-conditioned sparse surround perception): 대부분 직진이므로 8뷰를 항상 넣는 건 비효율.
직진엔 전면 위주 소수 뷰만, 회전·차선변경 때만 관련 뷰를 추가로 켠다. 이게 실제로 이득이 되려면:

  (A) 주행의 대다수가 정말 "직진"인가?  → Driving decision.Lateral 분포(즉각 기동 7종)
      ⚠️ mission_goal.command(GO_STRAIGHT/TURN_L/R 3종)는 route 단위라 즉각 기동(특히 차선변경)을 못 잡음.
         대신 reasoning 키프레임의 Driving decision.Lateral(No lateral action / lane change / turn / slightly move)
         을 쓴다. 단 Lateral은 **키프레임(clip당 ~3개, 0.2Hz)에만** 있으므로 [A]/[C]는 그 프레임 대상.
  (B) 8개 뷰 각각에 유의미한 객체가 얼마나 자주 있나(=끌 수 있는 뷰가 많은가)?
      → Spatial.per_camera_results[cam].objects 존재율/평균 개수 (reasoning 프레임 = 8뷰 GT가 있는 프레임)
      + 기동 타입(Lateral)별로 뷰별 객체 존재율이 실제로 갈리는가(직진이면 측·후면이 비는가,
         좌차선변경이면 left/back_left가 뜨는가)?

원본 clip 디렉터리(pkl/json 직접 파싱)를 순회해 위 통계를 계산하고 로그로 출력한다.
결과 로그: results/analysis/view_selection_stats_<timestamp>.log (+ 콘솔).

Run:  python -m analysis.view_selection_stats [--clips-root <dir>] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_CLIPS = Path("/home/etri/DATASET/nureasoning/clips")
RESULTS = REPO / "results" / "analysis"

# per_camera_results의 8개 카메라(고정 순서). front 계열 3개가 "직진 기본 뷰" 후보.
CAMERAS = ["front", "front_left", "front_right", "left", "right", "back", "back_left", "back_right"]
FRONT_VIEWS = {"front", "front_left", "front_right"}          # 직진 시 유지할 전면 3뷰 후보

# Driving decision.Lateral(즉각 횡방향 기동) → 대분류. 실측 7종:
#   "No lateral action" / "Slightly move left|right in the lane" / "Turn left|right" / "Left|Right lane change".
#   selective-view 관점에선 방향(left/right)이 핵심이라 방향까지 보존한 세부 라벨을 쓴다.
def lateral_class(lateral: str) -> str:
    s = (lateral or "").strip().lower()
    if not s or "no lateral" in s:
        return "straight"                     # 직진(횡방향 동작 없음)
    left = "left" in s
    right = "right" in s
    if "lane change" in s:
        return "lane_change_left" if left else "lane_change_right" if right else "lane_change"
    if "turn" in s:
        return "turn_left" if left else "turn_right" if right else "turn"
    if "slightly move" in s:
        return "nudge_left" if left else "nudge_right" if right else "nudge"
    return "other"


# 출력 정렬용 표준 순서(있는 것만 출력).
MANEUVER_ORDER = ["straight", "nudge_left", "nudge_right", "turn_left", "turn_right",
                  "lane_change_left", "lane_change_right", "lane_change", "turn", "nudge", "other", "unknown"]


def load_reasoning_frames(clip: Path) -> list:
    """clip의 reasoning JSON들(=8뷰 per_camera GT가 있는 프레임)을 로드."""
    out = []
    rdir = clip / "reasoning"
    if not rdir.is_dir():
        return out
    for rf in sorted(rdir.glob("*.json")):
        try:
            out.append((int(rf.stem), json.loads(rf.read_text())))
        except Exception:
            continue
    return out


# lateral_of: reasoning JSON에서 Driving decision.Lateral(즉각 기동) 문자열 추출(없으면 None).
#   Driving은 dict일 때만 decision 보유(키프레임=clip당 ~3개). spatial-only 프레임은 None → 'unknown'.
def lateral_of(rj: dict) -> "str | None":
    dv = rj.get("Driving")
    if not isinstance(dv, dict):
        return None
    dd = dv.get("Driving decision")
    if isinstance(dd, dict):
        lat = dd.get("Lateral")
        if isinstance(lat, str) and lat.strip():
            return lat.strip()
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", default=str(DEFAULT_CLIPS))
    ap.add_argument("--limit", type=int, default=0, help="clip 수 상한(0=전체)")
    ap.add_argument("--min-objs", type=int, default=1,
                    help="'유의미 객체 존재'로 칠 뷰당 최소 객체 수(기본 1=하나라도 있으면 존재)")
    args = ap.parse_args()

    clips_root = Path(args.clips_root)
    clip_dirs = sorted(c for c in clips_root.iterdir() if c.is_dir())
    if args.limit:
        clip_dirs = clip_dirs[: args.limit]
    if not clip_dirs:
        raise SystemExit(f"no clips under {clips_root}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS / f"view_selection_stats_{ts}.log"
    lines: list = []

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        lines.append(msg)

    # ── 집계 컨테이너 ───────────────────────────────────────────────
    # (A) 기동 분포: 주 신호 = Driving decision.Lateral(키프레임에만). 참고로 mission command도 병행 집계.
    cmd_all = Counter()                      # 참고: 모든 프레임의 mission command(route 단위)
    lat_atom = Counter()                     # Lateral 원자값(키프레임)
    man_lat = Counter()                      # Lateral 대분류(키프레임)
    # (B) 카메라별 유의미 객체 존재/개수 — reasoning 프레임 전체 + 기동별
    cam_present = Counter()                  # cam → (objects>=min_objs인 프레임 수)
    cam_objsum = Counter()                   # cam → 총 객체 수
    n_reas_frames = 0                        # per_camera_results 있는 reasoning 프레임 수(=8뷰 GT)
    n_kf = 0                                 # 그중 Lateral(키프레임) 있는 프레임 수(=[C] 분모)
    # 기동(Lateral)별 카메라 존재율: man → cam → present count, man → 프레임 수
    man_cam_present = defaultdict(Counter)
    man_frame_n = Counter()
    # 활성 뷰 수 분포(유의미 객체 있는 뷰가 프레임당 몇 개인가) — reasoning 프레임 전체 기준
    active_view_hist = Counter()
    # front-only로 충분한가: front 3뷰 밖(측·후면 5뷰)에 객체가 하나도 없는 프레임 비율
    front_sufficient = 0

    n_clip = 0
    for clip in clip_dirs:
        meta_path = clip / "metadata.json"
        if meta_path.is_file():                           # mission command는 참고용으로만 집계
            try:
                meta = json.loads(meta_path.read_text())
                for f in meta.get("frames", []):
                    mg = f.get("mission_goal") or {}
                    cmd = mg.get("command") if isinstance(mg, dict) else None
                    if cmd:
                        cmd_all[cmd] += 1
            except Exception:
                pass

        # reasoning 프레임(8뷰 GT 존재): per_camera_results로 (B) 통계, Lateral로 (A)/(C) 기동 판정
        for _rts, rj in load_reasoning_frames(clip):
            pcr = (rj.get("Spatial") or {}).get("per_camera_results")
            if not isinstance(pcr, dict):
                continue
            n_reas_frames += 1
            lat = lateral_of(rj)                          # Lateral(키프레임에만); None이면 이 프레임은 (A)/(C) 제외
            has_kf = lat is not None
            if has_kf:
                n_kf += 1
                lat_atom[lat] += 1
                man = lateral_class(lat)
                man_lat[man] += 1
                man_frame_n[man] += 1

            active = 0
            side_back_objs = 0                # front 3뷰 밖의 객체 총합
            for cam in CAMERAS:
                v = pcr.get(cam) or {}
                nobj = len(v.get("objects", [])) if isinstance(v, dict) else 0
                cam_objsum[cam] += nobj
                present = nobj >= args.min_objs
                if present:
                    cam_present[cam] += 1
                    active += 1
                    if has_kf:                            # 기동별 카메라 존재율은 Lateral 있는 프레임만
                        man_cam_present[man][cam] += 1
                if cam not in FRONT_VIEWS:
                    side_back_objs += nobj
            active_view_hist[active] += 1
            if side_back_objs == 0:           # 전면 3뷰 밖에 객체 전무 → front-only로 충분한 프레임
                front_sufficient += 1
        n_clip += 1

    # ── 출력 ───────────────────────────────────────────────────────
    log("=" * 70)
    log(f"selective-view 필요성 검증 통계  |  clips={n_clip}  reasoning_frames={n_reas_frames}")
    log(f"  clips-root: {clips_root}")
    log(f"  '유의미 객체 존재' 기준: 뷰당 objects >= {args.min_objs}")
    log("=" * 70)

    # (A) 기동 분포 — 주 신호: Driving decision.Lateral(즉각 횡방향 기동, 키프레임에만)
    tot_lat = sum(man_lat.values())
    log(f"\n[A] Driving decision.Lateral — 즉각 기동 분포 (키프레임 n={n_kf})")
    log(f"    (커버리지: reasoning 프레임 {n_reas_frames}개 중 Lateral 있는 키프레임 {n_kf}개 "
        f"= {100*n_kf/max(1,n_reas_frames):.1f}%; 나머지는 spatial-only라 즉각 기동 라벨 없음)")
    for man in MANEUVER_ORDER:
        n = man_lat.get(man, 0)
        if n:
            log(f"    {man:18s}: {n:6d}  ({100*n/max(1,tot_lat):5.1f}%)")
    log("\n    (원자 Lateral 값)")
    for v, n in lat_atom.most_common():
        log(f"      {v:34s}: {n:6d}  ({100*n/max(1,tot_lat):5.1f}%)")

    log("\n[A-ref] 참고: mission_goal.command (route 단위 — 즉각 기동·차선변경 못 잡음)")
    tot_cmd = sum(cmd_all.values())
    for cmd, n in cmd_all.most_common(8):
        log(f"      {cmd:24s}: {n:7d}  ({100*n/max(1,tot_cmd):5.1f}%)")

    # (B) 카메라별 유의미 객체
    log(f"\n[B] 카메라별 유의미 객체 존재율 (reasoning 프레임 n={n_reas_frames})")
    log(f"    {'camera':12s}  {'존재율':>8s}  {'평균객체수':>10s}")
    for cam in CAMERAS:
        pres = cam_present[cam]
        rate = 100 * pres / max(1, n_reas_frames)
        avg = cam_objsum[cam] / max(1, n_reas_frames)
        tag = " (front)" if cam in FRONT_VIEWS else ""
        log(f"    {cam:12s}  {rate:7.1f}%  {avg:10.2f}{tag}")

    # 활성 뷰 수 분포 + front-only 충분성
    log(f"\n[B'] 프레임당 '유의미 객체 있는 뷰' 개수 분포")
    for k in sorted(active_view_hist):
        n = active_view_hist[k]
        log(f"    {k}개 뷰 활성: {n:7d}  ({100*n/max(1,n_reas_frames):5.1f}%)")
    log(f"\n    전면 3뷰(front/front_left/front_right) 밖에 객체가 전무한 프레임 "
        f"= front-only 충분: {front_sufficient}/{n_reas_frames} "
        f"({100*front_sufficient/max(1,n_reas_frames):.1f}%)")

    # (C) 기동(Lateral)별 카메라 존재율 — 직진이면 측·후면이 비는가? 좌차선변경이면 left계열이 뜨는가?
    log(f"\n[C] 즉각 기동(Lateral)별 카메라 유의미-객체 존재율 (핵심 검증, 키프레임 n={n_kf})")
    header = "    " + f"{'maneuver':18s}" + "".join(f"{c[:9]:>10s}" for c in CAMERAS)
    log(header)
    for man in MANEUVER_ORDER:
        fn = man_frame_n.get(man, 0)
        if fn == 0:
            continue
        row = f"    {man:18s}"
        for cam in CAMERAS:
            r = 100 * man_cam_present[man][cam] / fn
            row += f"{r:9.1f}%"
        log(f"{row}   (n={fn})")

    # ── 결론 요약 ──────────────────────────────────────────────────
    straight_pct = 100 * man_lat.get("straight", 0) / max(1, tot_lat)
    log("\n" + "=" * 70)
    log("[요약] 프로젝트 필요성 지표")
    log(f"  · 직진(No lateral action) 비율: {straight_pct:.1f}%  (Lateral 기준, 키프레임)  → 높을수록 selective-view 이득")
    log(f"  · front-only 충분 프레임: {100*front_sufficient/max(1,n_reas_frames):.1f}%  "
        f"→ 측·후면을 꺼도 되는 프레임 비율(reasoning 프레임 전체)")
    avg_active = sum(k*v for k,v in active_view_hist.items()) / max(1, n_reas_frames)
    log(f"  · 프레임당 평균 활성 뷰: {avg_active:.2f}/8  "
        f"→ 8뷰 항상 대비 이론적 뷰 절감 여지")
    log("=" * 70)

    log_path.write_text("\n".join(lines))
    print(f"\nwrote {log_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
