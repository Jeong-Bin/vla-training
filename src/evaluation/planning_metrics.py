"""nuReasoning Table 3 planning metrics — NPS와 그 하위 점수(NC/DA/EP/CF/HL).

논문(arXiv:2605.31572, Table 3)의 **nuReasoning Planning Score(NPS)**를 미니프로젝트 데이터로 재현한다.
NPS는 안전 게이트(충돌·주행가능영역)를 곱한 뒤 진행/편안함/사람다움의 가중합을 매기는 합성 점수:

    NPS = s_NC · s_DA · (w_EP·s_EP + w_CF·s_CF + w_HL·s_HL) / (w_EP + w_CF + w_HL)
          (논문 가중치 w_EP=0.3, w_CF=0.2, w_HL=0.5; 합=1.0)

하위 점수(전부 0~1, 높을수록 좋음):
  - **NC** (No at-fault Collision): 예측 궤적을 따라간 ego 박스가, 등속 전개한 주변 객체 박스와 **ego 과실**로
    충돌하면 0, 아니면 1. (게이트)
  - **DA** (Drivable-area Compliance): 모든 waypoint가 HD맵 주행가능영역(차선∪교차로∪road_block) 안이면 1,
    벗어나면 0. (게이트)
  - **EP** (Ego Progress): GT 경로를 따라 예측이 진행한 호 길이 / GT 총 호 길이, [0,1] clip. 정지 장면(GT<1m)은 1.
  - **CF** (Comfort): 예측 궤적의 종/횡 가속도·jerk·yaw-rate가 nuPlan 편안함 한계 내면 1, 초과면 0.
  - **HL** (Human-likeness): waypoint별 변위 d_k에 대해 mean exp(-d_k/D0) (D0=2m). GT 모사도(=ADE의 부드러운 [0,1]판).

좌표계: 예측/GT waypoints는 **ego-frame** (fwd=+x 전방, left=+y 좌측, 미터). HD맵·객체는 global UTM이라
`global_to_ego`로 옮긴다(`bev.scene_context`로 원본 클립/프레임을 되찾아 맵·ego pose·객체를 얻음).

의존성: **순수 numpy만 사용**(shapely 불필요). 점-내부 판정(ray casting)·OBB 교차(분리축 정리, SAT)·
폴리라인 투영을 직접 구현 → 학습/평가 conda 환경(`vla`)에 추가 패키지 없이 동작.

⚠️ 논문과의 차이(미니 근사 — 로그/문서에 명시):
  - EgoState에 차량 dimensions가 없어 **고정 ego 박스**(4.6×2.0m)를 쓴다.
  - 주변 객체의 미래 위치는 **등속(constant-velocity) 전개**로 근사(closed-loop 시뮬레이터 아님).
  - HL의 정확한 논문 공식은 미공개라 변위 기반 exp 점수로 근사. CF 임계값은 nuPlan 표준값.
  - dt = 1/frame_rate_hz (= 0.1s) → 50 waypoint = 5s horizon(논문과 동일; trajectory_future가 10Hz임을 검증).
  - waypoints에 ~0.8m 위치 노이즈가 있어 CF는 NAVSIM처럼 ~2Hz로 리샘플 후 미분(아래 cf_score 주석).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from nureasoning import global_to_ego
from evaluation.bev import scene_context, _coords

# NPS 가중치(논문 Table 3): 사람다움(HL)에 최대 가중.
W_EP, W_CF, W_HL = 0.3, 0.2, 0.5

# 고정 ego 박스(미터). EgoState에 dimensions가 없어 대표 승용차 footprint를 쓴다.
EGO_L, EGO_W = 4.6, 2.0

# HL의 변위 스케일(미터): d=D0에서 점수 1/e. ADE를 [0,1]로 부드럽게 매핑하는 상수.
HL_D0 = 2.0

# nuPlan 편안함 임계값(절대값, SI). 종가속 한계는 가속/감속 비대칭.
CF_MAX_LON_ACCEL = 2.40      # m/s^2 (가속)
CF_MIN_LON_ACCEL = -4.05     # m/s^2 (감속)
CF_MAX_LAT_ACCEL = 4.89      # m/s^2
CF_MAX_YAW_RATE = 0.95       # rad/s
CF_MAX_YAW_ACCEL = 1.93      # rad/s^2
CF_MAX_ABS_JERK = 8.37       # m/s^3 (가속도 크기의 변화율)

# 주변 객체 고려 반경(m): 이보다 먼 객체는 충돌 후보에서 제외(속도·연산 절약).
_OBJ_RADIUS = 40.0


# ---------------------------------------------------------------------------
# 기하 헬퍼 (순수 numpy)
# ---------------------------------------------------------------------------
def _obb_corners(cx: float, cy: float, heading: float, length: float, width: float) -> np.ndarray:
    """중심(cx,cy)·heading(rad)·크기(length×width)의 oriented box → 모서리 (4,2).

    좌표는 ego-frame (x=fwd, y=left). +x_local(길이 방향)이 heading을 향한다(bev.draw_objects와 동일 규약).
    """
    hl, hw = length / 2.0, width / 2.0
    c, s = np.cos(heading), np.sin(heading)
    local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def _obb_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """두 볼록 사각형(4,2)의 교차 여부 — 분리축 정리(SAT). 분리축이 하나라도 있으면 미교차."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])           # 모서리 법선
            pa, pb = a @ axis, b @ axis                    # 두 박스를 축에 투영
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False                               # 분리축 발견 → 미교차
    return True


def _pts_in_poly(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """폴리곤 내부 판정(ray casting). pts(N,2)가 poly(M,2) 안인지 bool(N,). N점 한번에 벡터화."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    M = len(poly)
    j = M - 1
    for i in range(M):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= cond
        j = i
    return inside


def _project_arclen(pt: np.ndarray, line: np.ndarray) -> float:
    """점 pt를 폴리라인 line(M,2)에 투영했을 때, 가장 가까운 점까지의 **호 길이**(시작점 기준)."""
    seg = line[1:] - line[:-1]                             # (M-1,2)
    seglen = np.linalg.norm(seg, axis=1)                   # (M-1,)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    best_d, best_s = np.inf, 0.0
    for i in range(len(seg)):
        a, ab = line[i], seg[i]
        l2 = float(ab @ ab)
        t = 0.0 if l2 < 1e-9 else float(np.clip((pt - a) @ ab / l2, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.linalg.norm(pt - proj))
        if d < best_d:
            best_d, best_s = d, cum[i] + t * seglen[i]
    return best_s


def _heading_along(traj: np.ndarray, k: int) -> float:
    """waypoint k에서 진행 방향(rad). 인접 변위의 atan2(d_left, d_fwd). 정지면 0(전방)."""
    n = len(traj)
    b = traj[min(k + 1, n - 1)]
    prev = traj[max(k - 1, 0)]
    d = b - prev                                           # 중앙차분(끝점은 전/후방차분)
    if abs(d[0]) < 1e-6 and abs(d[1]) < 1e-6:
        return 0.0
    return float(np.arctan2(d[1], d[0]))


# ---------------------------------------------------------------------------
# 하위 점수
# ---------------------------------------------------------------------------
def drivable_polys(clip, ego_x: float, ego_y: float, ego_yaw: float, rng: float = 60.0) -> list:
    """HD맵 주행가능영역(차선∪교차로∪road_block)을 ego-frame 폴리곤 리스트로. 비면 []."""
    m = clip.map()
    polys = []
    for layer in ("lanes", "road_blocks", "intersections"):
        for el in getattr(m, layer, []) or []:
            g = _coords(el)
            if g is None or len(g) < 3:
                continue
            e = global_to_ego(g, ego_x, ego_y, ego_yaw)    # (M,2) [fwd,left]
            if e[:, 0].min() > rng or e[:, 0].max() < -rng:  # 표시창 밖이면 스킵
                continue
            polys.append(e)
    return polys


def nc_score(pred: np.ndarray, frame, ego_x: float, ego_y: float, ego_yaw: float,
             dt: float) -> float:
    """No at-fault Collision: 등속 전개한 주변 객체와 ego 과실 충돌이 있으면 0, 없으면 1.

    각 미래 step k에서 ego 박스(pred[k], 진행방향 heading)와, 객체를 등속 전개해 ego-frame으로 옮긴
    박스의 교차를 SAT로 검사. 충돌 객체가 ego **앞/옆**(ego-frame fwd > -EGO_L/2)이면 ego 과실로 본다
    (뒤에서 들이받히는 경우는 ego 과실 아님 → 무시). 객체가 없으면 1.
    """
    ann = frame.annotations()
    objs = getattr(ann, "objects", None) if ann is not None else None
    if not objs:
        return 1.0
    cand = []                                              # (gx,gy,yaw,vx,vy,l,w) 후보 객체
    for o in objs:
        pose = getattr(o, "pose", None) or {}
        dims = getattr(o, "dimensions", None) or {}
        if not pose or not dims:
            continue
        vel = getattr(o, "velocity", None) or {}
        cand.append((float(pose["x"]), float(pose["y"]), float(pose.get("yaw", 0.0)),
                     float(vel.get("vx", 0.0)), float(vel.get("vy", 0.0)),
                     float(dims.get("l", 1.0)), float(dims.get("w", 1.0))))
    if not cand:
        return 1.0
    n = len(pred)
    for k in range(1, n):                                  # k=0(ego 시작점)은 제외
        ego_box = _obb_corners(float(pred[k, 0]), float(pred[k, 1]),
                               _heading_along(pred, k), EGO_L, EGO_W)
        t = k * dt
        for ox, oy, oyaw, ovx, ovy, ol, ow in cand:
            gx, gy = ox + ovx * t, oy + ovy * t            # 등속 전개(global)
            cf, cl = global_to_ego(np.array([[gx, gy]]), ego_x, ego_y, ego_yaw)[0]
            if cf * cf + cl * cl > _OBJ_RADIUS * _OBJ_RADIUS:
                continue
            if cf < -EGO_L / 2:                            # ego 뒤쪽 객체 → ego 과실 아님
                continue
            obj_box = _obb_corners(cf, cl, oyaw - ego_yaw, ol, ow)
            if _obb_overlap(ego_box, obj_box):
                return 0.0
    return 1.0


def da_score(pred: np.ndarray, polys: list) -> float:
    """Drivable-area Compliance: 모든 waypoint 중심이 주행가능영역(어느 폴리곤이든) 안이면 1, 아니면 0.

    polys가 비면(맵 없음) nan(평균에서 제외).
    """
    if not polys:
        return float("nan")
    inside = np.zeros(len(pred), dtype=bool)
    for poly in polys:
        inside |= _pts_in_poly(pred, poly)
        if inside.all():
            break
    return 1.0 if inside.all() else 0.0


def ep_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Ego Progress: GT 경로를 따라 예측이 도달한 호 길이 / GT 총 호 길이, [0,1] clip.

    GT를 사람 기준 경로로 보고, 예측 종점을 GT 폴리라인에 투영한 호 길이로 진행도를 잰다.
    정지 장면(GT 총 길이 < 1m)은 진행 요구가 없으므로 1.
    """
    seg = float(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum())
    if seg < 1.0:
        return 1.0
    reached = _project_arclen(pred[-1], gt)                # 종점의 호 길이
    return float(np.clip(reached / seg, 0.0, 1.0))


def cf_score(pred: np.ndarray, dt: float) -> float:
    """Comfort: 종/횡 가속도·jerk·yaw-rate·yaw-accel이 nuPlan 한계 내면 1, 초과면 0.

    ⚠️ waypoints에 ~0.8m 위치 노이즈가 있어 10Hz 전체 해상도의 3차 미분(jerk)·yaw는 폭증한다
    (GT조차 jerk≈295 → 항상 0). NAVSIM과 동일하게 **~2Hz로 리샘플**(stride≈0.5s)한 뒤 미분해
    물리적으로 의미 있는 가속/jerk를 얻는다(검증: 2Hz에서 GT jerk≈1.8, 한계 내).
    """
    stride = max(1, round(0.5 / dt))                       # ~2Hz로 리샘플(dt=0.1 → stride 5)
    pred = np.asarray(pred)[::stride]
    dt = dt * stride
    if len(pred) < 4:
        return 1.0
    vel = np.diff(pred, axis=0) / dt                       # (N-1,2) [fwd,left] 속도
    heading = np.arctan2(vel[:, 1], vel[:, 0])             # 진행 방향
    acc = np.diff(vel, axis=0) / dt                        # (N-2,2) 가속도
    h = heading[:-1]                                       # 종/횡 가속도 = 가속도를 heading에 투영
    lon = acc[:, 0] * np.cos(h) + acc[:, 1] * np.sin(h)
    lat = -acc[:, 0] * np.sin(h) + acc[:, 1] * np.cos(h)
    jerk = np.linalg.norm(np.diff(acc, axis=0) / dt, axis=1)  # (N-3,) 가속도 크기 변화율
    dheading = np.arctan2(np.sin(np.diff(heading)), np.cos(np.diff(heading)))  # wrap
    yaw_rate = dheading / dt
    yaw_accel = np.diff(yaw_rate) / dt
    ok = (
        (lon.max(initial=0.0) <= CF_MAX_LON_ACCEL) and (lon.min(initial=0.0) >= CF_MIN_LON_ACCEL)
        and (np.abs(lat).max(initial=0.0) <= CF_MAX_LAT_ACCEL)
        and (jerk.max(initial=0.0) <= CF_MAX_ABS_JERK)
        and (np.abs(yaw_rate).max(initial=0.0) <= CF_MAX_YAW_RATE)
        and (np.abs(yaw_accel).max(initial=0.0) <= CF_MAX_YAW_ACCEL)
    )
    return 1.0 if ok else 0.0


def hl_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Human-likeness: waypoint별 변위 d_k에 대해 mean exp(-d_k/D0). GT 모사도의 [0,1] 점수."""
    d = np.linalg.norm(pred - gt, axis=1)
    return float(np.mean(np.exp(-d / HL_D0)))


# ---------------------------------------------------------------------------
# 한 샘플 종합 + 집계
# ---------------------------------------------------------------------------
def planning_scores(rec: dict, pred: np.ndarray, gt: np.ndarray) -> dict:
    """한 trajectory 샘플의 Table 3 하위 점수 + NPS. 맵/프레임 복구 실패 항목은 nan.

    rec  : SFT 레코드({id, images, ...}) — scene_context로 원본 클립/프레임 복구.
    pred : (N,2) ego-frame 예측 waypoints(미터). gt: (N,2) GT.
    반환 : {"NC","DA","EP","CF","HL","NPS"} (값 0~1, 일부 nan 가능).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    nc = da = float("nan")
    dt = 0.1                                               # 기본(frame_rate_hz로 갱신)
    try:
        clip, frame = scene_context(rec)
        dt = 1.0 / (clip.frame_rate_hz or 10.0)
        if frame is not None:
            pose = frame.ego_state().pose
            ex, ey, eyaw = pose["x"], pose["y"], pose["yaw"]
            nc = nc_score(pred, frame, ex, ey, eyaw, dt)
            da = da_score(pred, drivable_polys(clip, ex, ey, eyaw))
    except Exception:  # noqa: BLE001  (맵/프레임 매칭 실패 → 게이트는 nan)
        pass
    ep = ep_score(pred, gt)
    cf = cf_score(pred, dt)
    hl = hl_score(pred, gt)
    # NPS: 게이트가 nan이면 1.0으로 간주(해당 안전요소 미평가, 페널티 없음). 진행/편안함/사람다움은 항상 계산됨.
    g_nc = 1.0 if np.isnan(nc) else nc
    g_da = 1.0 if np.isnan(da) else da
    nps = g_nc * g_da * (W_EP * ep + W_CF * cf + W_HL * hl)  # 가중치 합=1
    return {"NC": nc, "DA": da, "EP": ep, "CF": cf, "HL": hl, "NPS": float(nps)}


def aggregate(rows: list[dict]) -> dict:
    """샘플별 점수 dict 리스트 → 각 지표 nanmean + NPS 유효 표본 수.

    반환 키: NC_mean/DA_mean/EP_mean/CF_mean/HL_mean/NPS_mean(전부 0~1), n_nps(NPS 평균에 쓴 표본 수).
    """
    if not rows:
        return {}
    out = {}
    for key in ("NC", "DA", "EP", "CF", "HL", "NPS"):
        vals = np.array([r[key] for r in rows], dtype=np.float64)
        out[f"{key}_mean"] = round(float(np.nanmean(vals)), 4) if np.isfinite(vals).any() else float("nan")
    out["n_nps"] = int(np.isfinite([r["NPS"] for r in rows]).sum())
    return out


def format_table(agg: dict, ade: Optional[float] = None) -> str:
    """Table 3 스타일 한 줄(헤더 포함). 높을수록 좋음(ADE만 미터·낮을수록 좋음).

    표시 전용 변환: NC/DA/EP/CF/HL/NPS는 저장값(0~1)에 ×100해 논문 표기(%)와 같은 스케일로, 소수 둘째
    자리까지 표시(agg 자체는 0~1 그대로 — JSON 저장·내부 계산에 영향 없음). ADE(m)만 소수 셋째자리.
    """
    cols = ["NC", "DA", "EP", "CF", "HL", "NPS"]
    head = "  ".join(f"{c:>7}" for c in cols) + ("  " + f"{'ADE(m)':>7}" if ade is not None else "")
    vals = "  ".join(f"{agg.get(c + '_mean', float('nan')) * 100:>7.2f}" for c in cols)
    if ade is not None:
        vals += "  " + f"{ade:>7.3f}"
    return head + "\n" + vals


__all__ = ["planning_scores", "aggregate", "format_table",
           "nc_score", "da_score", "ep_score", "cf_score", "hl_score", "drivable_polys",
           "W_EP", "W_CF", "W_HL"]
