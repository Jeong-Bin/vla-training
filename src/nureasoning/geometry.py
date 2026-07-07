"""Ego-frame geometry helpers (shared spine).

nuReasoning의 ``EgoState.trajectory_future``는 **global UTM 좌표**(x, y, yaw)로 저장된다
(예: x≈664472, y≈3996658). 궤적 planning 학습/평가는 ego 차량 기준 **상대 좌표**가 필요하므로
여기서 global→ego 변환을 한 곳에 모은다. 컨벤션은 ``Frame.camera_objects``의 ``center_3d_ego``와
동일하게 맞춘다: **forward(+x) = 전방, left(+y) = 좌측** (오른손 좌표, yaw는 +x축 기준 반시계 라디안).

검증(2026-06-23): trajectory_future[0]을 변환하면 ego 원점 근처(≈0.7m 전방)에서 시작해 forward로
단조 증가하며, 50 waypoint 총거리가 velocity*시간과 일관됨(44m ≈ 6.84m/s × ~6.5s).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def global_to_ego(points_xy: np.ndarray, ego_x: float, ego_y: float, ego_yaw: float) -> np.ndarray:
    """Global UTM (x, y) 점들을 ego-frame (forward, left)로 회전·평행이동.

    points_xy : (N, 2) global UTM 좌표.
    반환       : (N, 2) [forward, left]  (forward=전방+, left=좌측+).
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    d = pts - np.array([ego_x, ego_y], dtype=np.float64)     # ego 원점으로 평행이동
    c, s = np.cos(ego_yaw), np.sin(ego_yaw)
    fwd = d[:, 0] * c + d[:, 1] * s                          # R(-yaw) 적용(heading을 +x로 정렬)
    left = -d[:, 0] * s + d[:, 1] * c
    return np.stack([fwd, left], axis=1)


def future_waypoints_ego(ego_state, n_points: int = 0, stride: int = 1,
                         with_heading: bool = False) -> Optional[np.ndarray]:
    """``EgoState`` → ego-frame 미래 궤적 (n, 2 또는 3).

    ego_state    : ``data_schema.EgoState`` (pose dict + trajectory_future (50,3) [x,y,yaw] global @10Hz).
    n_points     : >0이면 (서브샘플 후) 앞에서 그만큼만 취함(시퀀스 길이 통제). 0이면 전체.
    stride       : 원본 10Hz 궤적을 이 간격으로 서브샘플. 논문 재현(2Hz)은 stride=5 → 미래프레임
                   +5,+10,…,+50(=0.5,1.0,…,5.0s)에서 10개 waypoint. stride=1이면 10Hz 전체(하위호환).
    with_heading : True면 3번째 채널로 ego 상대 heading θ(=yaw_global − ego_yaw, [-π,π] wrap) 추가 →
                   논문의 (x, y, θ) 표현. False면 (forward, left) 2D(하위호환).
    반환         : (n, 2) 또는 (n, 3) ndarray [forward, left, (θ)]. trajectory_future가 없으면 None.
    """
    if ego_state is None:
        return None
    tf = getattr(ego_state, "trajectory_future", None)
    if tf is None or len(tf) == 0:
        return None
    pose = ego_state.pose
    tf = np.asarray(tf, dtype=np.float64)                    # (50, 3) [x, y, yaw] @10Hz
    if stride > 1:                                           # 2Hz면 stride=5 → 프레임 +5,+10,…,+50
        tf = tf[stride - 1::stride]
    ego_xy = global_to_ego(tf[:, :2], pose["x"], pose["y"], pose["yaw"])   # (n,2) [fwd,left]
    if with_heading:                                         # 논문 (x,y,θ): 상대 heading을 3번째 채널로
        rel = tf[:, 2] - pose["yaw"]
        rel = (rel + np.pi) % (2 * np.pi) - np.pi            # [-π,π] wrap
        out = np.concatenate([ego_xy, rel[:, None]], axis=1)  # (n,3) [fwd,left,θ]
    else:
        out = ego_xy
    if n_points and n_points < len(out):
        out = out[:n_points]
    return out


# past_waypoints_ego: 논문 nuVLA의 "ego motion history"(action expert의 state token 구성요소) —
#   EgoState.trajectory_history(과거 pose, global UTM [x,y,yaw] @10Hz)를 **현재 ego-frame**으로 변환.
#   trajectory_history는 [0]=가장 먼 과거(~-3s), [-1]=직전(~-0.1s) 순 → 반전해 **recent→old**로 서브샘플.
#   2Hz(stride=5)면 -5,-10,…프레임의 과거 pose. 과거 부족(클립 초반)이면 가장 먼 pose로 패딩해 고정 길이 보장.
def past_waypoints_ego(ego_state, n_points: int = 0, stride: int = 1,
                       with_heading: bool = False) -> Optional[np.ndarray]:
    if ego_state is None:
        return None
    th = getattr(ego_state, "trajectory_history", None)
    if th is None or len(th) == 0:
        return None
    pose = ego_state.pose
    th = np.ascontiguousarray(np.asarray(th, dtype=np.float64)[::-1])   # recent→old ([-1]직전이 맨앞)
    if stride > 1:
        th = th[stride - 1::stride]                        # 2Hz면 -5,-10,… 프레임
    ego_xy = global_to_ego(th[:, :2], pose["x"], pose["y"], pose["yaw"])   # (n,2) [fwd,left] (뒤쪽=음수)
    if with_heading:
        rel = th[:, 2] - pose["yaw"]
        rel = (rel + np.pi) % (2 * np.pi) - np.pi
        out = np.concatenate([ego_xy, rel[:, None]], axis=1)
    else:
        out = ego_xy
    if n_points:
        if len(out) >= n_points:
            out = out[:n_points]
        elif len(out) > 0:                                 # 과거 부족 → 가장 먼 과거 pose 반복 패딩
            out = np.concatenate([out, np.repeat(out[-1:], n_points - len(out), axis=0)], axis=0)
        else:
            out = np.zeros((n_points, 3 if with_heading else 2), dtype=np.float64)
    return out


# upsample_waypoints: 논문 planning ADE 평가용 — 예측/GT 궤적(T개, ego-frame [fwd,left,…])을
#   Δt=0.1s(10Hz)로 선형 보간해 dense (M,2) [fwd,left]를 만든다(heading θ는 ADE에 불필요해 제외).
#   논문: "5초 horizon at Δt=0.1s, yielding 51 poses including the current pose" → 현재 pose(원점)를
#   t=0에 prepend + 10@2Hz waypoint를 51 poses @10Hz로 보간. in_hz는 len(wp)/horizon으로 자동(10점→2Hz,
#   50점→10Hz 하위호환). planning_metrics는 dt=0.1(10Hz)를 가정하므로 이 dense 궤적을 그대로 먹인다.
def upsample_waypoints(wp, out_hz: float = 10.0, horizon_s: float = 5.0,
                       in_hz: Optional[float] = None, include_current: bool = True) -> np.ndarray:
    wp = np.asarray(wp, dtype=np.float64)[:, :2]           # x,y만(θ 제외)
    n = len(wp)
    if in_hz is None:
        in_hz = n / horizon_s                              # 10점→2Hz, 50점→10Hz 자동
    t_knots = np.arange(1, n + 1) / in_hz                  # waypoint 시각: 1/in_hz,…,horizon
    if include_current:                                    # 현재 pose(ego 원점)를 t=0에 prepend
        t_knots = np.concatenate([[0.0], t_knots])
        wp = np.concatenate([[[0.0, 0.0]], wp], axis=0)
    n_out = int(round(horizon_s * out_hz)) + (1 if include_current else 0)   # 5s*10Hz+1 = 51
    t_out = np.linspace(0.0, horizon_s, n_out)             # 0,0.1,…,5.0
    fwd = np.interp(t_out, t_knots, wp[:, 0])
    left = np.interp(t_out, t_knots, wp[:, 1])
    return np.stack([fwd, left], axis=1)                   # (n_out, 2) @out_hz


__all__ = ["global_to_ego", "future_waypoints_ego", "past_waypoints_ego", "upsample_waypoints"]
