"""BEV(조감도) 시각화 — HD맵 + 객체 GT 박스 + 예측/GT 궤적을 ego-frame 한 장에 겹쳐 그린다.

trajectory 평가의 정성 보조: 빈 좌표축에 궤적만 찍는 대신, 그 장면의 **차선·경계·횡단보도·신호등**(HD맵)과
**주변 객체 3D 박스**(GT) 위에 예측·정답 궤적을 올려 "모델이 도로 맥락에서 말이 되는 경로를 내는가"를
눈으로 확인하게 한다.

좌표계: 전부 **ego-frame** (forward=+x 전방, left=+y 좌측)으로 통일.
  - 궤적·객체(`corners_3d_ego`)는 이미 ego-frame.
  - HD맵은 **global UTM**이라 `global_to_ego(pose)`로 변환해야 한다.
그림은 보기 좋게 가로축=left(+왼쪽), 세로축=forward(+위) → 위쪽이 차량 전방.

SFT 레코드(`rec`)에는 맵/객체가 없으므로, 이미지 path로 원본 클립·프레임을 되찾아(`scene_context`)
맵·ego pose·객체를 가져온다.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from nureasoning import global_to_ego, parse_clip

REPO = Path(__file__).resolve().parents[2]

# 맵 레이어별 스타일(선색, 선두께, 채움). lanes는 폴리곤 채움, 나머지는 폴리라인.
_MAP_STYLE = {
    "lanes":          dict(color="#cccccc", lw=0.6, fill="#f2f2f2"),  # 차선 영역
    "boundaries":     dict(color="#888888", lw=0.8, fill=None),       # 도로 경계
    "crosswalks":     dict(color="#4a90d9", lw=1.0, fill="#dbeafe"),  # 횡단보도
    "stop_polygons":  dict(color="#e07b39", lw=0.8, fill=None),       # 정지선 영역
    "traffic_lights": dict(color="#d64545", lw=0.0, fill=None),       # 신호등(점)
}


# scene_context: SFT 레코드 → (clip, frame). 이미지 path로 클립 폴더를 찾고, 파일명 timestamp로 프레임 매칭.
# 반환 (clip, frame) 또는 매칭 실패 시 (clip, None). 맵은 clip.map(), pose/객체는 frame에서.
def scene_context(rec: dict):
    img_path = rec["images"][0]["path"]                    # e.g. /home/etri/DATASET/nureasoning/clips/<clip>/cameras/CAM_M_F/CAM_M_F_<ts>.jpg
    parts = Path(img_path).parts
    ci = parts.index("clips")
    clip_dir = REPO / Path(*parts[: ci + 2])               # /home/etri/DATASET/nureasoning/clips/<clip>
    clip = parse_clip(clip_dir)
    m = re.search(r"_(\d+)\.jpg$", Path(img_path).name)     # 카메라 파일명의 timestamp
    ts = int(m.group(1)) if m else None
    frame = None
    if ts is not None:
        # front 카메라 timestamp가 가장 가까운 프레임 선택. ⚠️ 인접 두 프레임이 **동일 front
        # timestamp**를 갖는 경우가 있어(원본 데이터 특성), 그중 하나만 Spatial(객체)을 보유한다.
        # → 동률 타이브레이크로 **has_spatial 프레임을 우선**(키 = (거리, has_spatial 아님)).
        #    안 그러면 객체 없는 쌍둥이 프레임에 매칭돼 BEV에 객체가 안 그려진다.
        best, best_key = None, None
        for fr in clip:
            try:
                p = fr.camera_path("front")
            except Exception:
                p = None
            if not p:
                continue
            mm = re.search(r"_(\d+)\.jpg$", Path(p).name)
            if not mm:
                continue
            key = (abs(int(mm.group(1)) - ts), 0 if fr.has_spatial() else 1)
            if best_key is None or key < best_key:
                best, best_key = fr, key
        frame = best
    return clip, frame


# _ego_pose: frame.ego_state().pose(dict)에서 (x, y, yaw) global 추출.
def _ego_pose(frame):
    pose = frame.ego_state().pose
    return pose["x"], pose["y"], pose["yaw"]


# _coords: 맵 요소에서 좌표 리스트를 꺼낸다(레이어마다 속성명이 다름: lanes=polygon, 그 외=geometry).
def _coords(elem):
    raw = getattr(elem, "polygon", None) or getattr(elem, "geometry", None)
    if not raw:
        return None
    return np.asarray(raw, dtype=np.float64)[:, :2]        # x,y만(z 버림)


# draw_map: ego-frame으로 변환한 HD맵 레이어들을 ax에 그린다. clip.map()의 모든 레이어 + ego pose 필요.
def draw_map(ax, clip, ego_x, ego_y, ego_yaw, rng=60.0):
    import matplotlib.patches as mpatches

    m = clip.map()
    for layer, st in _MAP_STYLE.items():
        items = getattr(m, layer, []) or []
        for el in items:
            g = _coords(el)
            if g is None or len(g) == 0:
                continue
            e = global_to_ego(g, ego_x, ego_y, ego_yaw)    # (N,2) [fwd,left]
            # 화면 밖(±rng 초과) 요소는 건너뛰어 그림을 깔끔히.
            if e[:, 0].min() > rng or e[:, 0].max() < -rng or e[:, 1].min() > rng or e[:, 1].max() < -rng:
                continue
            xs, ys = e[:, 1], e[:, 0]                       # 가로=left, 세로=forward
            if layer == "traffic_lights":                  # 신호등은 점
                ax.scatter(xs, ys, c=st["color"], s=18, marker="s", zorder=3)
            elif st["fill"]:                               # 폴리곤(채움)
                ax.add_patch(mpatches.Polygon(np.column_stack([xs, ys]), closed=True,
                                              fc=st["fill"], ec=st["color"], lw=st["lw"], zorder=1))
            else:                                          # 폴리라인
                ax.plot(xs, ys, color=st["color"], lw=st["lw"], zorder=2)


# 객체 카테고리별 색(legend도 이 키로 만든다).
_OBJ_COLOR = {"human": "#d62728", "vehicle": "#2ca02c", "other": "#9467bd"}


# draw_objects: 프레임의 **전체 3D 객체 annotation**(annotations.pkl의 objects)을 ego-frame BEV 박스로 그린다.
# ⚠️ 과거엔 Spatial per-camera 검출(camera_objects)을 썼는데 그건 **차량 위주**라 보행자 등이 빠졌다.
#    annotations는 human/vehicle/other 전부 보유하지만 pose가 **global UTM**이라 global_to_ego로 옮긴다.
#    각 객체: pose(x,y,yaw)+dimensions(l,w) → 중심 변환 + ego-frame heading(yaw-ego_yaw)으로 사각형.
# 표시 창(fwd_lo..fwd_hi 전방, ±left_hi 좌우) 밖(중심 기준) 객체는 생략.
def draw_objects(ax, frame, ego_x, ego_y, ego_yaw, fwd_lo, fwd_hi, left_hi):
    import matplotlib.patches as mpatches

    ann = frame.annotations()
    objs = getattr(ann, "objects", None) if ann is not None else None
    if not objs:
        return
    for o in objs:
        pose = getattr(o, "pose", None) or {}
        dims = getattr(o, "dimensions", None) or {}
        if not pose or not dims:
            continue
        cf, cl = global_to_ego(np.array([[pose["x"], pose["y"]]]), ego_x, ego_y, ego_yaw)[0]
        if not (fwd_lo <= cf <= fwd_hi and -left_hi <= cl <= left_hi):
            continue                                       # 표시 창 밖(중심) → 스킵
        yaw = float(pose.get("yaw", 0.0)) - ego_yaw        # ego-frame heading
        l, w = float(dims.get("l", 1.0)), float(dims.get("w", 1.0))
        # 로컬 박스 모서리(heading +x 따라 ±l/2, 좌우 ±w/2) → yaw 회전해 ego-frame [fwd,left]로.
        local = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
        c, s = np.cos(yaw), np.sin(yaw)
        fwd = cf + local[:, 0] * c - local[:, 1] * s
        left = cl + local[:, 0] * s + local[:, 1] * c
        cat = (getattr(o, "category", "") or "").split(".")[0]
        col = _OBJ_COLOR.get(cat, "#7f7f7f")
        ax.add_patch(mpatches.Polygon(np.column_stack([left, fwd]), closed=True,  # 가로=left, 세로=fwd
                                      fill=False, ec=col, lw=1.0, zorder=4, clip_on=True))


# draw_trajectory: 예측(pred_color, 기본 파랑)·GT(검정) 궤적 + ego 원점을 그린다(이미 ego-frame).
def draw_trajectory(ax, gt, pred, pred_color="b"):
    gt, pred = np.asarray(gt), np.asarray(pred)
    ax.plot(gt[:, 1], gt[:, 0], "k-o", ms=2, lw=1.4, label="GT", zorder=6)
    ax.plot(pred[:, 1], pred[:, 0], "-o", color=pred_color, ms=2, lw=1.4, label="pred", zorder=6)
    ax.scatter([0], [0], c="red", s=60, marker="^", zorder=7, label="ego")


# render_bev: 한 샘플의 BEV(맵+객체+궤적)를 ax에 통째로 그린다. 맵/객체 실패해도 궤적은 항상 그림.
#   fwd_lo..fwd_hi = 전방 표시 범위, lat_rng = 좌우 표시 반경. 8뷰 우측 패널은 좌우를 좁혀(lat_rng 작게)
#   Figure 5처럼 **세로로 긴** 패널이 된다. 격자(plot_grid)는 기본값(±60)으로 정사각형에 가깝게.
def render_bev(ax, rec, gt, pred, ade, fde, fwd_lo=-10.0, fwd_hi=60.0, lat_rng=60.0, pred_color="b"):
    import matplotlib.patches as mpatches

    cull = max(fwd_hi, lat_rng)                            # 맵 컬링은 더 넓은 쪽 기준
    try:
        clip, frame = scene_context(rec)
        if frame is not None:
            ex, ey, eyaw = _ego_pose(frame)
            draw_map(ax, clip, ex, ey, eyaw, cull)
            draw_objects(ax, frame, ex, ey, eyaw, fwd_lo, fwd_hi, lat_rng)
    except Exception as e:                                  # 맵/객체 없거나 매칭 실패 → 궤적만이라도
        ax.text(0.02, 0.98, f"(map/obj 생략: {type(e).__name__})", transform=ax.transAxes,
                fontsize=5, va="top", color="gray")
    draw_trajectory(ax, gt, pred, pred_color=pred_color)
    ax.set_title(f"{rec['id']}\nADE {ade:.2f}m  FDE {fde:.2f}m", fontsize=8)
    ax.set_xlim(lat_rng, -lat_rng); ax.set_ylim(fwd_lo, fwd_hi)  # 가로 반전(왼쪽이 +left), 세로 전방
    ax.set_aspect("equal")
    # 궤적(GT/pred/ego) + 객체 카테고리 색을 한 범례에 묶는다.
    obj_handles = [mpatches.Patch(ec=c, fc="none", label=k) for k, c in _OBJ_COLOR.items()]
    h, _ = ax.get_legend_handles_labels()
    ax.legend(handles=h + obj_handles, fontsize=6, loc="upper right")
