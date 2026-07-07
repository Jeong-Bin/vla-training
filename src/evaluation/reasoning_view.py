"""3-view reasoning 정지 이미지 렌더(노트북 view_reasoning.ipynb의 "Preview One Frame" 이식).

한 프레임을 노트북과 동일한 레이아웃으로 렌더한다: 위쪽 = DRIVING / COUNTERFACTUAL reasoning 텍스트(2열),
아래쪽 = front_left / front / front_right 3카메라(Spatial per-camera 객체 박스 오버레이 + ego 속도/가속도).
데이터는 원본 클립의 reasoning JSON(Driving/Counterfactual/Spatial) + annotations.pkl + ego_state.pkl에서 온다.

eval_qualitative의 SFT 샘플(8뷰 이미지 path 보유)에서 `frame_context_from_sample`로 원본 클립·프레임·reasoning을
되찾아 `render_3view`로 저장한다. 영상 렌더(process_clip 등)는 제외하고 정지 프레임 경로만 이식했다.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── 레이아웃 상수(노트북과 동일) ──────────────────────────────────────────────
FRAME_W = 2880
CAM_W = FRAME_W // 3
CAM_H = int(round(CAM_W * 1856 / 2816))
CAM_PAD_T = 16
CAM_PAD_B = 38
CAM_SEC_H = CAM_PAD_T + CAM_H + CAM_PAD_B

TEXT_H = 900
COL_W = FRAME_W // 2
FRAME_H = TEXT_H + CAM_SEC_H

BG_RGB = (18, 18, 18)
TEXT_RGB = (235, 235, 235)
HEADING_RGB = (100, 220, 255)
SUBHEAD_RGB = (255, 200, 70)
SAFE_RGB = (90, 255, 120)
UNSAFE_RGB = (255, 90, 90)
SUBOPT_RGB = (255, 195, 80)
DIM_RGB = (185, 185, 185)
COMP_RGB = (160, 220, 160)
DIV_RGB = (65, 65, 65)
VEL_BGR = (255, 220, 80)
ACCEL_BGR = (60, 220, 60)
DECEL_BGR = (60, 60, 255)
BG_BGR = (18, 18, 18)
DIV_BGR = (65, 65, 65)
CATEGORY_COLORS_BGR = {
    "car": (40, 220, 255), "vehicle.car": (40, 220, 255),
    "truck": (70, 255, 120), "vehicle.truck": (70, 255, 120),
    "bus": (255, 180, 60), "vehicle.bus": (255, 180, 60),
    "pedestrian": (255, 90, 180), "human.pedestrian": (255, 90, 180),
    "bicycle": (180, 130, 255), "vehicle.bicycle": (180, 130, 255),
    "motorcycle": (120, 200, 255), "vehicle.motorcycle": (120, 200, 255),
    "road obstacle": (70, 190, 255), "construction.traffic_cone": (70, 190, 255),
}
DEFAULT_BOX_COLOR_BGR = (235, 235, 235)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    # matplotlib 번들 DejaVu를 우선 사용(PIL이 bare 이름을 못 찾는 시스템 대비) → 실패 시 bare → default.
    try:
        from matplotlib import font_manager
        name = "DejaVu Sans" + (":bold" if bold else "")
        return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans",
                                  weight="bold" if bold else "normal")), size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


fT = _font(44, bold=True)
fS = _font(32, bold=True)
fB = _font(26)
fSm = _font(22)


def _lh(font) -> int:
    bb = font.getbbox("Ag")
    return (bb[3] - bb[1]) + 5


def _wrap(text: str, font, max_w: int, draw: "ImageDraw.ImageDraw") -> list:
    words = str(text).split()
    lines, cur = [], []
    for w in words:
        probe = " ".join(cur + [w])
        if draw.textlength(probe, font=font) <= max_w:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines or [""]


# ── pickle/JSON 로더(노트북과 동일: 커스텀 클래스를 dict로 안전 언피클) ──────────────
class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return type(name, (), {
            "__repr__": lambda self: f"<{module}.{name}>",
            "__init__": lambda self, **kw: self.__dict__.update(kw),
        })


def _obj_to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {k: _obj_to_dict(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_obj_to_dict(v) for v in obj]
    return obj


def load_ego_state(pkl_path: str) -> dict:
    try:
        with open(pkl_path, "rb") as f:
            return _obj_to_dict(_SafeUnpickler(f).load())
    except Exception:
        return {}


def load_annotation(clip_dir: str, frame: dict) -> dict:
    ann_rel = frame.get("annotations", "")
    if not ann_rel:
        return {}
    ann_path = os.path.join(clip_dir, ann_rel)
    if not os.path.exists(ann_path):
        return {}
    try:
        with open(ann_path, "rb") as f:
            return _obj_to_dict(_SafeUnpickler(f).load())
    except Exception:
        return {}


def annotation_by_track_token(annotation: dict) -> dict:
    return {o.get("track_token"): o for o in annotation.get("objects", [])
            if isinstance(o, dict) and o.get("track_token")}


def load_reasoning(clip_dir: str, frame: dict) -> dict:
    rel = frame.get("reasoning", "")
    if not rel:
        return {}
    p = os.path.join(clip_dir, rel)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def has_complete_reasoning(rjson: dict) -> bool:
    driving = rjson.get("Driving", "")
    counterfactual = rjson.get("Counterfactual", "")
    if not isinstance(driving, dict) or not driving.get("Driving decision"):
        return False
    if not isinstance(counterfactual, dict) or not counterfactual:
        return False
    return True


def frame_reasoning_sections(rjson: dict):
    """(driving, counterfactual, per_camera, complete). 미완이면 driving/cf/per_camera는 {} 반환."""
    if not has_complete_reasoning(rjson):
        return {}, {}, {}, False
    return (rjson.get("Driving", {}), rjson.get("Counterfactual", {}),
            rjson.get("Spatial", {}).get("per_camera_results", {}), True)


# ── 상단 텍스트(DRIVING / COUNTERFACTUAL 2열) ─────────────────────────────────
def _render_column(img, x0: int, section_w: int, title: str, content_fn) -> None:
    draw = ImageDraw.Draw(img)
    M = 22
    max_w = section_w - 2 * M
    x = x0 + M
    y_ref = [16]

    def put(text: str, font=fB, color=TEXT_RGB, indent: int = 0, sp: int = 3):
        for line in _wrap(text, font, max_w - indent, draw):
            if y_ref[0] + _lh(font) + 4 > TEXT_H - 4:
                return
            draw.text((x + indent, y_ref[0]), line, font=font, fill=color)
            y_ref[0] += _lh(font) + sp

    def gap(px: int = 8):
        y_ref[0] += px

    def sep():
        yy = y_ref[0] + 3
        draw.line([(x, yy), (x0 + section_w - M, yy)], fill=DIV_RGB, width=1)
        y_ref[0] += 12

    put(title, font=fT, color=HEADING_RGB)
    sep()
    content_fn(draw, put, gap, sep)


def _driving_content(driving: dict):
    def fn(draw, put, gap, sep):
        scene = driving.get("Scene description", "")
        if scene:
            put("Scene", font=fS, color=SUBHEAD_RGB)
            put(scene, indent=14)
            gap()
        comps = driving.get("Critical components", {})
        if comps:
            put("Critical Components", font=fS, color=SUBHEAD_RGB)
            for name, info in comps.items():
                ctype = info.get("Type", "")
                put(f"* {name}" + (f"  [{ctype}]" if ctype else ""), color=COMP_RGB, indent=12)
                for k, v in info.items():
                    if k == "Type":
                        continue
                    put(f"{k}: {v}", font=fSm, color=DIM_RGB, indent=26)
            gap()
        decision = driving.get("Driving decision", {})
        if decision:
            put("Decision", font=fS, color=SUBHEAD_RGB)
            put(f"Longitudinal:  {decision.get('Longitudinal', '')}", color=SAFE_RGB, indent=14)
            put(f"Lateral:       {decision.get('Lateral', '')}", color=SAFE_RGB, indent=14)
            gap()
        trace = driving.get("Reasoning trace", "")
        if trace:
            put("Reasoning Trace", font=fS, color=SUBHEAD_RGB)
            put(trace, indent=14)
    return fn


def _counterfactual_content(cf: dict):
    def fn(draw, put, gap, sep):
        alt = cf.get("Alternative actions", [])
        if alt:
            put("Alternative Actions", font=fS, color=SUBHEAD_RGB)
            for action in alt:
                risk = action.get("Risk level", "")
                rl = risk.lower()
                color = SAFE_RGB if "safe" in rl and "unsafe" not in rl else (
                    SUBOPT_RGB if "suboptimal" in rl else UNSAFE_RGB)
                put(f"* {action.get('Longitudinal','')} / {action.get('Lateral','')}   [{risk}]", color=color, indent=12)
                if action.get("Reason", ""):
                    put(action["Reason"], font=fSm, color=DIM_RGB, indent=28)
            gap()
        top = cf.get("Top safety-critical actions", [])
        if top:
            put("Safety-Critical Actions", font=fS, color=SUBHEAD_RGB)
            for action in top:
                risk = action.get("Risk level", "")
                rl = risk.lower()
                color = UNSAFE_RGB if "unsafe" in rl else (SUBOPT_RGB if "suboptimal" in rl else SAFE_RGB)
                put(f"* {action.get('Longitudinal','')} / {action.get('Lateral','')}   [{risk}]", color=color, indent=12)
                if action.get("Reason", ""):
                    put(action["Reason"], font=fSm, color=DIM_RGB, indent=28)
            gap()
    return fn


def render_top_section(driving: dict | None, counterfactual: dict | None, status: str = "") -> np.ndarray:
    img = Image.new("RGB", (FRAME_W, TEXT_H), BG_RGB)
    draw = ImageDraw.Draw(img)
    draw.line([(COL_W, 0), (COL_W, TEXT_H)], fill=DIV_RGB, width=2)
    if status:
        banner_color = SAFE_RGB if status.startswith("NEW") else DIM_RGB
        tw = draw.textlength(status, font=fSm)
        draw.rounded_rectangle((FRAME_W - tw - 54, 16, FRAME_W - 22, 54), radius=8,
                               fill=(35, 35, 35), outline=banner_color, width=2)
        draw.text((FRAME_W - tw - 38, 23), status, font=fSm, fill=banner_color)
    if driving or counterfactual:
        _render_column(img, 0, COL_W, "DRIVING", _driving_content(driving or {}))
        _render_column(img, COL_W, COL_W, "COUNTERFACTUAL", _counterfactual_content(counterfactual or {}))
    else:
        message = "Digesting data for reasoning..."
        for x0, title in ((0, "DRIVING"), (COL_W, "COUNTERFACTUAL")):
            M = 22
            draw.text((x0 + M, 16), title, font=fT, fill=HEADING_RGB)
            sep_y = 16 + _lh(fT) + 6
            draw.line([(x0 + M, sep_y), (x0 + COL_W - M, sep_y)], fill=DIV_RGB, width=1)
            tw = draw.textlength(message, font=fS)
            th = fS.getbbox(message)[3] - fS.getbbox(message)[1]
            draw.text((x0 + (COL_W - tw) / 2, (TEXT_H - th) / 2), message, font=fS, fill=DIM_RGB)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── GT vs Pred 2열 상단(DRIVING+COUNTERFACTUAL 세로 스택) ─────────────────────
REASONING_ORDER = ("spatial", "decision", "counterfactual")
_PRED_LABEL = {"spatial": "Spatial", "decision": "Decision (Reasoning Trace)", "counterfactual": "Counterfactual"}


def parse_pred_reasoning(text: str) -> dict:
    """모델이 생성한 combined reasoning 텍스트를 {spatial,decision,counterfactual}로 파싱.
    학습 타깃 포맷이 'Spatial: …\\nDecision: …\\nCounterfactual: …'이므로 그 마커로 구간 분할.
    마커가 없으면 통째로 decision에 담는다(하위호환)."""
    import re
    out: dict = {}
    if not text:
        return out
    hits = []
    for key in REASONING_ORDER:
        m = re.search(rf"(?im)^\s*{key}\s*:", text)
        if m:
            hits.append((m.start(), m.end(), key))
    hits.sort()
    for i, (s, e, key) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        seg = text[e:end].strip()
        if seg:
            out[key] = seg
    if not hits and text.strip():
        out["decision"] = text.strip()
    return out


def _gt_column_content(driving: dict, counterfactual: dict):
    """왼쪽 GT 열: DRIVING 섹션들 + (아래에) COUNTERFACTUAL 섹션을 한 열에 세로로 쌓는다."""
    dfn = _driving_content(driving or {})
    cfn = _counterfactual_content(counterfactual or {})

    def fn(draw, put, gap, sep):
        if driving:
            dfn(draw, put, gap, sep)
        if counterfactual:
            gap()
            put("COUNTERFACTUAL", font=fS, color=HEADING_RGB)   # Reasoning Trace 아래로 이동된 CF 시작
            cfn(draw, put, gap, sep)
        if not driving and not counterfactual:
            put("(no GT reasoning)", font=fS, color=DIM_RGB, indent=8)
    return fn


def _pred_column_content(pred_parts: dict, reasoning_types) -> "callable":
    """오른쪽 PRED 열: 학습한 reasoning 타입(reasoning_types)만 소제목+생성텍스트로 표시.
    baseline(빈 reasoning_types)이면 안내만. 학습했는데 파싱 결과가 없으면 '(none)'."""
    types = [t for t in REASONING_ORDER if t in (reasoning_types or [])]

    def fn(draw, put, gap, sep):
        if not types:                                       # DejaVu 폰트에 한글 글리프 없음 → 영어로
            put("(baseline: no reasoning trained)", font=fS, color=DIM_RGB, indent=8)
            return
        for t in types:
            put(_PRED_LABEL[t], font=fS, color=SUBHEAD_RGB)
            put((pred_parts or {}).get(t) or "(none)", indent=14)
            gap()
    return fn


def render_top_section_gtpred(driving, counterfactual, pred_parts, reasoning_types,
                              status: str = "REASONING FRAME") -> np.ndarray:
    """상단 텍스트: 왼쪽 열=GT(DRIVING+COUNTERFACTUAL), 오른쪽 열=Pred(학습 타입별 생성 reasoning)."""
    img = Image.new("RGB", (FRAME_W, TEXT_H), BG_RGB)
    draw = ImageDraw.Draw(img)
    draw.line([(COL_W, 0), (COL_W, TEXT_H)], fill=DIV_RGB, width=2)
    if status:
        tw = draw.textlength(status, font=fSm)
        draw.rounded_rectangle((FRAME_W - tw - 54, 16, FRAME_W - 22, 54), radius=8,
                               fill=(35, 35, 35), outline=DIM_RGB, width=2)
        draw.text((FRAME_W - tw - 38, 23), status, font=fSm, fill=DIM_RGB)
    _render_column(img, 0, COL_W, "GT", _gt_column_content(driving, counterfactual))
    _render_column(img, COL_W, COL_W, "PRED", _pred_column_content(pred_parts, reasoning_types))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── 카메라 + 객체 박스 ────────────────────────────────────────────────────────
def load_cam(base_dir: str, rel_path: str) -> np.ndarray:
    if not rel_path:
        return np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    img = cv2.imread(os.path.join(base_dir, rel_path))
    if img is None:
        return np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    return cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)


def _category_key(obj: dict) -> str:
    return str(obj.get("detection_label") or obj.get("category") or "object").lower()


def _box_color(obj: dict):
    category = _category_key(obj)
    if category in CATEGORY_COLORS_BGR:
        return CATEGORY_COLORS_BGR[category]
    for key, color in CATEGORY_COLORS_BGR.items():
        if key in category:
            return color
    return DEFAULT_BOX_COLOR_BGR


def _annotation_lines(obj: dict, annotation_lookup: dict) -> list:
    ann = annotation_lookup.get(obj.get("track_token"), {})
    velocity = ann.get("velocity", {})
    label = str(obj.get("detection_label") or obj.get("category") or "object")[:28]
    lines = [label]
    if obj.get("detection_bbox_3d", {}).get("center_3d_ego"):
        center = obj["detection_bbox_3d"]["center_3d_ego"]
        lines.append(f"ego x={center.get('x', 0.0):.1f}, y={center.get('y', 0.0):.1f}")
    if velocity:
        lines.append(f"vx={velocity.get('vx', 0.0):.2f}, vy={velocity.get('vy', 0.0):.2f}")
    return lines


def _rects_overlap(a, b) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_label_position(x1, y1, x2, label_w, label_h, occupied):
    margin = 8
    right_x = x2 + margin
    left_x = x1 - label_w - margin
    x_options = []
    if right_x + label_w < CAM_W:
        x_options.append(right_x)
    if left_x >= 0:
        x_options.append(left_x)
    if not x_options:
        x_options.append(max(0, min(CAM_W - label_w - 1, right_x)))
    offsets = [0]
    for step in range(1, 10):
        delta = step * (label_h + 8)
        offsets.extend([-delta, delta])
    for lx in x_options:
        for offset in offsets:
            ly = max(0, min(CAM_H - label_h - 1, y1 + offset))
            rect = (lx, ly, lx + label_w, ly + label_h)
            if not any(_rects_overlap(rect, prev) for prev in occupied):
                return rect
    lx = x_options[0]
    ly = max(0, min(CAM_H - label_h - 1, y1))
    return (lx, ly, lx + label_w, ly + label_h)


def _draw_box_label(tile, x1, y1, x2, y2, lines, color, occupied) -> None:
    if not lines:
        return
    lf = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, row_h, pad = 0.52, 1, 22, 4
    text_sizes = [cv2.getTextSize(line, lf, scale, thick)[0] for line in lines]
    label_w = max(w for w, _ in text_sizes) + pad
    label_h = len(lines) * row_h
    lx, ly, rx, by = _choose_label_position(x1, y1, x2, label_w, label_h, occupied)
    occupied.append((lx, ly, rx, by))
    for i, line in enumerate(lines):
        yy = ly + 16 + i * row_h
        cv2.putText(tile, line, (lx + pad + 1, yy + 1), lf, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(tile, line, (lx + pad, yy), lf, scale, color, thick, cv2.LINE_AA)


def _draw_object_boxes(tile, camera_reasoning: dict, annotation_lookup: dict) -> None:
    occupied_labels = []
    for obj in camera_reasoning.get("objects", []):
        bbox = obj.get("detection_bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        sx, sy = CAM_W / 2816.0, CAM_H / 1856.0
        x1, x2 = int(round(x1 * sx)), int(round(x2 * sx))
        y1, y2 = int(round(y1 * sy)), int(round(y2 * sy))
        x1, x2 = max(0, min(CAM_W - 1, x1)), max(0, min(CAM_W - 1, x2))
        y1, y2 = max(0, min(CAM_H - 1, y1)), max(0, min(CAM_H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        color = _box_color(obj)
        cv2.rectangle(tile, (x1, y1), (x2, y2), color, 3)
        _draw_box_label(tile, x1, y1, x2, y2, _annotation_lines(obj, annotation_lookup), color, occupied_labels)


def _overlay_ego(canvas, cam_x, cam_y, cam_w, cam_h, ego: dict) -> None:
    vel = ego.get("velocity", {})
    acc = ego.get("acceleration", {})
    lf = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick, lh, pad = 1.1, 2, 42, 18

    def _put(text, x, y, color):
        cv2.putText(canvas, text, (x + 2, y + 2), lf, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), lf, scale, color, thick, cv2.LINE_AA)

    if vel:
        vx, vy = vel.get("vx", 0.0), vel.get("vy", 0.0)
        speed_ms = (vx ** 2 + vy ** 2) ** 0.5
        x0 = cam_x + pad
        y0 = cam_y + pad + int(scale * 28)
        _put("Velocity:", x0, y0, VEL_BGR)
        _put(f"{speed_ms:.2f} m/s", x0, y0 + lh, VEL_BGR)
        _put(f"{speed_ms * 3.6:.1f} km/h", x0, y0 + 2 * lh, VEL_BGR)
    if acc:
        ax = acc.get("ax", 0.0)
        acc_color = ACCEL_BGR if ax >= 0 else DECEL_BGR
        lines = ["Accelerating" if ax >= 0 else "Decelerating", f"{ax:+.2f} m/s^2"]
        max_tw = max(cv2.getTextSize(line, lf, scale, thick)[0][0] for line in lines)
        x0 = cam_x + cam_w - pad - max_tw
        y0 = cam_y + pad + int(scale * 28)
        for i, line in enumerate(lines):
            tw = cv2.getTextSize(line, lf, scale, thick)[0][0]
            _put(line, x0 + (max_tw - tw), y0 + i * lh, acc_color)


def compose_frame(left, front, right, top_section, frame_num, total_frames, ego_state=None) -> np.ndarray:
    canvas = np.full((FRAME_H, FRAME_W, 3), BG_BGR, dtype=np.uint8)
    canvas[0:TEXT_H, 0:FRAME_W] = top_section
    cv2.line(canvas, (0, TEXT_H), (FRAME_W, TEXT_H), DIV_BGR, 3)
    cam_y = TEXT_H + CAM_PAD_T
    canvas[cam_y:cam_y + CAM_H, 0:CAM_W] = left
    canvas[cam_y:cam_y + CAM_H, CAM_W:2 * CAM_W] = front
    canvas[cam_y:cam_y + CAM_H, 2 * CAM_W:3 * CAM_W] = right
    if ego_state:
        _overlay_ego(canvas, CAM_W, cam_y, CAM_W, CAM_H, ego_state)
    for x_div in (CAM_W, 2 * CAM_W):
        cv2.line(canvas, (x_div, cam_y), (x_div, cam_y + CAM_H), DIV_BGR, 2)
    label_y = cam_y + CAM_H + 22
    lf = cv2.FONT_HERSHEY_SIMPLEX
    for label, cx in [("LEFT", CAM_W // 2), ("FRONT", CAM_W + CAM_W // 2), ("RIGHT", 2 * CAM_W + CAM_W // 2)]:
        tw = cv2.getTextSize(label, lf, 0.9, 2)[0][0]
        cv2.putText(canvas, label, (cx - tw // 2, label_y), lf, 0.9, (170, 170, 170), 2, cv2.LINE_AA)
    fc_text = f"{frame_num + 1} / {total_frames}"
    fc_tw = cv2.getTextSize(fc_text, lf, 0.7, 1)[0][0]
    cv2.putText(canvas, fc_text, (FRAME_W - fc_tw - 12, cam_y + 28), lf, 0.7, (120, 120, 120), 1, cv2.LINE_AA)
    return canvas


# ── SFT 샘플 → 원본 클립/프레임/reasoning 복구 + 3-view 저장 ────────────────────
def frame_context_from_sample(sample: dict):
    """SFT 샘플(8뷰 이미지 path 보유) → (clip_dir(str), frame_dict, rjson, total_frames).
    front 카메라 파일 basename으로 metadata 프레임을 정확 매칭. 실패 시 (clip_dir|None, None, {}, 0)."""
    imgs = sample.get("images", [])
    front = next((im["path"] for im in imgs if im.get("view") == "front"), None) or (imgs[0]["path"] if imgs else None)
    if not front:
        return None, None, {}, 0
    parts = Path(front).parts
    if "clips" not in parts:
        return None, None, {}, 0
    ci = parts.index("clips")
    clip_dir = str(Path(*parts[: ci + 2]))
    meta_path = os.path.join(clip_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return clip_dir, None, {}, 0
    with open(meta_path) as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    target = Path(front).name
    # ⚠️ 인접 두 프레임(예: index 99·100)이 **같은 front 파일**을 공유하고 그중 하나만 reasoning을 보유하는
    #    경우가 있다(원본 데이터 특성; bev.scene_context와 동일 이슈). basename이 일치하는 프레임들 중
    #    **reasoning 있는 프레임을 우선** 매칭해야 build_sft가 쓴 프레임과 일치한다(아니면 "(no GT reasoning)").
    matches = [fr for fr in frames
               if (fr.get("sensors", {}) or {}).get("cameras", {}).get("front", "")
               and Path((fr["sensors"]["cameras"]["front"])).name == target]
    frame = next((fr for fr in matches if fr.get("reasoning")), matches[0] if matches else None)
    if frame is None:
        return clip_dir, None, {}, len(frames)
    return clip_dir, frame, load_reasoning(clip_dir, frame), len(frames)


def render_3view(clip_dir: str, frame: dict, rjson: dict, total_frames: int, out_path,
                 pred_parts: dict | None = None, reasoning_types=()) -> bool:
    """한 프레임을 정지 이미지로 저장. 상단=GT(왼쪽)|Pred(오른쪽) 2열 reasoning, 하단=3카메라+GT 박스.
    pred_parts=파싱된 생성 reasoning({spatial,decision,counterfactual}), reasoning_types=학습한 종류.
    frame 없으면 False."""
    if frame is None:
        return False
    # GT reasoning: 완결 여부와 무관하게 있는 대로 표시(부분 프레임도 렌더). 박스용 per_camera도 직접 추출.
    driving = rjson.get("Driving", {}) if isinstance(rjson.get("Driving"), dict) else {}
    counterfactual = rjson.get("Counterfactual", {}) if isinstance(rjson.get("Counterfactual"), dict) else {}
    per_camera = rjson.get("Spatial", {}).get("per_camera_results", {}) if isinstance(rjson.get("Spatial"), dict) else {}
    top = render_top_section_gtpred(driving, counterfactual, pred_parts, reasoning_types)
    annotation_lookup = annotation_by_track_token(load_annotation(clip_dir, frame))
    cams = frame.get("sensors", {}).get("cameras", {})
    left_img = load_cam(clip_dir, cams.get("front_left", ""))
    front_img = load_cam(clip_dir, cams.get("front", ""))
    right_img = load_cam(clip_dir, cams.get("front_right", ""))
    _draw_object_boxes(left_img, per_camera.get("front_left", {}), annotation_lookup)
    _draw_object_boxes(front_img, per_camera.get("front", {}), annotation_lookup)
    _draw_object_boxes(right_img, per_camera.get("front_right", {}), annotation_lookup)
    ego_rel = frame.get("ego_state", "")
    ego_state = load_ego_state(os.path.join(clip_dir, ego_rel)) if ego_rel else {}
    idx = int(frame.get("frame_index", 0))
    composed = compose_frame(left_img, front_img, right_img, top, idx, total_frames or (idx + 1), ego_state)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), composed)
    return True


def reasoning_trace_of(rjson: dict) -> str:
    """Driving.Reasoning trace 텍스트(없으면 "")."""
    drv = rjson.get("Driving", {}) if isinstance(rjson, dict) else {}
    return drv.get("Reasoning trace", "") if isinstance(drv, dict) else ""
