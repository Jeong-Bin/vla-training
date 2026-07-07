"""Eyeball-verify the common loader against a real extracted clip (spec §6).

Parses a clip, selects the action-labelled keyframes, prints the fields each stage
will consume, and renders the first keyframe's front camera with a caption overlay
(driving decision + reasoning) to results/ so the parse can be checked by eye.

Run:  python scripts/check_loader.py [CLIP_OR_ROOT]
Default CLIP_OR_ROOT = the Milestone-0 probe extract under data/raw/_probe_extract.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nureasoning import NuReasoningDataset, find_clip_dirs, parse_clip, select_keyframes

DEFAULT_ROOT = REPO / "data" / "raw" / "_probe_extract"
RESULTS = REPO / "results"


def _speed(ego) -> float:
    v = getattr(ego, "velocity", {}) or {}
    return (v.get("vx", 0.0) ** 2 + v.get("vy", 0.0) ** 2) ** 0.5


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROOT
    clip_dirs = find_clip_dirs(root)
    if not clip_dirs:
        raise SystemExit(f"no clip (metadata.json) found under {root}")

    clip = parse_clip(clip_dirs[0])
    print(f"== clip ==\n{clip}\n  dir: {clip.clip_dir}")
    print(f"  total_frames(meta)={clip.total_frames}  parsed_frames={len(clip)}  "
          f"frame_rate_hz={clip.frame_rate_hz}")

    keyframes = select_keyframes(clip, policy="driving")
    print(f"\n== driving keyframes: {len(keyframes)} "
          f"(frame_index={[f.frame_index for f in keyframes]}) ==")

    for f in keyframes:
        dec = f.driving_decision()
        ego = f.ego_state()
        ann = f.annotations()
        cf = f.counterfactual() or {}
        n_obj = len(getattr(ann, "objects", []) or [])
        n_tl = len(getattr(ann, "traffic_light_states", []) or [])
        print(f"\n  [{f.sample_id}] frame_index={f.frame_index} t={f.relative_time_s:.1f}s")
        print(f"    mission        : {f.mission_command}")
        print(f"    front camera   : exists={f.has_camera('front')}  {f.camera_relpaths.get('front')}")
        print(f"    ego speed      : {_speed(ego):.2f} m/s   "
              f"ax={(getattr(ego,'acceleration',{}) or {}).get('ax', float('nan')):.2f}")
        print(f"    objects/lights : {n_obj} objects, {n_tl} traffic-light states")
        print(f"    decision       : long={dec['Longitudinal']!r}  lat={dec['Lateral']!r}")
        print(f"    reasoning      : {(f.reasoning_trace() or '')[:120]}")
        print(f"    counterfactual : "
              f"{len(cf.get('Alternative actions', []))} alt, "
              f"{len(cf.get('Top safety-critical actions', []))} safety-critical")

    # Dataset wrapper smoke test (default keyframe policy = driving).
    ds = NuReasoningDataset(clip)
    print(f"\n== Dataset ==\n  {ds}")
    s0 = ds[0]
    print(f"  sample[0] keys={list(s0)}  id={s0['id']}")

    # Visual verification: render the first keyframe's front view with the GT
    # object boxes (Spatial.per_camera_results[front], pixel coords) drawn on top,
    # plus the GT driving decision. This lets the loaded labels be checked against
    # the actual scene (boxes should land on the real vehicles / cones).
    kf = keyframes[0]
    if kf.has_camera("front"):
        _render_front_with_gt(kf, RESULTS / "loader_sample.jpg")
    else:
        print("\n  (front camera not present in this extract; skipped visual render)")

    print("\nOK: loader parsed the clip and produced loader-ready fields.")


def _box_color(category: str) -> tuple[int, int, int]:
    c = (category or "").lower()
    if c.startswith("vehicle"):
        return (90, 170, 255)      # blue
    if "pedestrian" in c or "human" in c:
        return (255, 90, 90)       # red
    if "cycle" in c or "bicycle" in c:
        return (255, 180, 40)      # orange
    return (120, 230, 120)         # green (cones/barriers/other)


def _render_front_with_gt(frame, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageOps  # noqa: F401

    img = ImageOps.autocontrast(frame.image("front"))  # raw frames are dark; lift for viz
    scale = 1280 / img.width
    canvas = img.resize((1280, int(img.height * scale)))
    draw = ImageDraw.Draw(canvas)

    objs = frame.camera_objects("front")
    for o in objs:
        bb = o.get("detection_bbox_2d")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x1, y1, x2, y2 = (v * scale for v in bb)
        color = _box_color(o.get("category", ""))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = o.get("detection_label") or o.get("category", "")
        ty = max(0, y1 - 14)
        draw.rectangle([x1, ty, x1 + 8 * len(label) + 6, ty + 14], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill=(0, 0, 0))

    dec = frame.driving_decision()
    header = [
        f"GROUND TRUTH  {frame.sample_id}  (reasoning/{frame.timestamp_us}.json)",
        f"mission={frame.mission_command}   2D boxes drawn: {len(objs)}",
        f"LONG: {dec['Longitudinal']}    LAT: {dec['Lateral']}",
        f"why: {(frame.reasoning_trace() or '')[:110]}",
    ]
    draw.rectangle([0, 0, canvas.width, 80], fill=(0, 0, 0))
    draw.multiline_text((8, 6), "\n".join(header), fill=(255, 255, 255), spacing=3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    print(f"\n  visual check saved -> {out_path.relative_to(REPO)} "
          f"({len(objs)} GT boxes on front camera)")


if __name__ == "__main__":
    main()
