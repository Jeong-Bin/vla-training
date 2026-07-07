"""Visualise the 8 surround-view cameras of a driving keyframe (multi-view sanity check).

Renders the ego vehicle's 8 cameras in a geometric 3×3 surround layout with the GT
driving decision + reasoning in the centre cell. This is the visual counterpart to the
8-view SFT pipeline: it confirms every view is on disk and lets a human read the same
scene the model is trained on.

Layout (matches how the cameras sit around the car):
    front_left   front    front_right
    left        [ GT ]    right
    back_left    back     back_right

Run:  python scripts/view_8cam.py [--clip <dirname|path>] [--kf 0] [--out PATH]
      (no --clip → first local clip that has all 8 views extracted)
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nureasoning import CAMERA_VIEWS, VIEW_LABELS, parse_clip, select_keyframes  # noqa: E402

CLIPS_DIR = Path("/home/etri/DATASET/nureasoning/clips")   # 대용량 10Hz 클립 루트(REPO 밖)
OUT_DIR = REPO / "results" / "viz"

# (row, col) in the 3×3 grid for each view; (1,1) centre is reserved for the GT text.
GRID_POS = {
    "front_left": (0, 0), "front": (0, 1), "front_right": (0, 2),
    "left": (1, 0),                          "right": (1, 2),
    "back_left": (2, 0), "back": (2, 1), "back_right": (2, 2),
}


def first_full_clip() -> Path:
    """First local clip whose driving keyframes have all 8 views on disk."""
    for d in sorted(CLIPS_DIR.glob("*/metadata.json")):
        clip = parse_clip(d.parent)
        kfs = select_keyframes(clip, policy="driving")
        if kfs and all(kfs[0].has_camera(v) for v in CAMERA_VIEWS):
            return d.parent
    raise SystemExit("no local clip has all 8 views yet — run download_clips.py --refetch-local --views all")


def resolve_clip(arg: str | None) -> Path:
    if not arg:
        return first_full_clip()
    p = Path(arg)
    if p.is_dir():
        return p
    cand = CLIPS_DIR / arg
    if cand.is_dir():
        return cand
    raise SystemExit(f"clip not found: {arg}")


def render(clip_dir: Path, kf_idx: int, out_path: Path) -> None:
    clip = parse_clip(clip_dir)
    kfs = select_keyframes(clip, policy="driving")
    if not kfs:
        raise SystemExit(f"no driving keyframes in {clip_dir.name}")
    frame = kfs[max(0, min(kf_idx, len(kfs) - 1))]

    dec = frame.driving_decision() or {}
    reasoning = frame.reasoning_trace() or ""
    gt_text = (
        f"GT decision\n"
        f"Longitudinal: {dec.get('Longitudinal', '—')}\n"
        f"Lateral: {dec.get('Lateral', '—')}\n\n"
        + "\n".join(textwrap.wrap(reasoning, 38))
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 11))
    for v in CAMERA_VIEWS:
        r, c = GRID_POS[v]
        ax = axes[r][c]
        ax.set_title(VIEW_LABELS[v], fontsize=11)
        ax.axis("off")
        if frame.has_camera(v):
            ax.imshow(frame.image(v))
        else:
            ax.text(0.5, 0.5, f"{VIEW_LABELS[v]}\n(missing)", ha="center", va="center")

    centre = axes[1][1]
    centre.axis("off")
    centre.text(0.5, 0.5, gt_text, ha="center", va="center", fontsize=11,
                family="monospace", wrap=True,
                bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#999"))

    fig.suptitle(f"{clip.clip_token}  ·  frame {frame.frame_index}  ·  mission: {frame.mission_command}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}  (frame {frame.frame_index}, "
          f"{sum(frame.has_camera(v) for v in CAMERA_VIEWS)}/8 views)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None, help="clip dirname or path (default: first full 8-view clip)")
    ap.add_argument("--kf", type=int, default=0, help="which driving keyframe (0-based)")
    ap.add_argument("--out", default=None, help="output PNG path")
    args = ap.parse_args()

    clip_dir = resolve_clip(args.clip)
    out = Path(args.out) if args.out else OUT_DIR / f"{clip_dir.name}_kf{args.kf}_8view.png"
    render(clip_dir, args.kf, out)


if __name__ == "__main__":
    main()
