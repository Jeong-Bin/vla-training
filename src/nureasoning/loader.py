"""nuReasoning common loader/parser — the shared spine for all pipeline stages.

Two layers are kept deliberately separate (spec §6):

  (a) Parser   — a clip directory becomes lightweight ``Frame`` objects that hold
                 only paths and small scalar fields. Raw assets (images, ego/annot
                 pickles, reasoning JSON) are loaded *lazily*, on first access, and
                 cached per frame. Nothing heavy is pulled into memory at parse time.

  (b) Dataset  — ``NuReasoningDataset`` is a ``torch.utils.data.Dataset`` over a set
                 of selected keyframes. The per-sample shape of ``__getitem__`` is
                 delegated to a ``sample_fn`` callback so each stage can produce its
                 own output (SFT pair, eval record, retrieval features)
                 without subclassing.

Keyframe scoping (DECISIONS.md, 2026-06-16): reasoning is sparse — only the frames
whose ``Driving.Driving decision`` is populated carry an action label (≈3 per clip).
Those are the default keyframes (``policy="driving"``); ``"uniform"`` / ``"all"``
are provided for generic use.

Important: the ego_state/annotations ``.pkl`` files are pickled instances of the
dataclasses in the repo-root ``data_schema.py``. That module must therefore be
importable for unpickling to succeed; this module ensures it is (see below).
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Union

# ---------------------------------------------------------------------------
# Make the repo-root `data_schema` module importable. The pickled EgoState /
# Annotations objects reference the *top-level* module name ``data_schema``, so
# it must resolve regardless of where this package is imported from.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import data_schema  # noqa: E402  (required so pickle can resolve EgoState/Annotations)

# torch is optional at import time so the parser stays usable for non-torch
# consumers (e.g. retrieval EDA). The Dataset only makes sense with torch present.
try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # noqa: BLE001
    _TorchDataset = object  # type: ignore[assignment, misc]

# 8 logical camera views — keys of metadata ``frames[i].sensors.cameras``.
# Order is canonical: front first, then the surround views clockwise-ish. Multi-view
# consumers (SFT chat format / eval prompt) interleave images in this order so the
# model always sees the same view sequence.
CAMERA_VIEWS: tuple[str, ...] = (
    "front", "front_left", "front_right", "left",
    "right", "back", "back_left", "back_right",
)

# Human-readable label per view, used to caption each image in multi-view prompts so
# the model can tell which camera is which (Qwen-VL interleaves text and images).
VIEW_LABELS: dict[str, str] = {
    "front": "Front", "front_left": "Front-left", "front_right": "Front-right",
    "left": "Left", "right": "Right", "back": "Back",
    "back_left": "Back-left", "back_right": "Back-right",
}

PathLike = Union[str, Path]


# ===========================================================================
# (a) Parser — Frame / Clip
# ===========================================================================
@dataclass
class Frame:
    """A single frame: paths + small fields now, raw assets loaded lazily.

    Relative paths (``camera_relpaths``, ``ego_state_relpath`` …) come straight
    from ``metadata.json`` and are resolved against ``clip_dir`` on demand. Camera
    file names carry their *own* timestamp (≠ frame ``timestamp_us``), so we always
    use the metadata-provided path rather than reconstructing it.
    """

    clip_dir: Path
    clip_token: str
    index: int                       # position in the metadata frames array
    frame_index: int                 # metadata-reported frame_index (usually == index)
    token: str
    timestamp_us: int
    relative_time_s: float

    camera_relpaths: dict[str, str]  # logical view -> relative path (present views only)
    ego_state_relpath: str
    annotations_relpath: str
    reasoning_relpath: str
    mission_goal: Optional[dict] = None

    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- identity -----------------------------------------------------------
    @property
    def sample_id(self) -> str:
        """Stable id used across pipeline stages: ``<clip_token>_<frame_index>``."""
        return f"{self.clip_token}_{self.frame_index}"

    @property
    def mission_command(self) -> Optional[str]:
        return (self.mission_goal or {}).get("command")

    # -- cameras ------------------------------------------------------------
    def camera_path(self, view: str = "front") -> Optional[Path]:
        """Absolute path to a camera image, or ``None`` if the view is absent."""
        rel = self.camera_relpaths.get(view)
        return (self.clip_dir / rel) if rel else None

    def has_camera(self, view: str = "front") -> bool:
        p = self.camera_path(view)
        return p is not None and p.exists()

    def image(self, view: str = "front"):
        """Lazily open and cache a camera image as a PIL ``Image`` (RGB)."""
        key = f"image:{view}"
        if key not in self._cache:
            p = self.camera_path(view)
            if p is None:
                raise KeyError(f"frame {self.sample_id}: camera view {view!r} not in metadata")
            if not p.exists():
                raise FileNotFoundError(f"frame {self.sample_id}: camera file missing: {p}")
            from PIL import Image  # lazy import
            self._cache[key] = Image.open(p).convert("RGB")
        return self._cache[key]

    # -- pickled assets -----------------------------------------------------
    def ego_state(self) -> Optional["data_schema.EgoState"]:
        return self._load_pickle("ego_state", self.ego_state_relpath)

    def annotations(self) -> Optional["data_schema.Annotations"]:
        return self._load_pickle("annotations", self.annotations_relpath)

    def _load_pickle(self, key: str, relpath: str) -> Any:
        if key not in self._cache:
            if not relpath:
                self._cache[key] = None
            else:
                p = self.clip_dir / relpath
                with open(p, "rb") as fh:
                    self._cache[key] = pickle.load(fh)
        return self._cache[key]

    # -- reasoning ----------------------------------------------------------
    def reasoning(self) -> Optional[dict]:
        """Reasoning JSON as a dict, or ``None`` if no reasoning file for this frame.

        Reasoning is sparse (~1 Hz); most frames have no file. When a file exists,
        ``Driving`` / ``Counterfactual`` are ``dict`` when populated and the empty
        string ``""`` when not — callers should not assume ``dict``.
        """
        if "reasoning" not in self._cache:
            p = self.clip_dir / self.reasoning_relpath if self.reasoning_relpath else None
            if p is None or not p.exists():
                self._cache["reasoning"] = None
            else:
                self._cache["reasoning"] = json.loads(p.read_text())
        return self._cache["reasoning"]

    def driving_decision(self) -> Optional[dict]:
        """``{"Longitudinal", "Lateral"}`` if a non-empty decision exists, else ``None``."""
        r = self.reasoning()
        if not r:
            return None
        driving = r.get("Driving")
        if not isinstance(driving, dict):
            return None
        dec = driving.get("Driving decision")
        if not isinstance(dec, dict):
            return None
        lon = (dec.get("Longitudinal") or "").strip()
        lat = (dec.get("Lateral") or "").strip()
        if not lon and not lat:
            return None
        return {"Longitudinal": lon, "Lateral": lat}

    def reasoning_trace(self) -> Optional[str]:
        r = self.reasoning()
        driving = r.get("Driving") if r else None
        if not isinstance(driving, dict):
            return None
        trace = (driving.get("Reasoning trace") or "").strip()
        return trace or None

    def counterfactual(self) -> Optional[dict]:
        """``Counterfactual`` block (Alternative / Top safety-critical actions) or ``None``."""
        r = self.reasoning()
        cf = r.get("Counterfactual") if r else None
        return cf if isinstance(cf, dict) else None

    def camera_objects(self, view: str = "front") -> list[dict]:
        """GT per-camera detections from ``Spatial.per_camera_results[view].objects``.

        Each item holds ``category``, ``detection_label``, ``track_token``,
        ``detection_bbox_2d`` = ``[x1, y1, x2, y2]`` in **image pixel coords**, and
        ``detection_bbox_3d`` (ego-frame center/corners/size/yaw). Returns ``[]`` when
        the frame has no reasoning or no per-camera result for ``view``. These 2D boxes
        let a loaded frame's labels be checked directly against its camera image.
        """
        r = self.reasoning()
        spatial = r.get("Spatial") if r else None
        if not isinstance(spatial, dict):
            return []
        pcr = spatial.get("per_camera_results")
        if not isinstance(pcr, dict):
            return []
        cam = pcr.get(view)
        objs = cam.get("objects") if isinstance(cam, dict) else None
        return objs if isinstance(objs, list) else []

    def has_driving_decision(self) -> bool:
        return self.driving_decision() is not None

    def has_spatial(self) -> bool:
        """True if this frame carries a populated ``Spatial`` annotation (the ~1Hz set).

        These are the frames with a reasoning file at all (Spatial is always present when
        a reasoning file exists; Driving/CF are the sparser 0.2Hz subset). Superset of the
        ``driving`` keyframes — used to reproduce the paper's 1Hz annotated-frame usage.
        """
        r = self.reasoning()
        if not r:
            return False
        sp = r.get("Spatial")
        return isinstance(sp, dict) and bool(sp)


@dataclass
class Clip:
    """A parsed clip: metadata + lazily-loadable frames and map."""

    clip_dir: Path
    metadata: dict
    frames: list[Frame]
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # -- convenience metadata accessors ------------------------------------
    @property
    def clip_token(self) -> str:
        return self.metadata.get("clip_token", "")

    @property
    def scenario_type(self) -> Optional[str]:
        return self.metadata.get("scenario_type")

    @property
    def clip_location(self) -> Optional[str]:
        return self.metadata.get("clip_location")

    @property
    def frame_rate_hz(self) -> Optional[float]:
        return self.metadata.get("frame_rate_hz")

    @property
    def total_frames(self) -> Optional[int]:
        return self.metadata.get("total_frames")

    @property
    def camera_calibrations(self) -> dict:
        return self.metadata.get("camera_calibrations", {})

    def map(self) -> Any:
        """Lazily unpickle ``map.pkl`` (a ``data_schema.nuReasoningStaticMap``)."""
        if "map" not in self._cache:
            rel = self.metadata.get("map_annotation", "map.pkl")
            p = self.clip_dir / rel
            with open(p, "rb") as fh:
                self._cache["map"] = pickle.load(fh)
        return self._cache["map"]

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[Frame]:
        return iter(self.frames)

    def __repr__(self) -> str:  # concise, frames not dumped
        return (
            f"Clip(token={self.clip_token!r}, scenario={self.scenario_type!r}, "
            f"location={self.clip_location!r}, frames={len(self.frames)})"
        )


def parse_clip(clip_dir: PathLike) -> Clip:
    """Parse one extracted clip directory (the one containing ``metadata.json``)."""
    clip_dir = Path(clip_dir)
    meta_path = clip_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"no metadata.json in {clip_dir}")
    metadata = json.loads(meta_path.read_text())

    clip_token = metadata.get("clip_token", "")
    frames: list[Frame] = []
    for i, fr in enumerate(metadata.get("frames", [])):
        cams = (fr.get("sensors", {}) or {}).get("cameras", {}) or {}
        frames.append(
            Frame(
                clip_dir=clip_dir,
                clip_token=clip_token,
                index=i,
                frame_index=fr.get("frame_index", i),
                token=fr.get("token", ""),
                timestamp_us=fr.get("timestamp_us", 0),
                relative_time_s=fr.get("relative_time_s", 0.0),
                camera_relpaths={v: cams[v] for v in CAMERA_VIEWS if cams.get(v)},
                ego_state_relpath=fr.get("ego_state", ""),
                annotations_relpath=fr.get("annotations", ""),
                reasoning_relpath=fr.get("reasoning", ""),
                mission_goal=fr.get("mission_goal"),
            )
        )
    return Clip(clip_dir=clip_dir, metadata=metadata, frames=frames)


def find_clip_dirs(root: PathLike) -> list[Path]:
    """Discover clip directories under ``root`` (each contains a ``metadata.json``)."""
    root = Path(root)
    if (root / "metadata.json").exists():
        return [root]
    return sorted({p.parent for p in root.rglob("metadata.json")})


def iter_clips(root: PathLike) -> Iterator[Clip]:
    """Parse every clip found under ``root``, one at a time."""
    for d in find_clip_dirs(root):
        yield parse_clip(d)


# ===========================================================================
# Keyframe selection (scoping)
# ===========================================================================
def select_keyframes(clip: Clip, policy: str = "driving", n: int = 1) -> list[Frame]:
    """Pick keyframes from a clip to control scale.

    Policies:
      - ``"driving"`` (default): frames with a populated ``Driving decision``
        (the action-labelled frames the SFT/eval stages consume; ≈3 per clip).
      - ``"spatial"``: frames with a populated ``Spatial`` annotation (the ~1Hz set,
        ≈13 per clip; superset of ``driving``). Matches the paper's 1Hz frame usage.
      - ``"uniform"``: ``n`` evenly-spaced frames (n=1 -> the middle frame).
      - ``"all"``: every frame.
    """
    if policy == "driving":
        return [f for f in clip.frames if f.has_driving_decision()]
    if policy == "spatial":
        return [f for f in clip.frames if f.has_spatial()]
    if policy == "all":
        return list(clip.frames)
    if policy == "uniform":
        total = len(clip.frames)
        if total == 0 or n <= 0:
            return []
        if n == 1:
            return [clip.frames[total // 2]]
        if n >= total:
            return list(clip.frames)
        idxs = sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})
        return [clip.frames[i] for i in idxs]
    raise ValueError(f"unknown keyframe policy: {policy!r} (use driving|spatial|uniform|all)")


# ===========================================================================
# (b) Dataset wrapper
# ===========================================================================
def _default_sample(frame: Frame) -> dict:
    """Default ``__getitem__`` output: identity + the lazy ``Frame`` handle.

    Consumers override this via ``sample_fn`` to emit their own record shape, but the
    raw frame is always reachable so nothing is lost.
    """
    return {"id": frame.sample_id, "frame": frame}


ClipSource = Union[PathLike, Clip, Iterable[Union[PathLike, Clip]]]


def _coerce_clips(source: ClipSource) -> list[Clip]:
    """Accept a root path, a clip dir, a ``Clip``, or a list of those."""
    if isinstance(source, Clip):
        return [source]
    if isinstance(source, (str, Path)):
        return [parse_clip(d) for d in find_clip_dirs(source)]
    clips: list[Clip] = []
    for item in source:
        if isinstance(item, Clip):
            clips.append(item)
        else:
            clips.extend(parse_clip(d) for d in find_clip_dirs(item))
    return clips


class NuReasoningDataset(_TorchDataset):
    """Flat dataset over selected keyframes across one or more clips.

    Parameters
    ----------
    source : root dir / clip dir / ``Clip`` / iterable of those.
    keyframe_policy, n_keyframes : forwarded to :func:`select_keyframes`.
    sample_fn : ``Frame -> Any`` builder for ``__getitem__`` (stage-specific output).
                Defaults to ``{"id", "frame"}``.
    """

    def __init__(
        self,
        source: ClipSource,
        *,
        keyframe_policy: str = "driving",
        n_keyframes: int = 1,
        sample_fn: Optional[Callable[[Frame], Any]] = None,
    ) -> None:
        self.clips = _coerce_clips(source)
        self.keyframe_policy = keyframe_policy
        self.n_keyframes = n_keyframes
        self.sample_fn = sample_fn or _default_sample
        self.samples: list[Frame] = []
        for clip in self.clips:
            self.samples.extend(select_keyframes(clip, keyframe_policy, n_keyframes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Any:
        return self.sample_fn(self.samples[i])

    def __repr__(self) -> str:
        return (
            f"NuReasoningDataset(clips={len(self.clips)}, samples={len(self.samples)}, "
            f"policy={self.keyframe_policy!r})"
        )


__all__ = [
    "CAMERA_VIEWS",
    "Frame",
    "Clip",
    "parse_clip",
    "find_clip_dirs",
    "iter_clips",
    "select_keyframes",
    "NuReasoningDataset",
]
