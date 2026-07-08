"""nuReasoning common loader/parser package (the shared spine for all pipeline stages)."""
from .loader import (
    CAMERA_VIEWS,
    VIEW_LABELS,
    Clip,
    Frame,
    NuReasoningDataset,
    find_clip_dirs,
    iter_clips,
    parse_clip,
    select_keyframes,
)
from .geometry import (  # global UTM → ego-frame 궤적 변환(planning 학습/평가 공통)
    future_waypoints_ego,
    global_to_ego,
    past_waypoints_ego,
    upsample_waypoints,
)
from .vlm import (  # 학습·평가 공통 VLM 설정(단일 출처) — transformers는 load_processor 호출 시 지연 import
    DEFAULT_MODEL,
    IMAGE_MAX_PIXELS,
    IMAGE_MIN_PIXELS,
    IMAGE_SQUARE,
    SEED,
    SFT_DIR,
    SFT_TRAIN,
    SFT_VAL,
    TEMPORAL,
    TEMPORAL_HISTORY_OFFSET,
    load_image,
    load_processor,
    resolve_path,
)

__all__ = [
    "CAMERA_VIEWS",
    "VIEW_LABELS",
    "Clip",
    "Frame",
    "NuReasoningDataset",
    "find_clip_dirs",
    "iter_clips",
    "parse_clip",
    "select_keyframes",
    # 궤적 좌표 변환
    "global_to_ego",
    "future_waypoints_ego",
    "past_waypoints_ego",
    "upsample_waypoints",
    # VLM 공통 설정
    "DEFAULT_MODEL",
    "SEED",
    "IMAGE_MAX_PIXELS",
    "IMAGE_MIN_PIXELS",
    "IMAGE_SQUARE",
    "SFT_DIR",
    "SFT_TRAIN",
    "SFT_VAL",
    "TEMPORAL",
    "TEMPORAL_HISTORY_OFFSET",
    "load_image",
    "load_processor",
    "resolve_path",
]
