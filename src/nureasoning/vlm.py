"""Shared VLM harness config — single source of truth for the vision-token cap, base
model id, and seed used by BOTH training and evaluation.

Lives in the neutral ``nureasoning`` spine (not under train or eval) so neither module
owns the other: the fine-tuned-vs-zero-shot comparison is only fair if both use the
*same* image preprocessing, so that contract belongs in one place both import — rather
than the (historical) arrangement where the eval module defined it and training reached
across to borrow it.

``transformers`` is imported lazily inside :func:`load_processor` so ``import nureasoning``
stays lightweight for non-model consumers (e.g. retrieval EDA / the parser).

논문(nuVLA) vs 미니프로젝트 VLM 비교
======================================
| 항목              | 논문 nuReasoning/nuVLA            | 미니프로젝트 (우리)                          |
|-------------------|----------------------------------|---------------------------------------------|
| Clip size         | 20,000(17,000 2,000 1,000)       | 4,148(3,108 1,040)                          |
| VLM 백본          | Qwen3-VL-2B                      | Qwen3-VL-2B (논문 동일·후보 Qwen2.5-VL-3B)    |
| 추가 헤드         | flow-matching DiT (궤적 planning) | flow-matching DiT 재현(trajectory objective) |
| 학습 방식         | Full fine-tune (VLM+헤드)         | objective 분기: text_sft=4-bit QLoRA / trajectory=VLM+DiT 공동학습(--vlm-mode full(기본)·lora·frozen) |
| 이미지 해상도     | 448×448 고정 리사이즈 (256토큰)     | 동적 해상도, 종횡비 보존 416×256 (104토큰)    |
| 배치 사이즈       | 64                                | 1~                                           |
| 입력 뷰 수        | 8뷰                               | 8뷰                                         |
| 학습 태스크       | 궤적 + reasoning supervision      | text_sft: driving(action+reasoning)+spatial(객체) / trajectory: 미래 궤적 + reasoning 공동(flow_loss+λ·LM_loss) |
| 옵티마이저        | AdamW                            | AdamW / paged8bit AdamW (2B optimizer state 압축) |

현재 우리 프로젝트의 샘플 구성
| Split | samples | unique clips  | spatial | trajectory | driving |
|-------|---------|---------------|---------|------------|---------|
| train | 3,462   | 3,108         | 3,108   | 3,108      | 3,108   |
| val   | 1,160   | 1,040         | 520     | 520        | 120     |
"""
from __future__ import annotations

# 기본 베이스 VLM = 원본 논문(nuVLA) 백본과 동일한 Qwen3-VL-2B → 논문 재현·공정 비교용.
# transformers 5.12.1이 qwen3_vl 아키텍처를 정식 지원(AutoModelForImageTextToText 매핑 포함).
DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"  # 후보(대체본): 같은 Qwen-VL 계열, 3B QLoRA가 단일 24GB에 더 검증됨
SEED = 12345 

# 이미지 해상도 캡(학습·평가 공통). Qwen-VL은 max_pixels=N*28*28 → 이미지당 비전토큰 상한 제어.
# 8뷰 + spatial 멀티태스크를 한 시퀀스에 넣으면 256예산은 단일 GPU(24GB) QLoRA에서 OOM →
# 160으로 낮춰 억제. (참고: 실효 토큰 수는 백본 patch_size에 따라 다름 — Qwen2.5-VL patch14는
# 2816×1856→~150토큰, Qwen3-VL patch16는 동일 캡에서 ~104토큰. 캡 자체는 두 백본 공통 적용.)
#
# 토큰캡 → 실제 리사이즈 크기 (우리 데이터 종횡비 2816×1856 기준, Qwen smart_resize):
#   Qwen-VL은 토큰당 28×28px(14px 패치 × 2×2 merge). 종횡비를 보존하며 픽셀 수를 캡 이하로 맞추고
#   가로·세로를 28의 배수로 반올림 → 실제 토큰 수는 캡보다 약간 적을 수 있다(예: 160캡 → 실제 150).
#
#   | 토큰캡 | max_pixels | 리사이즈(가로×세로) | 실제 토큰 |
#   |--------|-----------|---------------------|-----------|
#   |     64 |     50176 |       252 × 168     |     54    |
#   |    104 |     81536 |       336 × 224     |     96    |
#   |    128 |    100352 |       364 × 252     |    117    |
#   | >>160<<|    125440 |       420 × 280     |    150    | ← 현재 설정
#   |    256 |    200704 |       532 × 336     |    228    |
#   |    512 |    401408 |       756 × 504     |    486    |
#
#   (다른 종횡비 입력이면 리사이즈 크기·실제 토큰은 달라짐. 위는 우리 8뷰 카메라 표준 크기 기준.)
IMAGE_MAX_PIXELS = 160 * 28 * 28
IMAGE_MIN_PIXELS = 64 * 28 * 28

# 정사각형 강제 리사이즈 스위치(학습·평가 공통). 두 전처리 정책 중 택1:
#   None  = Qwen 네이티브 동적 해상도. 종횡비 보존(2816×1856 → 420×280 직사각형, ~150토큰). 기본.
#   정수 S = 입력을 S×S로 **강제 리사이즈**(종횡비 무시) 후 투입 → 논문(nuVLA) 448×448 방식 재현.
#           예: 448이면 1:1 → 16×16=256토큰. ⚠️ 가로로 넓은 카메라가 횡압축돼 객체가 찌그러짐 +
#           토큰↑(메모리↑). 둘은 근본적으로 다른 전처리라 비교 실험 시 한쪽으로 통일해야 공정.
IMAGE_SQUARE: "int | None" = None

# ─── SFT 데이터셋 선택(학습·평가 공통 단일 출처) ──────────────────────────────────
# 어떤 SFT 빌드를 학습/평가에 쓸지 여기서 중앙 결정한다. build_sft.py는 이 경로에 {train,val}.jsonl을
# 만들고(--out 미지정 시), train_traj_reas.py/eval_*.py는 이 경로를 기본 train/val로 읽는다(--train/--val로
# override 가능). 새 빌드를 낼 때 기존을 덮지 않으려면 이 값만 바꾸면 된다(예: data/sft → data/sft_v2).
#   sft    : 초기 빌드.
#   sft_v2 : gate_direction(과거 결정 이월, selective-view 폐루프 게이트 타깃) 포함 재빌드.
import pathlib as _pathlib
_REPO = _pathlib.Path(__file__).resolve().parents[2]
SFT_DIR: "_pathlib.Path" = _REPO / "data" / "sft_v2"
SFT_TRAIN: "_pathlib.Path" = SFT_DIR / "train.jsonl"
SFT_VAL: "_pathlib.Path" = SFT_DIR / "val.jsonl"

# ─── 시간 맥락(temporal multi-frame) 입력 스위치 ──────────────────────────────────
# 논문 nuVLA는 condition으로 "현재 + 과거 timestep"(최대 2×8=16장)을 쓴다. 미니프로젝트 기본은
# 단일 프레임(현재 8뷰)이지만, 아래 두 값으로 과거 1 timestep을 더 넣어 논문 방식을 재현할 수 있다.
#   TEMPORAL : True면 과거 프레임 1개(8뷰)를 현재 8뷰와 **함께** VLM에 투입. False면 현재 프레임만(기본).
#   TEMPORAL_HISTORY_OFFSET : 과거 프레임을 현재 키프레임에서 "몇 프레임 전"으로 잡을지(10Hz 수집 기준).
#       offset=10 → 1.0초 전, 5 → 0.5초 전, 1 → 0.1초 전. ✅ 이제 클립을 **10Hz 전체 이미지**(--policy all)로
#       받으므로 **모든 정수 offset이 유효**하다(과거 어떤 프레임에도 8뷰 이미지가 존재) — 과거의
#       1Hz-only 서브셋 시절과 달리 "offset은 10의 배수" 제약이 사라졌다. 단 클립 시작 이전
#       (f.index-offset<0)인 초반 프레임은 과거뷰가 없어 그 샘플만 단일 프레임 처리된다(hit 수 리포트).
# 적용 흐름(단일 출처): 이 두 값은 **build_sft.py가 읽어** SFT 레코드에 history_images(과거 8뷰 경로)를
#   심는다 — offset 적용은 클립 프레임 구조가 필요해 빌드 시점에만 가능하다(⇒ offset/TEMPORAL을 바꾸면
#   데이터 재빌드 필요). 학습은 TEMPORAL이 True이고 데이터에 history_images가 있을 때만 그 과거뷰를
#   시퀀스에 추가하고, 그 사실을 traj_config.json(temporal 플래그)에 적는다. 평가는 그 config를 따라
#   동일하게 처리하므로 빌드·학습·평가가 항상 일관된다.
# ⚠️ 메모리: 과거뷰를 켜면 이미지 토큰이 8→16장으로 ~2배 → VLM forward 메모리도 ~2배(OOM 주의).
TEMPORAL: bool = False  # 4단계 ablation은 현재 프레임(8뷰)만 사용(비용↓, reasoning 효과에 집중). 데이터엔
                        # history_images가 심겨 있어 True로 바꾸면 재빌드 없이 16뷰 시간맥락을 켤 수 있음.
TEMPORAL_HISTORY_OFFSET: int = 10 # 10 프레임 전 = 1초 전(10Hz)


# load_image: 학습·평가 공통 이미지 로더(단일 진입점). IMAGE_SQUARE가 정수면 그 크기 정사각형으로
# 강제 리사이즈(논문 방식), None이면 원본 그대로 두고 프로세서의 동적 해상도(smart_resize)에 맡긴다.
# 모든 이미지 로딩을 이 함수로 통일해야 전처리 정책이 한곳에서 일관 적용된다(PIL 지연 import).
def load_image(path):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if IMAGE_SQUARE:
        img = img.resize((IMAGE_SQUARE, IMAGE_SQUARE))   # 종횡비 무시 강제 1:1(논문 nuVLA 방식)
    return img


# load_processor: 비전토큰 캡을 적용한 프로세서(학습·평가 공통 진입점). transformers 지연 import.
def load_processor(model_id: str = DEFAULT_MODEL):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(
        model_id, min_pixels=IMAGE_MIN_PIXELS, max_pixels=IMAGE_MAX_PIXELS,
    )


# resolve_path: SFT/manifest 레코드의 이미지 경로를 실제 파일 경로로 복원(학습·평가 공통).
#   클립 데이터가 REPO 밖(예: /home/etri/DATASET/nureasoning)에 있으면 build_sft가 **절대경로**로
#   저장하므로 그대로 쓰고, 과거 레코드처럼 REPO 기준 상대경로면 base를 접두한다 → 두 형식 모두 로드 가능.
def resolve_path(p, base):
    from pathlib import Path
    p = Path(p)
    return p if p.is_absolute() else Path(base) / p


__all__ = [
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
