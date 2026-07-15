"""VLM 백본 + flow-matching DiT 궤적 planning **+ reasoning 공동 supervision** 학습 (objective=trajectory).

논문 nuVLA의 "VLM + DiT 궤적 헤드 + reasoning supervision"을 한 학습에서 **공동**으로 재현한다
(text_sft 경로 `train_qlora.py`와 대등한 분기). 핵심은 두 손실을 한 forward로 함께 거는 것:
  - **flow_loss** (궤적): VLM이 8뷰+planning 프롬프트를 인코딩한 condition으로 TrajectoryDiT가 ego-frame
    미래 waypoints(N×2)를 flow-matching 회귀.
  - **LM_loss** (reasoning): 같은 VLM이 그 장면의 reasoning_trace를 텍스트로 생성(cross-entropy).
    reasoning은 0.2Hz(~23%) 프레임에만 있으므로 **있는 샘플에만** LM_loss를 더한다.
  - **공동 손실**: total = flow_loss + λ·LM_loss (λ=`--reas-weight`). reasoning을 배우는 게 planning
    표현을 돕는다는 논문 가설을 재현. HF Trainer 대신 커스텀 루프(공동 backward).

⚡ 한 forward로 둘 다 뽑는 트릭: Qwen-VL은 causal decoder라 "이미지+프롬프트" 토큰의 hidden은 뒤에
reasoning이 붙든 말든 **불변** → full 시퀀스(이미지+프롬프트+reasoning) 1회 forward에서 (a) reasoning
부분에 LM_loss, (b) 프롬프트까지 hidden을 mean-pool해 condition을 동시에 얻고, 이 condition이 평가
시(이미지+프롬프트만 forward)와 정확히 일치한다.

VLM 학습 범위 = ``--vlm-mode`` (논문은 VLM+헤드를 함께 full fine-tune):
  - **full** (기본, 논문 충실): VLM 전 파라미터 + DiT 공동 학습. ⚠️ 2B+8뷰는 단일 24GB OOM 위험 →
    gradient checkpointing + batch 1 + `--optimizer paged8bit`(2B optimizer state 압축)로 대응(빠듯하면 lora로).
  - **lora**: VLM 4-bit QLoRA + DiT → 24GB 안전. **frozen**: DiT만(VLM 미학습 → reasoning 학습 불가).
    ⚠️ frozen은 grad가 VLM에 안 흘러 reasoning supervision이 무의미 → 공동학습은 full/lora에서만 유효.

설계 메모:
  - waypoints는 TrajectoryNormalizer로 정규화(평균0·표준편차1)해 flow-matching 안정화. 평가 시 역정규화.
  - condition dim(Dc)은 모델 config 대신 첫 샘플을 probe해 측정(백본 교체에 견고).
  - full/lora면 학습된 VLM(전체/어댑터)을 `<out>/vlm`에 저장 → 평가가 동일 구성 복원.

멀티-GPU: `torchrun --nproc_per_node=N`으로 띄우면 GPU당 1프로세스 **표준 DDP**로 학습한다(각 rank가 자기
GPU에 VLM+DiT를 두고, 데이터를 rank별로 샤딩, gradient는 DDP 훅이 backward에서 자동 all-reduce로 동기화).
VLM+DiT를 TrajReasVLA(nn.Module)로 묶어 DDP로 감싸며, reasoning 유무로 lm_head가 조건부로 쓰이는 문제는
0-더미(0·logits)로 lm_head를 매 step 그래프에 연결해(find_unused_parameters=False로) 회피한다. torchrun 없이
그냥 `python ...`이면 단일 GPU로 폴백한다(DDP 미적용). DataParallel이 아니라 DDP인 이유: 4-bit QLoRA·
device_map 고정 모델과 호환되는 유일한 멀티-GPU 경로(자세한 건 `training/ddp.py`).

Run(단일 GPU):  CUDA_VISIBLE_DEVICES=0 python -m training.train_traj_reas [--vlm-mode full|lora|frozen] ...
Run(멀티 GPU): torchrun --nproc_per_node=4 -m training.train_traj_reas --vlm-mode full --reas-weight 1.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from contextlib import nullcontext
from functools import partial
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # DataLoader 워커 fork 시 토크나이저 병렬 경고/데드락 방지

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch                                              # noqa: E402  (Dataset/collate/forward_batch가 top-level torch 사용)
import torch.nn as nn                                     # noqa: E402  (TrajReasVLA 래퍼 베이스)
from torch.utils.data import DataLoader                   # noqa: E402  (프리페치 데이터 파이프라인)

from nureasoning import (  # noqa: E402  (학습·평가 공통 백본/캡/시간맥락 설정)
    DEFAULT_MODEL, IMAGE_MAX_PIXELS, SEED, SFT_TRAIN, SFT_VAL, TEMPORAL, TEMPORAL_HISTORY_OFFSET,
    VIEW_LABELS, load_image, load_processor, resolve_path, upsample_waypoints,
)
from training import ddp                                # noqa: E402  (torchrun DDP 헬퍼)
from training.dit_head import TrajectoryDiT, TrajectoryNormalizer, GateHead, focal_loss  # noqa: E402
from training.run_logging import RunLogger              # noqa: E402  (실시간 step·epoch 종합 로깅)
from training.train_text_sft import LORA_TARGETS       # noqa: E402  (VLM LoRA 타깃, text_sft와 동일)

DEFAULT_TRAIN = SFT_TRAIN     # 어떤 SFT를 쓸지는 vlm.SFT_DIR이 중앙 결정(현재 data/sft_v2). --train으로 override.
DEFAULT_VAL = SFT_VAL
TRAJ_PROMPT_FILE = REPO / "prompts" / "trajectory_plan_v1.txt"


# TrajDataset: train.jsonl에서 task=="trajectory" 레코드만 보관(원본 dict 그대로, 텐서화는 루프에서).
class TrajDataset:
    def __init__(self, path: Path, limit: int = 0):
        recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        self.records = [r for r in recs if r.get("task") == "trajectory"]
        if limit:
            self.records = self.records[:limit]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]

    def waypoints(self):
        return [r["output"]["waypoints"] for r in self.records]


# group_records_by_clip: trajectory 레코드를 clip_token별로 묶고 frame_index 오름차순 정렬한 clip 리스트 반환.
#   temporal-clip(시퀀스) 경로가 clip 단위로 시간순 롤아웃하는 데 쓴다. frame_index가 없는 구버전 데이터는
#   id의 "..._<frameidx>_tj"에서 복원(하위호환). 반환: [[rec0,rec1,...](clipA 시간순), [...](clipB), ...].
def group_records_by_clip(records: list) -> list:
    import re
    by_clip: dict = {}
    for r in records:
        tok = r.get("clip_token", "?")
        by_clip.setdefault(tok, []).append(r)

    def _fidx(r):
        fi = r.get("frame_index")
        if fi is not None:
            return fi
        m = re.match(r".*_(\d+)_tj$", r.get("id", ""))
        return int(m.group(1)) if m else 0

    clips = []
    for tok in sorted(by_clip):                            # clip_token 정렬 → 전 rank 동일 순서(샤딩 재현성)
        clips.append(sorted(by_clip[tok], key=_fidx))
    return clips


# load_vlm: vlm_mode에 따라 VLM 로드/학습 구성.
#   full   = bf16 전체 학습(gradient checkpointing). 논문 충실, 24GB OOM 위험.
#   lora   = 4-bit QLoRA(어댑터만 학습). VLM도 학습하되 메모리 안전.
#   frozen = 4-bit 동결(DiT만 학습). 가장 가벼움.
def load_vlm(model_id: str, vlm_mode: str = "full", device_index: int = 0):
    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    # DDP: 각 rank는 자기 GPU(local_rank)에만 모델을 둔다 → device_map을 그 인덱스로 고정.
    dmap = {"": device_index}

    if vlm_mode == "full":
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=dmap,
        )
        model.config.use_cache = False
        # activation 메모리 절감(full FT 필수). use_reentrant=False = 표준 DDP와 gradient checkpointing 공존
        #   (reentrant면 backward 재계산이 DDP 훅을 두 번 울려 "marked ready twice" 에러 → static_graph 없이 회피).
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        for p in model.parameters():
            p.requires_grad_(True)
        return model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb, dtype=torch.bfloat16, device_map=dmap,
    )
    if vlm_mode == "lora":
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        model.config.use_cache = False
        model = prepare_model_for_kbit_training(  # use_reentrant=False = 표준 DDP + gradient checkpointing 공존
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
        lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.1, bias="none",   # 논문 스펙(r32/α64/dropout0.1)
                          task_type="CAUSAL_LM", target_modules=LORA_TARGETS)
        model = get_peft_model(model, lora)
        return model

    # frozen
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# 시간+뷰 결합 태그(논문 nuVLA 포맷): 각 이미지 **바로 앞**에 "t = {ts}, {view} camera" 한 줄을 둬,
# 모델이 이미지마다 (어느 시점·어느 뷰)인지 자체적으로 식별하게 한다 — 논문 "Each image is inserted with
# an explicit textual tag indicating both its relative time and camera view"를 그대로 따른다(이미지
# 리스트를 순서 없는 집합으로 다루지 않게 함). t=0이 현재 프레임(=current), t=-1이 과거 1 timestep.
# 뷰 라벨은 VIEW_LABELS를 소문자화해 논문 예시("front camera", "front-left camera") 표기와 일치시킨다.
#   예) "t = 0 (current), front-left camera"  /  "t = -1, front camera"
# ⚠️ 우리는 과거 timestep을 1개만 쓰므로(논문 기본 설정과 동일) 과거는 항상 t=-1. 실제 시간 간격은
#    vlm.TEMPORAL_HISTORY_OFFSET(프레임)로 정해지지만(기본 10프레임=1.0초 전), 논문 표기는 timestep
#    인덱스라 초가 아닌 t=-1로 적는다.
def _image_tag(t: int, view: str) -> str:
    label = VIEW_LABELS.get(view, view).lower()           # "Front-left" → "front-left"(논문 표기)
    when = "t = 0 (current)" if t == 0 else f"t = {t}"
    return f"{when}, {label} camera"


# history_for: temporal_on일 때만 레코드의 과거 8뷰(history_images)를 반환(아니면 None). 학습·평가가
#   "과거뷰를 넣을지"를 한 줄로 일관 결정 — temporal_on은 학습=cfg/플래그, 평가=traj_config.json에서 옴.
def history_for(rec: dict, temporal_on: bool):
    return rec.get("history_images") if temporal_on else None


# ─── selective-view 게이팅(maneuver_lateral 기반 뷰 on/off) ───────────────────────────────
# 대부분 직진이라 8뷰 전부는 비효율 → 기동(궤적 유도 maneuver_lateral)에 따라 뷰 부분집합만 VLM에 투입.
#   항상: 전면 3뷰(front_left=CAM_M_L0, front=CAM_M_F, front_right=CAM_M_R0)
#   ml > +thr (좌회전/좌차선변경): 좌측 3뷰 추가(left=CAM_M_L1, back_left=CAM_M_L2, back=CAM_M_B)
#   ml < −thr (우회전/우차선변경): 우측 3뷰 추가(right=CAM_M_R1, back_right=CAM_M_R2, back=CAM_M_B)
#   |ml| ≤ thr (직진): 전면 3뷰만
# thr<0/None이면 게이팅 off = 전 8뷰(baseline). ml None(구형 데이터)이면 안전하게 전뷰 유지.
GATE_ALWAYS = ("front_left", "front", "front_right")     # CAM_M_L0 / CAM_M_F / CAM_M_R0
GATE_LEFT = ("left", "back_left", "back")                # CAM_M_L1 / CAM_M_L2 / CAM_M_B
GATE_RIGHT = ("right", "back_right", "back")             # CAM_M_R1 / CAM_M_R2 / CAM_M_B


def gate_views(image_recs, maneuver_lateral, thr):
    """maneuver_lateral·thr로 뷰 부분집합 선택. image_recs=None이면 None(과거뷰 없음 그대로)."""
    if image_recs is None:
        return None
    if thr is None or thr < 0 or maneuver_lateral is None:
        return image_recs                                # 게이팅 off 또는 신호 없음 → 전뷰
    keep = set(GATE_ALWAYS)
    if maneuver_lateral > thr:
        keep |= set(GATE_LEFT)
    elif maneuver_lateral < -thr:
        keep |= set(GATE_RIGHT)
    return [im for im in image_recs if im.get("view") in keep]


def rec_maneuver(rec: dict):
    """레코드의 maneuver_lateral(궤적 유도 기동 신호). 없으면 None."""
    return (rec.get("output") or {}).get("maneuver_lateral")


# ─── selective-view 게이팅(gate_direction 기반 뷰 on/off, 폐루프용) ────────────────────────────
# gate_views와 달리 **이산 방향 라벨**(0=straight/1=left/2=right)로 뷰를 켠다. 이 라벨의 출처는:
#   학습: build_sft가 심은 gate_direction GT(과거 결정 이월, teacher-forcing).
#   추론: 게이트가 이전 프레임에서 예측한 방향(진짜 폐루프).
# direction=None(과거 관측 없음=clip 첫 구간)이면 안전하게 전 8뷰(baseline) 유지 — 아직 방향을 모르므로.
def gate_views_by_direction(image_recs, direction):
    """direction: 0=straight/1=left/2=right/None(전뷰). None이면 게이팅 안 함(전 8뷰)."""
    if image_recs is None:
        return None
    if direction is None:                                # 방향 미상(첫 구간) → 안전하게 전뷰
        return image_recs
    keep = set(GATE_ALWAYS)
    if direction == 1:                                   # left
        keep |= set(GATE_LEFT)
    elif direction == 2:                                 # right
        keep |= set(GATE_RIGHT)
    # direction==0(straight)면 전면 3뷰만
    return [im for im in image_recs if im.get("view") in keep]


def rec_gate_direction(rec: dict):
    """레코드의 gate_direction(0/1/2, 과거 결정 이월 GT). 없으면 None."""
    return (rec.get("output") or {}).get("gate_direction")


# is_keyframe: 이 trajectory 레코드가 **decision 프레임**(0.2Hz, Driving decision 채워진 프레임, clip당 3개)인가.
#   ① build_sft가 심은 output.keyframe 플래그 우선(신규 빌드). ② 없으면(기존 sft_v2) reasoning_parts.decision
#   존재로 폴백 → **재빌드 없이도** 판정 가능(decision은 이 프레임에만 채워지므로 동치).
#   ⚠️ 이건 "decision 프레임(3개)"이지 "평가용 keyframe"이 아니다 — 평가 채점 대상은 select_keyframe_ids로 좁힌다.
def is_keyframe(rec: dict) -> bool:
    out = rec.get("output") or {}
    if "keyframe" in out:
        return bool(out["keyframe"])
    return bool((out.get("reasoning_parts") or {}).get("decision"))


# select_keyframe_ids: **clip-그룹 단위**로 평가 채점 대상 keyframe을 고른다. 각 clip의 decision 프레임들
#   (frame_index 오름차순)에서 **순서(ordinal)** select_idx에 해당하는 것만 골라 그 레코드 id 집합을 반환.
#   순서 기반인 이유: 절대 frame_index가 clip마다 50/100/150 또는 51/101/151로 어긋나므로(1프레임 오프셋),
#   "몇 번째 decision 프레임"으로 매칭해야 안전. 논문은 clip당 keyframe 1개 → select_idx=(1,)=정중앙(2번째)이
#   정황상 최선. select_idx=(0,1,2)면 3개 전부(현재까지 동작). records=평가 대상 trajectory 레코드 전체.
def select_keyframe_ids(records: list, select_idx=(0, 1, 2)) -> set:
    from collections import defaultdict
    byclip = defaultdict(list)
    for r in records:
        if is_keyframe(r):
            byclip[r.get("clip_token")].append(r)
    keep = set()
    sel = set(select_idx)
    for recs_c in byclip.values():
        recs_c.sort(key=lambda r: (r.get("frame_index") is None, r.get("frame_index")))  # decision 프레임 시간순
        for i, r in enumerate(recs_c):
            if i in sel:                                   # i번째 decision 프레임만 채점 대상
                keep.add(r["id"])
    return keep


# ─── reasoning 종류 선택(단계별 ablation: baseline→+spatial→+decision→+counterfactual) ──────────
# build_sft가 궤적 샘플에 심은 output.reasoning_parts({spatial?,decision?,counterfactual?})에서 --reasoning-types로
# 고른 종류만 하나의 텍스트로 합쳐 LM_loss 타깃으로 쓴다. 논문의 "all reasoning types 공동 supervision"을
# 단계적으로 재현: []=baseline(궤적만) → [spatial] → [spatial,decision] → [spatial,decision,counterfactual].
REASONING_ORDER = ("spatial", "decision", "counterfactual")   # 합칠 때 항상 이 표준 순서


def parse_reasoning_types(s: str) -> list:
    """'none'/'' → [](baseline). 'spatial,decision' 등 → 표준 순서로 정렬된 종류 리스트."""
    s = (s or "").strip().lower()
    if s in ("", "none"):
        return []
    want = {t.strip() for t in s.split(",") if t.strip()}
    unknown = want - set(REASONING_ORDER)
    if unknown:
        raise SystemExit(f"--reasoning-types에 알 수 없는 값 {sorted(unknown)} (허용: {list(REASONING_ORDER)} 또는 none)")
    return [t for t in REASONING_ORDER if t in want]


def reasoning_target(rec: dict, types: list) -> "str | None":
    """선택된 reasoning 종류를 표준 순서로 합쳐 하나의 LM 텍스트 타깃 생성(없으면 None).
    신형 데이터(output.reasoning_parts) 우선, 구형(output.reasoning=decision)은 하위호환 폴백."""
    if not types:
        return None
    out = rec.get("output", {})
    parts = out.get("reasoning_parts")
    if parts:
        segs = [f"{t.capitalize()}: {parts[t]}" for t in types if parts.get(t)]
    elif "decision" in types and out.get("reasoning"):        # 구형 데이터 폴백(decision만 존재)
        segs = [f"Decision: {out['reasoning']}"]
    else:
        segs = []
    return "\n".join(segs) if segs else None


# has_reasoning_annotation: 이 궤적 샘플에 reasoning 주석이 하나라도 붙어 있나(--reasoning-types와 무관).
#   논문의 "annotated frames only" 학습 = reasoning 주석 있는 프레임(clip당 frame 30~150, 1Hz)에서만 planning+
#   reasoning 공동 supervise. reasoning-free 프레임(clip 앞 0,10,20 + tail)은 미래·과거 궤적 경계 밖이라 ego
#   history도 퇴화 → 논문이 제외하는 프레임. **reasoning-types 선택과 독립**으로 판정해야 ablation(baseline/
#   +spatial/+decision/+cf)이 동일 프레임 집합에서 돌아 공정하다. 판정은 output.reasoning_parts(신형, spatial
#   포함) 우선, 구형 output.reasoning(decision만) 폴백 — reasoning_target과 달리 종류에 상관없이 '존재'만 본다.
def has_reasoning_annotation(rec: dict) -> bool:
    out = rec.get("output", {})
    return bool(out.get("reasoning_parts")) or bool(out.get("reasoning"))


# filter_reasoning_only: reasoning 주석 있는 궤적 샘플만 남긴다(논문 정렬). n_before/n_after 반환(로깅용).
def filter_reasoning_only(records: list) -> "tuple[list, int, int]":
    n0 = len(records)
    kept = [r for r in records if has_reasoning_annotation(r)]
    return kept, n0, len(kept)


# _append_views: content/images 리스트에 [시간+뷰 결합 태그 + 이미지]×N을 인터리브로 덧붙인다.
#   t = 해당 프레임의 timestep 인덱스(0=현재, -1=과거 1 timestep). 각 이미지 앞에 _image_tag로
#   "t = …, … camera"를 붙여 논문 nuVLA 포맷(시간·뷰 결합 단일 태그)을 그대로 따른다(과거·현재 공용).
def _append_views(content: list, images: list, image_recs: list, t: int = 0) -> None:
    for im in image_recs:
        img = load_image(resolve_path(im["path"], REPO))   # 절대경로면 그대로, 상대면 REPO 기준(정사각형 정책 공통)
        content.append({"type": "text", "text": _image_tag(t, im["view"])})
        content.append({"type": "image", "image": img})
        images.append(img)


# encode_condition: (과거 8뷰 옵션 +) 현재 8뷰 + planning 프롬프트 → VLM hidden mean-pool → condition (Dc,) fp32.
# generate가 아니라 forward(output_hidden_states)만 호출. attention mask로 패딩 제외 평균.
# history_recs가 있으면 과거 1 timestep(8뷰)을 현재 앞에 둬 논문 nuVLA식 시간 맥락을 만든다.
# ⚠️ gradient 제어는 **호출측**이 한다(학습 full/lora면 grad 흐름, frozen·평가면 no_grad로 감쌈).
def encode_condition(model, processor, image_recs: list, prompt: str, history_recs: list | None = None):
    content, images = _build_views_content(image_recs, prompt, history_recs)
    messages = [{"role": "user", "content": content}]
    # add_generation_prompt=True로 통일 → encode_and_lm_loss의 condition(프롬프트+"assistant\n"까지)과 동일 범위.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[-1]                              # (1, seq, hidden) 마지막 레이어
    amask = inputs["attention_mask"] > 0                    # (1, seq) 유효(=프롬프트, reasoning 미첨부라 전부)
    mask = amask.unsqueeze(-1).to(hs.dtype)                 # (1, seq, 1)
    pooled = (hs * mask).sum(1) / mask.sum(1).clamp_min(1.0)     # (1, hidden) 패딩 제외 평균
    # 반환: (pooled cond (hidden,), mem 시퀀스 (1,seq,hidden), mem_mask (1,seq)). cross_attn이면 mem 사용,
    #   구형이면 cond 사용. mem은 batch 차원 보존(sample/flow_loss가 그대로 받음). cond는 기존 계약대로 squeeze.
    return pooled.squeeze(0).float(), hs, amask


# gate_closedloop_encode: 폐루프 추론(방법 A의 추론측) — 게이트 예측으로 뷰를 게이팅해 planning cond 생성.
#   1) **전방 3뷰**로 VLM encode → cond_gate → 게이트가 방향 예측(argmax). 설계 취지(전방만 보고 판단)에 정합.
#   2) 그 예측 방향으로 뷰 게이팅 → 다시 encode → planning용 (cond, mem, mem_mask) 반환.
#   ⚠️ VLM forward 2회(전방3뷰 1 + 게이팅뷰 1). 진짜 배포 파이프라인과 동일(전방판단→뷰확장→planning).
#   ⚠️ 학습은 게이팅된 cond로 게이트 CE를 걸므로(속도 우선), Pass1(전방3뷰)와 입력 분포가 약간 다르다(train/infer
#      mismatch) — 게이트 정확도가 낮으면 이 지점을 재검토(학습도 전방3뷰 cond로 게이트 CE).
#   반환: (cond(1,Dc) on device, mem, mem_mask, pred_dir(0/1/2), used_views(list[str])).
@torch.no_grad()
def gate_closedloop_encode(vlm, gate_mod, processor, rec, prompt, temporal_on, device):
    # 1) 게이트 예측: **전방 3뷰**(GATE_ALWAYS)만으로 cond 얻어 방향 예측
    front_imgs = gate_views_by_direction(rec["images"], 0)   # 0=straight → 전방 3뷰만
    front_hist = gate_views_by_direction(history_for(rec, temporal_on), 0)
    cvec0, mem0, mm0 = encode_condition(vlm, processor, front_imgs, prompt, front_hist)
    cond0 = cvec0.unsqueeze(0).to(device)
    pred_dir = int(gate_mod(cond0).argmax(-1).item())      # 0=straight/1=left/2=right
    # 2) 예측 방향으로 뷰 게이팅. **straight(0)면 Pass1(전방3뷰)과 동일** → 재-encode 생략(74% 직진에서 큰 절감).
    if pred_dir == 0:
        cond, mem, mem_mask = cond0, mem0.to(device), mm0.to(device)
        used_views = sorted(im["view"] for im in front_imgs)
    else:
        imgs = gate_views_by_direction(rec["images"], pred_dir)
        hist = gate_views_by_direction(history_for(rec, temporal_on), pred_dir)
        cvec, mem, mem_mask = encode_condition(vlm, processor, imgs, prompt, hist)
        cond = cvec.unsqueeze(0).to(device); mem = mem.to(device); mem_mask = mem_mask.to(device)
        used_views = sorted(im["view"] for im in imgs)
    return cond, mem, mem_mask, pred_dir, used_views


# clip_closedloop_iter: temporal-clip 추론(진짜 폐루프, **Pass 구분 없음**). clip들을 시간 순서로 롤아웃하며
#   각 프레임에서 (rec, cond, mem, mem_mask, pred_dir)를 yield한다. 프레임 t의 뷰는 **이전 프레임 게이트 예측
#   dir(t-1)**로 게이팅(selective_view+gate), 첫 프레임은 전 8뷰. cond를 만든 그 게이트가 dir(t)를 예측해
#   다음 프레임으로 전달 → VLM forward가 프레임당 1회(학습과 동일 구조). gate=None이면 게이팅 없이 전 8뷰.
@torch.no_grad()
def clip_closedloop_iter(vlm, gate, processor, clips, prompt_tpl, temporal_on, device, selective_view):
    for clip_recs in clips:
        prev_dir = None                                    # clip 시작: 과거 예측 없음 → 전 8뷰
        for rec in clip_recs:
            prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
            if selective_view and gate is not None:
                imgs = gate_views_by_direction(rec["images"], prev_dir)
                hist = gate_views_by_direction(history_for(rec, temporal_on), prev_dir)
            else:
                imgs = rec["images"]; hist = history_for(rec, temporal_on)
            cvec, mem, mem_mask = encode_condition(vlm, processor, imgs, prompt, hist)
            cond = cvec.unsqueeze(0).to(device); mem = mem.to(device); mem_mask = mem_mask.to(device)
            pred_dir = None
            if gate is not None:
                pred_dir = int(gate(cond).argmax(-1).item())   # dir(t) 예측 → 다음 프레임 뷰용
                prev_dir = pred_dir
            yield rec, cond, mem, mem_mask, pred_dir


# ego_vec: SFT 레코드의 ego_state([vx,vy,ax,ay] ego-frame)를 길이 ego_dim 리스트로(없으면 0). DiT의
#   ego 조건화 경로 입력. 단일 프레임 이미지엔 속도 정보가 없어 이 명시적 운동상태가 planning의 핵심 입력.
def ego_vec(rec: dict, ego_dim: int) -> list:
    e = rec.get("ego_state") or []
    if len(e) >= ego_dim:
        return [float(x) for x in e[:ego_dim]]
    return [float(x) for x in e] + [0.0] * (ego_dim - len(e))


# _build_views_content: (과거 8뷰 옵션 +) 현재 8뷰를 시간·뷰 결합 태그와 인터리브한 user content +
#   PIL 이미지 리스트 반환(공용). history_recs가 있으면 과거 1 timestep(8뷰, t=-1)을 **현재(t=0) 앞**에
#   둔다(논문 nuVLA의 시간 맥락 — 이미지 토큰이 8→16장으로 늘어 메모리 ~2배). 없으면 단일 프레임(현재만).
#   시간·뷰는 _append_views(t=…)가 이미지별 태그로 붙이므로 별도 블록 마커는 두지 않는다(논문 포맷 충실).
def _build_views_content(image_recs: list, prompt: str, history_recs: list | None = None):
    content: list = []
    images: list = []
    if history_recs:                                       # 과거 1 timestep(8뷰): t=-1
        _append_views(content, images, history_recs, t=-1)
    _append_views(content, images, image_recs, t=0)        # 현재 프레임(항상): t=0
    content.append({"type": "text", "text": prompt})
    return content, images


# build_model_inputs: **워커 오프로드 대상**인 "load + preprocess"만 수행해 CPU 텐서(BatchEncoding)를
#   돌려준다(디바이스 이동·forward는 호출측=메인 프로세스 GPU). DataLoader 워커(TrajTrainSet.__getitem__)와
#   평가 경로(encode_and_lm_loss)가 **같은 토큰화 규칙**(plen 범위·label 마스킹)을 쓰도록 하는 단일 출처.
#     - plen = user+"assistant\n"까지 토큰 수(이미지 pad 확장 포함이라 images 필수) → condition 평균 범위.
#     - reasoning 있으면 full 시퀀스 인코딩 + labels(프롬프트 구간 -100 마스킹), 없으면 프롬프트만·labels=None.
#   반환: (enc: BatchEncoding[batch=1, CPU], labels: LongTensor|None, plen: int).
def build_model_inputs(processor, image_recs: list, prompt: str, reasoning: str | None,
                       history_recs: list | None = None):
    content, images = _build_views_content(image_recs, prompt, history_recs)   # LOAD: 디스크→PIL
    user_msg = {"role": "user", "content": content}
    prompt_text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    plen = processor(text=[prompt_text], images=images, return_tensors="pt")["input_ids"].shape[1]
    if reasoning:                                          # full 시퀀스(프롬프트+reasoning 타깃)
        msgs = [user_msg, {"role": "assistant", "content": reasoning}]
        full_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = processor(text=[full_text], images=images, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[:, :plen] = -100                            # 프롬프트 구간 마스킹 → reasoning 토큰만 loss
    else:                                                  # 프롬프트만(생성 프롬프트)
        enc = processor(text=[prompt_text], images=images, return_tensors="pt")
        labels = None
    return enc, labels, plen


# encode_and_lm_loss: build_model_inputs(load+preprocess)로 얻은 CPU 입력을 디바이스로 올려 한 forward로
#   (condition, LM_loss)를 동시에 얻는 **단일 샘플 동기 경로**(평가·probe·cond-fit용; 학습 핫루프는
#   DataLoader+forward_batch를 쓴다). condition = 프롬프트까지 hidden mean-pool(causal이라 reasoning
#   유무와 무관 → 평가와 일치). ⚠️ grad 제어는 호출측. frozen이면 LM_loss가 VLM에 backprop 안 돼 무의미.
def encode_and_lm_loss(model, processor, image_recs: list, prompt: str, reasoning: str | None,
                       history_recs: list | None = None):
    enc, labels, plen = build_model_inputs(processor, image_recs, prompt, reasoning, history_recs)
    inputs = enc.to(model.device)
    if labels is not None:
        labels = labels.to(model.device)
        out = model(**inputs, labels=labels, output_hidden_states=True)
        lm_loss = out.loss
    else:
        out = model(**inputs, output_hidden_states=True)
        lm_loss = None
    hs = out.hidden_states[-1]                             # (1, seq, hidden)
    cmask = (inputs["attention_mask"] > 0).clone()
    cmask[:, plen:] = False                                # 프롬프트까지만(응답/reasoning 토큰 제외 → KV·평균 공통)
    m = cmask.unsqueeze(-1).to(hs.dtype)
    cond = (hs * m).sum(1) / m.sum(1).clamp_min(1.0)       # (1, hidden)
    # 반환에 mem 시퀀스(1,seq,hidden)+mem_mask(1,seq) 추가(cross_attn용). cond은 기존대로 squeeze.
    return cond.squeeze(0).float(), lm_loss, hs, cmask


# ─── DataLoader 프리페치 경로(학습 핫루프) ──────────────────────────────────────────────────
# 프로파일상 한 step의 ~58%가 데이터 준비(load 24% + preprocess 34%)였고 GPU(forward+backward)는 ~38%다.
# 그 데이터 준비를 DataLoader 워커(별도 CPU 프로세스 num_workers개)로 옮겨 **다음 배치**를 미리 만들면
# (prefetch), 메인 GPU가 **현재 배치**를 forward/backward하는 시간과 겹쳐 step time이 순차합에서
# max(데이터준비, GPU)로 줄어든다. VLM forward는 GPU라 메인에 남지만(정상), 넘길 건 CPU 병목뿐이다.
#   경계선: TrajTrainSet.__getitem__(워커)=build_model_inputs로 CPU 텐서까지 → collate_traj로 배치 패딩
#           → forward_batch(메인)=디바이스 이동 + VLM forward + condition/LM_loss.
class TrajTrainSet:
    """레코드+processor를 들고 __getitem__에서 load+preprocess(build_model_inputs)까지 수행해 CPU 텐서
    dict를 돌려주는 map-style 데이터셋(DataLoader 워커에서 실행). torch.utils.data.Dataset을 상속하지
    않아도 __len__/__getitem__만 있으면 DataLoader가 받는다."""
    def __init__(self, records, processor, prompt_tpl, temporal_on, reasoning_types, joint, ego_dim,
                 maneuver_thr=-1.0, gate_teacher_forcing=False):
        self.records = records
        self.processor = processor
        self.prompt_tpl = prompt_tpl
        self.temporal_on = temporal_on
        self.reasoning_types = reasoning_types
        self.joint = joint
        self.ego_dim = ego_dim
        self.maneuver_thr = maneuver_thr                   # selective-view 게이팅 임계값(<0=off, 전 8뷰)
        # gate_teacher_forcing=True(--gate-weight>0): maneuver_thr(미래 GT) 대신 gate_direction GT(과거 결정
        #   이월값)로 뷰 게이팅. 방법 A — "완벽한 과거 게이트 예측"으로 뷰 결정(teacher-forcing). 추론 땐 게이트
        #   실제 예측을 씀(진짜 폐루프). 이 플래그가 True면 maneuver_thr 게이팅은 무시된다.
        self.gate_teacher_forcing = gate_teacher_forcing

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        prompt = self.prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
        reasoning = reasoning_target(rec, self.reasoning_types) if self.joint else None  # 선택 종류 합친 LM 타깃
        if self.gate_teacher_forcing:                       # 폐루프 학습: gate_direction GT로 뷰 게이팅
            gdir = rec_gate_direction(rec)                  # 0/1/2/None(첫 구간→전뷰)
            imgs = gate_views_by_direction(rec["images"], gdir)
            hist = gate_views_by_direction(history_for(rec, self.temporal_on), gdir)
        else:                                               # 구형: maneuver_lateral(미래 GT)+thr 게이팅
            ml = rec_maneuver(rec)                          # 기동 신호(궤적 유도)
            imgs = gate_views(rec["images"], ml, self.maneuver_thr)              # 현재뷰 게이팅
            hist = gate_views(history_for(rec, self.temporal_on), ml, self.maneuver_thr)  # 과거뷰도 동일 게이팅
        enc, labels, plen = build_model_inputs(self.processor, imgs, prompt, reasoning, hist)
        ego = torch.tensor(ego_vec(rec, self.ego_dim), dtype=torch.float32) if self.ego_dim > 0 else None
        return {
            "input_ids": enc["input_ids"][0],              # (seq,)
            "attention_mask": enc["attention_mask"][0],    # (seq,)
            "mm_token_type_ids": enc["mm_token_type_ids"][0],  # (seq,) Qwen3-VL M-RoPE 필수(멀티모달 토큰 표시)
            "pixel_values": enc["pixel_values"],           # (patches, feat) — 배치 시 concat
            "image_grid_thw": enc["image_grid_thw"],       # (n_img, 3) — 배치 시 concat
            "labels": (labels[0] if labels is not None else None),   # (seq,) or None
            "plen": int(plen),
            "waypoints": torch.tensor(rec["output"]["waypoints"], dtype=torch.float32),   # (N,2)
            "ego": ego,                                    # (ego_dim,) or None
            # selective-view 게이트 타깃(0=straight/1=left/2=right). 과거 결정 이월값(build_sft) — 없으면
            # None(구버전 데이터 하위호환) → collate가 배치에 하나도 없으면 gate_CE 자체를 생략.
            "gate_direction": rec["output"].get("gate_direction"),
        }


# collate_traj: 가변 길이 시퀀스를 자동 패딩해 배치를 만든다(batch>1 지원). Qwen-VL 특성상
#   - input_ids/attention_mask/labels: 배치 최대 길이로 **오른쪽 패딩**(causal+right-pad라 프롬프트 hidden이
#     불변 → condition이 단일 샘플과 동일; labels는 -100 패딩이라 loss에 무영향). train_text_sft 콜레이터와 동일 규칙.
#   - pixel_values/image_grid_thw: 이미지 패치는 **concat**(모델이 image_grid_thw로 되쪼갬).
#   - labels: 배치에 reasoning 샘플이 하나도 없으면 None(→ forward에서 LM_loss 생략).
def collate_traj(batch, pad_id):
    B = len(batch)
    maxlen = max(b["input_ids"].shape[0] for b in batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, maxlen), dtype=torch.long)
    mm_token_type_ids = torch.zeros((B, maxlen), dtype=torch.long)   # 패딩 위치=0(텍스트 타입)
    any_labels = any(b["labels"] is not None for b in batch)
    labels = torch.full((B, maxlen), -100, dtype=torch.long) if any_labels else None
    plen = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].shape[0]
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        mm_token_type_ids[i, :L] = b["mm_token_type_ids"]
        plen[i] = b["plen"]
        if any_labels and b["labels"] is not None:
            labels[i, :L] = b["labels"]
    pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
    image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    waypoints = torch.stack([b["waypoints"] for b in batch], dim=0)
    ego = torch.stack([b["ego"] for b in batch], dim=0) if batch[0]["ego"] is not None else None
    # gate_direction: 배치 내 값 있는 샘플만 (idx, label) 형태로 골라 반환 → forward_batch가 그 idx의
    #   cond만 뽑아 gate_CE 계산(레코드 일부에 없어도 배치 전체가 깨지지 않음).
    gate_idx, gate_lbl = [], []
    for i, b in enumerate(batch):
        if b.get("gate_direction") is not None:
            gate_idx.append(i); gate_lbl.append(b["gate_direction"])
    gate_direction = (torch.tensor(gate_idx, dtype=torch.long),
                      torch.tensor(gate_lbl, dtype=torch.long)) if gate_idx else None
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
            "labels": labels, "plen": plen, "waypoints": waypoints, "ego": ego,
            "gate_direction": gate_direction}


# forward_batch: **메인 프로세스 GPU**에서 도는 부분 — collate된 CPU 배치를 디바이스로 올려 VLM forward,
#   프롬프트 구간(plen)까지 hidden mean-pool로 per-sample condition + (labels 있으면) 배치 LM_loss.
#   반환: (cond (B,Dc) fp32, lm_loss 스칼라 or None, lm_dummy 스칼라 or None). grad: train_vlm이면 흐르고 frozen이면 no_grad.
def forward_batch(model, batch, device, train_vlm):
    inputs = {k: batch[k].to(device, non_blocking=True)
              for k in ("input_ids", "attention_mask", "mm_token_type_ids", "pixel_values", "image_grid_thw")}
    labels = batch["labels"].to(device, non_blocking=True) if batch["labels"] is not None else None
    ctx = nullcontext() if train_vlm else torch.no_grad()
    with ctx:
        if labels is not None:
            out = model(**inputs, labels=labels, output_hidden_states=True)
        else:
            out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[-1]                             # (B, seq, hidden)
    plen = batch["plen"].to(device)                        # (B,)
    pos = torch.arange(hs.shape[1], device=device).unsqueeze(0)          # (1, seq)
    cmask = (inputs["attention_mask"] > 0) & (pos < plen.unsqueeze(1))   # 프롬프트 토큰만 True(per-sample plen)
    m = cmask.unsqueeze(-1).to(hs.dtype)
    cond = (hs * m).sum(1) / m.sum(1).clamp_min(1.0)       # (B, hidden)
    # mem 시퀀스(hs)+mem_mask(cmask)를 함께 반환 → cross_attn DiT가 프롬프트 토큰(KV)에 cross-attention.
    #   cmask가 응답/패딩 토큰을 False로 마스킹하므로 KV는 image+prompt 토큰만(per-sample plen).
    lm_loss = out.loss if labels is not None else None
    # 0-더미(표준 DDP용): train_vlm이면 lm_head 출력(logits)을 매 step 손실 그래프에 0계수로 연결한다.
    #   reasoning 없는 step에도 lm_head가 항상 grad(=0)를 받아 rank 간 "grad 받는 param 집합"이 고정됨
    #   → find_unused_parameters=False로 표준 DDP가 데드락 없이 동작. 손실 값은 +0(학습 수치 무변경).
    #   ⚠️ mean()이라 bf16 합 overflow(→0*inf=nan) 위험 없음; float()로 상향; **곱셈**이어야(detach 상수 대체 금지)
    #      lm_head가 그래프에 실제 연결된다.
    lm_dummy = (0.0 * out.logits.float().mean()) if train_vlm else None
    return cond.float(), hs, cmask, lm_loss, lm_dummy


# TrajReasVLA: VLM 백본 + TrajectoryDiT 궤적 헤드를 **한 nn.Module**로 묶는 래퍼(표준 DDP 진입점).
#   forward(batch)가 collate된 배치를 받아 한 곳에서 (total_loss, flow, lm_loss)를 계산한다 → 이 단일
#   forward를 DDP로 감싸 grad를 자동 동기화한다(0-더미로 lm_head 조건부 사용 문제 회피 — forward_batch 주석).
#   구성요소 self.vlm/self.dit는 기존 vlm_mod/dit_mod 별칭이 그대로 가리켜 저장·평가·probe·optimizer는 무변경.
#   ⚠️ self.normalizer는 nn.Module이 아니라 mean/std 텐서만 가진 헬퍼(forward에서 read-only) — DDP 동기화 대상
#      아님이 맞다(전 rank가 같은 데이터로 fit해 동일). model.train()/eval()은 호출하지 않는다(vlm의 mode를
#      load_vlm 설정대로, dit는 별도 dit.train()으로 유지 — frozen vlm에 dropout이 켜지는 것 방지).
#   gate(GateHead|None): selective-view 1단계 게이트(3-class, cond→direction). gate_weight=μ로 total에 합류
#     (total = flow + λ·LM + μ·gate_CE). alpha=클래스 가중(불균형 보정), focal_gamma=focal 감쇠 지수(0=순수
#     weighted CE). 배치에 gate 라벨이 하나도 없는 step에도 0-더미로 head를 그래프에 연결(표준 DDP 안전).
class TrajReasVLA(nn.Module):
    def __init__(self, vlm, dit, normalizer, reas_weight, train_vlm, device,
                 gate=None, gate_weight=1.0, gate_alpha=None, gate_focal_gamma=2.0):
        super().__init__()
        self.vlm = vlm
        self.dit = dit
        self.normalizer = normalizer
        self.reas_weight = reas_weight
        self.train_vlm = train_vlm
        self.device = device
        self.gate = gate
        self.gate_weight = gate_weight
        # register_buffer로 등록해야 DDP(model, device_ids=[...])가 모델을 이 rank GPU로 옮길 때 함께
        # 이동한다(plain attribute는 자동 전파 안 돼 "cuda:1 vs cpu" 디바이스 불일치로 rank>0에서 죽는다).
        if gate_alpha is not None:
            self.register_buffer("gate_alpha", gate_alpha.to(device))
        else:
            self.gate_alpha = None
        self.gate_focal_gamma = gate_focal_gamma

    def forward(self, batch):
        cond, mem, mem_mask, lm_loss, lm_dummy = forward_batch(self.vlm, batch, self.device, self.train_vlm)
        ego = batch["ego"].to(self.device) if batch["ego"] is not None else None      # (B, ego_dim)
        x1 = self.normalizer.normalize(batch["waypoints"].to(self.device))            # (B,N,2) 정규화 공간
        flow = self.dit.flow_loss(x1, cond, ego, mem=mem, mem_mask=mem_mask)          # cross_attn이면 mem 사용
        loss = flow + (self.reas_weight * lm_loss if lm_loss is not None else 0.0)     # 공동 손실 total = flow + λ·LM
        if lm_dummy is not None:
            loss = loss + lm_dummy                # +0: lm_head를 그래프에 연결(표준 DDP static param 집합)
        gate_ce = None
        if self.gate is not None:
            gd = batch.get("gate_direction")
            logits_all = self.gate(cond)           # (B,3) — 전 샘플에 forward(뒤에서 0-더미로도 재사용)
            if gd is not None:
                idx, lbl = gd
                idx = idx.to(self.device); lbl = lbl.to(self.device)
                gate_ce = focal_loss(logits_all[idx], lbl, alpha=self.gate_alpha, gamma=self.gate_focal_gamma)
                loss = loss + self.gate_weight * gate_ce
            else:                                  # 이 step에 gate 라벨 0개 → 0-더미로 head를 그래프에 유지
                loss = loss + 0.0 * logits_all.float().mean()
        # flow/lm_loss/gate_ce는 로깅 전용 → detach해 반환(DDP가 loss 외 별도 grad-root로 오인하지 않게).
        return (loss, flow.detach(), (lm_loss.detach() if lm_loss is not None else None),
                (gate_ce.detach() if gate_ce is not None else None))


# ─── clip 시퀀스 롤아웃(temporal-clip 경로) ─────────────────────────────────────────────
# clip_rollout_forward: 한 clip의 프레임을 **시간 순서로** 롤아웃. **프레임마다 즉시 backward**해
#   그래프를 해제한다 — 뷰 게이팅 방향은 .item()/GT라 프레임 간 grad 연결이 없으므로(각 프레임 손실이 독립
#   그래프) 안전하고, clip 전체 그래프를 동시 보유하지 않아 메모리 효율적(13프레임 동시 보유 시 OOM 회피).
#   뷰 게이팅(selective_view일 때만; 아니면 항상 전 8뷰) — forcing으로 **게이팅 방향의 출처**가 갈린다:
#     • student(폐루프): 프레임 t 뷰 = **이전 프레임 게이트 예측 dir(t-1)**(argmax, detach). 추론과 동일 분포
#       (exposure-bias 없음)지만, 게이트가 나쁘면 틀린 뷰로 학습돼 붕괴가 자기강화될 수 있다. 첫 프레임=전 8뷰.
#     • teacher: 프레임 t 뷰 = **그 프레임의 GT gate_direction**(rec_gate_direction). 프레임 간 의존성 없음
#       (입력 독립) → 게이트/planning이 항상 올바른 뷰로 학습돼 안정적. GT 없는 프레임(예: 0~49)은 전 8뷰.
#   각 프레임 t: 뷰 결정 → VLM forward(grad는 train_vlm이면 흐름) → cond/mem → flow + λ·LM + μ·gate_CE(항상
#     현재 프레임 gate_direction GT로 지도) → backward(로컬 누적, sync는 호출측이 clip 경계에서 ddp.sync_grads).
#     ⚠️ gate_CE 지도는 두 forcing 모두 GT 대비 동일; forcing은 오직 **뷰 게이팅 방향의 출처**만 바꾼다.
#   backward 스케일 = grad_scale(=1/grad_accum). clip은 프레임 손실 **합**으로 기여(≈프레임 수만큼 큰 배치).
#   반환: stats(로깅용, grad는 이미 파라미터 .grad에 누적됨).
def clip_rollout_forward(vlm, dit, gate, normalizer, processor, prompt_tpl, clip_recs, device, grad_scale,
                         *, train_vlm, joint, reasoning_types, ego_dim, temporal_on,
                         reas_weight, gate_weight, gate_alpha, gate_focal_gamma, selective_view,
                         forcing="student"):
    import torch

    if gate_alpha is not None:
        gate_alpha = gate_alpha.to(device)     # 원본은 CPU(rank 공유) → 이 rank GPU로(focal_loss device 일치)
    prev_dir = None                                        # 첫 프레임: 과거 게이트 예측 없음 → 전 8뷰
    n = 0
    flow_sum = lm_sum = gate_sum = 0.0
    lm_n = gate_n = 0
    ctx = nullcontext() if train_vlm else torch.no_grad()
    for rec in clip_recs:
        prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
        reasoning = reasoning_target(rec, reasoning_types) if joint else None
        # 뷰 게이팅 방향: student=prev_dir(t-1 예측), teacher=현재 프레임 GT gate_direction. 아니면 전 8뷰.
        #   history도 동일 방향으로 게이팅. teacher는 GT가 None(예: 0~49)이면 gate_views_by_direction이 전 8뷰.
        if selective_view:
            gate_dir = rec_gate_direction(rec) if forcing == "teacher" else prev_dir
            imgs = gate_views_by_direction(rec["images"], gate_dir)      # None→8뷰
            hist = gate_views_by_direction(history_for(rec, temporal_on), gate_dir)
        else:
            imgs = rec["images"]
            hist = history_for(rec, temporal_on)
        with ctx:
            cvec, lm_loss, hs, cmask = encode_and_lm_loss(vlm, processor, imgs, prompt, reasoning, hist)
        cond = cvec.unsqueeze(0).to(device)                # (1, hidden)
        ego = torch.tensor([ego_vec(rec, ego_dim)], dtype=torch.float32, device=device) if ego_dim > 0 else None
        x1 = normalizer.normalize(torch.tensor([rec["output"]["waypoints"]], dtype=torch.float32, device=device))
        flow = dit.flow_loss(x1, cond, ego, mem=hs.to(device), mem_mask=cmask.to(device))
        frame_loss = flow
        flow_sum += float(flow); n += 1
        if lm_loss is not None:
            frame_loss = frame_loss + reas_weight * lm_loss
            lm_sum += float(lm_loss); lm_n += 1
        # 게이트: 현재 프레임 gate_direction GT로 focal 지도 + 다음 프레임 뷰용 dir 예측
        if gate is not None:
            logits = gate(cond)                            # (1,3)
            gd = rec_gate_direction(rec)
            if gate_weight > 0 and gd is not None:
                gce = focal_loss(logits, torch.tensor([gd], dtype=torch.long, device=device),
                                 alpha=gate_alpha, gamma=gate_focal_gamma)
                frame_loss = frame_loss + gate_weight * gce
                gate_sum += float(gce); gate_n += 1
            else:                                          # gate 라벨 없는 프레임: 0-더미로 head 그래프 유지
                frame_loss = frame_loss + 0.0 * logits.float().mean()
            prev_dir = int(logits.argmax(-1).item())       # 다음 프레임 뷰 게이팅용(폐루프, item()=detach)
        (frame_loss * grad_scale).backward()               # 프레임별 즉시 backward → 그래프 해제(메모리 안전)
    stats = {"n": n, "flow_sum": flow_sum, "lm_sum": lm_sum, "lm_n": lm_n,
             "gate_sum": gate_sum, "gate_n": gate_n}
    return stats


# evaluate: val의 trajectory 샘플에 대해 학습과 동일 손실(flow + λ·LM)을 grad 없이 계산해 평균 반환.
#   학습 중 epoch 종합 지표용(가벼운 val loss). ADE/FDE 같은 궤적 지표는 eval_trajectory.py로 별도.
#   **분산 eval**: val을 rank별로 샤딩해 병렬 forward 후, 부분합/부분개수를 all_reduce(SUM)해 전 rank
#   동일 평균을 복원한다(rank0 단독 직렬보다 ~world_size배 빠르고, 모든 rank가 동일 collective를 호출해
#   데드락 없음). 반환: {"val_loss","val_flow","val_lm"} (전 rank reasoning 샘플 0이면 val_lm 생략).
def evaluate(vlm, dit_mod, processor, val_records, prompt_tpl, normalizer, reas_weight, device, joint,
             temporal_on=False, reasoning_types=(), maneuver_thr=-1.0,
             gate_mod=None, gate_weight=0.0, gate_alpha=None, gate_focal_gamma=2.0,
             temporal_clip=False, selective_view=False, forcing="student", keyframe_eval=False,
             keyframe_select=(0, 1, 2)):
    import torch

    was_training = dit_mod.training
    # 채점 대상 keyframe id 집합(clip당 decision 프레임 중 select된 순서만). keyframe_eval off면 미사용.
    kf_ids = select_keyframe_ids(val_records, keyframe_select) if keyframe_eval else None
    dit_mod.eval()
    gate_was_training = gate_mod.training if gate_mod is not None else None
    if gate_mod is not None:
        gate_mod.eval()
    if gate_alpha is not None:
        gate_alpha = gate_alpha.to(device)     # 각 rank가 자기 GPU로(원본은 CPU에 둬 rank 간 공유 텐서 재사용)
    flow_sum = tot_sum = lm_sum = 0.0
    n = lm_n = 0
    gate_ce_sum = gate_n = 0.0
    gate_correct = 0.0
    # 클래스별 (correct, total) — 안전 지표: left/right(측면 뷰 필요) 재현율을 특히 확인.
    gate_cls_correct = {0: 0.0, 1: 0.0, 2: 0.0}
    gate_cls_total = {0: 0.0, 1: 0.0, 2: 0.0}

    # _accum_loss: 한 프레임의 (cond, lml)로 flow/lm/gate_ce **손실만** 집계(val_loss = 학습과 동일 게이팅
    #   기준이어야 학습 신호와 일관). gate 예측(pred_dir)은 반환하되 정확도 집계는 여기서 하지 않는다
    #   — teacher forcing이면 cond가 GT로 게이팅돼 pred_dir 예측 자체가 leakage(자기 게이팅을 되읽음)라
    #   정확도 지표로 쓰면 안 되기 때문. 정확도는 항상 별도의 closed-loop cond로 _accum_gate_metrics가 잰다.
    def _accum_loss(rec, cond, mem, mem_mask, lml):
        nonlocal flow_sum, tot_sum, lm_sum, n, lm_n, gate_ce_sum, gate_n
        ego = torch.tensor([ego_vec(rec, dit_mod.ego_dim)], dtype=torch.float32, device=device) \
            if dit_mod.ego_dim > 0 else None
        x1 = normalizer.normalize(torch.tensor([rec["output"]["waypoints"]], dtype=torch.float32, device=device))
        flow = float(dit_mod.flow_loss(x1, cond, ego, mem=mem, mem_mask=mem_mask))
        tot = flow + reas_weight * float(lml) if lml is not None else flow
        pred_dir = None
        if gate_mod is not None:
            logits = gate_mod(cond)
            pred_dir = int(logits.argmax(-1).item())
            gd = rec["output"].get("gate_direction")
            if gd is not None:
                ce = float(focal_loss(logits, torch.tensor([gd], device=device),
                                      alpha=gate_alpha, gamma=gate_focal_gamma))
                tot = tot + gate_weight * ce
                gate_ce_sum += ce; gate_n += 1
        flow_sum += flow; tot_sum += tot; n += 1
        if lml is not None:
            lm_sum += float(lml); lm_n += 1
        return pred_dir

    # _accum_gate_metrics: **항상 closed-loop(예측 게이팅) cond**의 게이트 예측으로 acc/recall만 집계.
    #   forcing과 무관 — teacher든 student든 이 cond는 예측 방향으로 게이팅돼 GT 누출이 없다(inference 진실).
    def _accum_gate_metrics(rec, cond):
        nonlocal gate_correct
        if gate_mod is None:
            return None
        logits = gate_mod(cond)
        pred_dir = int(logits.argmax(-1).item())
        gd = rec["output"].get("gate_direction")
        if gd is not None:
            gate_cls_total[gd] += 1
            if pred_dir == gd:
                gate_correct += 1; gate_cls_correct[gd] += 1
        return pred_dir

    with torch.no_grad():
        if temporal_clip:
            # clip 순차 폐루프 val(학습과 동일 구조): clip을 rank별 샤딩, dir(t-1) 예측으로 뷰 게이팅.
            clips = group_records_by_clip(val_records)
            my_clip_idx = ddp.shard(list(range(len(clips))))
            for ci in my_clip_idx:
                prev_dir = None
                for rec in clips[ci]:
                    # keyframe_eval: 전 프레임을 롤아웃(prev_dir 폐루프 유지)하되 **선택된 keyframe만 채점**(논문 정렬).
                    kf = (not keyframe_eval) or (rec["id"] in kf_ids)
                    prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
                    reasoning = reasoning_target(rec, reasoning_types) if joint else None
                    # val_loss는 학습과 동일 게이팅: student=prev_dir(폐루프), teacher=현재 GT.
                    if selective_view and gate_mod is not None:
                        gate_dir = rec_gate_direction(rec) if forcing == "teacher" else prev_dir
                        imgs = gate_views_by_direction(rec["images"], gate_dir)
                        hist = gate_views_by_direction(history_for(rec, temporal_on), gate_dir)
                    else:
                        imgs = rec["images"]; hist = history_for(rec, temporal_on)
                    c, lml, mem, mem_mask = encode_and_lm_loss(vlm, processor, imgs, prompt, reasoning, hist)
                    cond = c.unsqueeze(0).to(device)
                    if kf:                                 # 손실은 keyframe만 집계
                        _accum_loss(rec, cond, mem.to(device), mem_mask.to(device), lml)
                    # 게이트: prev_dir 갱신은 **매 프레임**(폐루프), 정확도 지표는 keyframe만. 정확도는 forcing과
                    #   무관하게 항상 closed-loop(예측 게이팅) cond로 측정(leakage 방지) — teacher면 prev_dir로 재-encode.
                    if gate_mod is not None:
                        if forcing == "teacher" and selective_view:
                            imgs_cl = gate_views_by_direction(rec["images"], prev_dir)
                            hist_cl = gate_views_by_direction(history_for(rec, temporal_on), prev_dir)
                            c_cl, _, _, _ = encode_and_lm_loss(vlm, processor, imgs_cl, prompt, None, hist_cl)
                            cond_cl = c_cl.unsqueeze(0).to(device)
                        else:
                            cond_cl = cond
                        pd = _accum_gate_metrics(rec, cond_cl) if kf \
                            else int(gate_mod(cond_cl).argmax(-1).item())
                        prev_dir = pd                      # 다음 프레임 closed-loop 게이팅용(매 프레임 갱신)
        else:
            # 프레임 독립 경로: keyframe_eval이면 선택된 keyframe만 남겨 샤딩(rank 부하 균형 + 채점 대상 일치).
            idxs = [i for i in range(len(val_records))
                    if (not keyframe_eval) or (val_records[i]["id"] in kf_ids)]
            my = ddp.shard(idxs)                              # 이 rank가 맡을 val 인덱스(프레임 독립)
            for i in my:
                rec = val_records[i]
                prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
                reasoning = reasoning_target(rec, reasoning_types) if joint else None
                ml = rec_maneuver(rec)                       # 학습과 동일(구형) 게이팅으로 val_loss 측정
                imgs = gate_views(rec["images"], ml, maneuver_thr)
                c, lml, mem, mem_mask = encode_and_lm_loss(vlm, processor, imgs, prompt, reasoning,
                                                           gate_views(history_for(rec, temporal_on), ml, maneuver_thr))
                cond = c.unsqueeze(0).to(device)
                _accum_loss(rec, cond, mem.to(device), mem_mask.to(device), lml)
                _accum_gate_metrics(rec, cond)     # (구형 경로) GT-direction 게이팅이 아니라 leakage 없음
    if was_training:
        dit_mod.train()
    if gate_mod is not None and gate_was_training:
        gate_mod.train()
    # 전 rank 부분합/개수 집계(한 번의 collective). 모든 rank가 호출 → 동기화 보장.
    reduce_vals = [flow_sum, tot_sum, lm_sum, n, lm_n, gate_ce_sum, gate_n, gate_correct,
                  gate_cls_correct[0], gate_cls_correct[1], gate_cls_correct[2],
                  gate_cls_total[0], gate_cls_total[1], gate_cls_total[2]]
    (g_flow, g_tot, g_lm, g_n, g_lm_n, g_gate_ce, g_gate_n, g_gate_correct,
     g_c0, g_c1, g_c2, g_t0, g_t1, g_t2) = ddp.all_reduce_sum(reduce_vals)
    if g_n == 0:                                          # val에 trajectory 샘플이 전혀 없으면
        return None
    out = {"val_loss": g_tot / g_n, "val_flow": g_flow / g_n}
    if g_lm_n > 0:
        out["val_lm"] = g_lm / g_lm_n
    if g_gate_n > 0:
        out["val_gate_ce"] = g_gate_ce / g_gate_n          # 손실(_accum_loss, 학습과 동일 게이팅)
        # 정확도(_accum_gate_metrics, 항상 closed-loop 게이팅 — leakage 없음). g_gate_n과 분모가 같은 이유:
        # 두 함수 모두 프레임마다 정확히 1번씩, 같은 gate_direction is not None 조건으로 카운트하므로 항상 동수.
        out["val_gate_acc"] = g_gate_correct / g_gate_n
        # 안전 지표: left/right(측면 뷰 필요) 재현율 — 이게 낮으면 게이트가 방향 전환을 놓쳐 뷰를 못 킴.
        if g_t1 > 0:
            out["val_gate_recall_left"] = g_c1 / g_t1
        if g_t2 > 0:
            out["val_gate_recall_right"] = g_c2 / g_t2
    return out


# run_final_eval: 학습 종료 후 rank0이 부르는 최종 평가. 이미 메모리의 vlm_mod/dit_mod/normalizer/processor를
# 재사용(재로딩 없음)해 ① ADE/FDE(미터, eval_trajectory와 동일 계산) + ② BEV 시각화/8뷰/reasoning을
# run_dir에 저장한다. eval 스크립트의 재사용 함수(displacement/constant_velocity/plot_grid/...)를 import.
def run_final_eval(vlm_mod, dit_mod, processor, normalizer, cfg, prompt_tpl, device,
                   out_dir, val_path, ade_limit, tag="trajectory", viz_n=18, per_page=6, gate_mod=None):
    import numpy as np
    import torch

    from evaluation.eval_trajectory import constant_velocity, displacement_errors
    from evaluation.eval_qualitative import generate_reasoning, save_visuals, select_diverse
    from evaluation.planning_metrics import planning_scores, aggregate, format_table

    recs = [json.loads(l) for l in Path(val_path).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("task") == "trajectory"]
    if cfg.get("reasoning_only", False):                   # 학습과 동일 필터로 최종 planning 지표 산출(일관)
        recs, _rn0, _rn1 = filter_reasoning_only(recs)
    if not recs:
        return
    temporal_on = cfg.get("temporal", False)              # 학습이 과거뷰를 썼으면 평가도 동일하게(일관)
    ode_steps = cfg.get("ode_steps", 5)                   # 논문 K=5(flow-matching 소수스텝). cfg로 학습·평가 일관
    man_thr = cfg.get("maneuver_lateral_thr", -1.0)       # selective-view 게이팅(학습과 동일 thr로 평가)
    dit_mod.eval()
    vlm_was_train = vlm_mod.training
    vlm_mod.eval()

    # ① ADE/FDE + Table 3 planning 점수(NC/DA/EP/CF/HL/NPS): ade_limit개(0=전체) 샘플로 궤적 ODE
    #    샘플링 → GT와 미터 거리 + NPS 하위 점수. cv baseline 동반. 같은 pred로 둘 다 계산(일관).
    # 추론 속도 계측(eval_trajectory.py standalone과 동일 방식): 샘플당 VLM encode+DiT sample wall-clock(ms).
    #   cuda.synchronize()로 비동기 실행 보정, 첫 샘플(워밍업)은 통계 제외.
    import time
    infer_times_ms = []
    ade_recs = recs[:ade_limit] if ade_limit else recs
    rows, ades, fdes, cv_ades, cv_fdes, plan_rows = [], [], [], [], [], []
    gate_correct = 0; gate_total = 0                          # 폐루프 게이트 예측 정확도(gate_direction GT 대비)
    temporal_clip = cfg.get("temporal_clip", False)
    selective_view = cfg.get("selective_view", False)
    keyframe_eval = cfg.get("keyframe_eval", False)           # 논문 정렬: keyframe(decision, 0.2Hz)만 채점
    keyframe_select = tuple(cfg.get("keyframe_select", [1]))   # clip당 decision 3개 중 채점 순서(기본 [1]=정중앙)
    kf_ids = select_keyframe_ids(ade_recs, keyframe_select) if keyframe_eval else None

    # ── 속도 계측 원칙(편차·공정성 수정): "추론"만 계측한다.
    #   계측 대상 = VLM encode(게이팅된 뷰) + DiT sample. 이 둘이 실제 온라인 추론에서 실행되는 계산이며,
    #     selective-view의 뷰 개수(3 vs 8) 효과가 여기서 드러난다. 두 경로(temporal_clip/일반) 동일 기준.
    #   계측 제외 = planning_scores(BEV 충돌·맵 numpy CPU 계산) + denorm + ADE/FDE 집계. 이건 오프라인 평가지표
    #     계산이지 추론이 아니며, 장면 객체 수·맵 복잡도에 따라 크게 변동해 fastest/slowest 편차의 주범이었다.
    #   ⇒ _infer(계측 안, sample만) / _accum(계측 밖, scoring)로 분리하고, encode도 계측 구간 안으로 넣는다.

    def _infer(rec, cond, mem, mem_mask):
        ego = torch.tensor([ego_vec(rec, dit_mod.ego_dim)], dtype=torch.float32, device=device) \
            if dit_mod.ego_dim > 0 else None
        return dit_mod.sample(cond, steps=ode_steps, deterministic=True, ego=ego,
                              mem=mem, mem_mask=mem_mask)[0]

    # _accum: 계측 밖. denorm + ADE/FDE + planning 점수 집계(공통 tail). pred_dir 있으면 gate 정확도.
    def _accum(rec, pred_norm, pred_dir):
        nonlocal gate_correct, gate_total
        if pred_dir is not None:
            gd = rec_gate_direction(rec)
            if gd is not None:
                gate_total += 1; gate_correct += int(pred_dir == gd)
        pred_wp = normalizer.denormalize(pred_norm.cpu()).numpy()
        gt_wp = np.asarray(rec["output"]["waypoints"])
        pred = upsample_waypoints(pred_wp); gt = upsample_waypoints(gt_wp)   # (51,2) @10Hz
        ade, fde = displacement_errors(pred, gt)
        cv_ade, cv_fde = displacement_errors(constant_velocity(gt), gt)
        ades.append(ade); fdes.append(fde); cv_ades.append(cv_ade); cv_fdes.append(cv_fde)
        ps = planning_scores(rec, pred, gt)
        plan_rows.append(ps)
        rows.append({"id": rec["id"], "ade": round(ade, 3), "fde": round(fde, 3),
                     "cv_ade": round(cv_ade, 3), "cv_fde": round(cv_fde, 3),
                     **{k: round(v, 3) for k, v in ps.items()},
                     "pred_final": [round(float(x), 2) for x in pred[-1]],
                     "gt_final": [round(float(x), 2) for x in gt[-1]]})

    # ── 예열(warmup) 2-패스 계측: pass0=warmup(통계·집계 제외), pass1=측정.
    #   왜? encode 안(_append_views)에서 8뷰를 디스크에서 로드한다 → 콜드 캐시면 read가 수백 ms 튄다.
    #   특히 temporal_clip은 clip 단위 재정렬로 **clip 경계 첫 프레임마다** 콜드 read가 생겨 slowest가 튄다
    #   (실배포엔 없는 디스크 I/O·일회성 비용). 예열 패스가 페이지캐시·GPU allocator(3/6/8뷰 텐서 크기)·
    #   cuDNN autotune을 모두 데워서, 측정 패스는 순수 compute(encode+sample) tail만 남긴다 → 두 모드 공정 비교.
    #   비용: 최종평가 1회에 한해 encode+sample을 2배 수행(미니 val 규모라 수용 가능).
    with torch.no_grad():
        if temporal_clip:
            # 진짜 폐루프(Pass 구분 없음): clip 시간순 롤아웃, dir(t-1) 게이트 예측으로 프레임 t 뷰 게이팅.
            #   encode를 iterator 밖(여기 계측 구간)에서 직접 실행해야 뷰 개수 효과가 계측에 잡힌다.
            #   keyframe_eval: 전 프레임 롤아웃(prev_dir 폐루프 유지)하되 **keyframe만 계측·집계**. 비-keyframe은
            #   prev_dir 갱신용 encode+gate만(sample·계측 없음) → 논문식 "keyframe planning 평가".
            clips = group_records_by_clip(ade_recs)
            for warmup in (True, False):                        # pass0=예열(집계 제외), pass1=측정
                scored = 0                                     # 채점(keyframe) 프레임 수(첫 채점=워밍업 보수 제외)
                for clip_recs in clips:
                    prev_dir = None                            # clip 시작: 과거 예측 없음 → 전 8뷰
                    for rec in clip_recs:
                        prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
                        if selective_view and gate_mod is not None:
                            imgs = gate_views_by_direction(rec["images"], prev_dir)
                            hist = gate_views_by_direction(history_for(rec, temporal_on), prev_dir)
                        else:
                            imgs = rec["images"]; hist = history_for(rec, temporal_on)
                        if keyframe_eval and rec["id"] not in kf_ids:
                            # 비-채점 프레임: prev_dir 갱신만(폐루프), 계측·sample·집계 없음
                            if gate_mod is not None:
                                cvec, _m, _mm = encode_condition(vlm_mod, processor, imgs, prompt, hist)
                                prev_dir = int(gate_mod(cvec.unsqueeze(0).to(device)).argmax(-1).item())
                            continue
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()               # ── 계측 시작: encode + sample
                        cvec, mem, mem_mask = encode_condition(vlm_mod, processor, imgs, prompt, hist)
                        cond = cvec.unsqueeze(0).to(device); mem = mem.to(device); mem_mask = mem_mask.to(device)
                        pred_dir = None
                        if gate_mod is not None:
                            pred_dir = int(gate_mod(cond).argmax(-1).item())   # dir(t) 예측 → 다음 프레임 뷰용
                            prev_dir = pred_dir
                        pred_norm = _infer(rec, cond, mem, mem_mask)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        if not warmup:                         # 예열 패스는 계측·집계 안 함
                            scored += 1
                            if scored > 1:                     # ── 계측 끝(측정패스 첫 채점도 보수적 제외)
                                infer_times_ms.append((time.perf_counter() - t0) * 1000.0)
                            _accum(rec, pred_norm, pred_dir)   # ── 계측 밖: 지표 집계
        else:
            # 프레임 독립: keyframe_eval이면 선택된 keyframe만 채점(prev_dir 의존 없어 upfront 필터).
            scored_recs = [r for r in ade_recs if (not keyframe_eval) or (r["id"] in kf_ids)]
            for warmup in (True, False):                        # pass0=예열(집계 제외), pass1=측정
                for i, rec in enumerate(scored_recs, 1):
                    prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()                   # ── 계측 시작: encode + sample
                    if gate_mod is not None:                  # (구형 Pass 게이팅) 전방3뷰 예측→뷰게이팅
                        cond, mem, mem_mask, pred_dir, _uv = gate_closedloop_encode(
                            vlm_mod, gate_mod, processor, rec, prompt, temporal_on, device)
                    else:                                     # (구형) maneuver_lateral(미래 GT)+thr 게이팅
                        pred_dir = None
                        _ml = rec_maneuver(rec)
                        cvec, mem, mem_mask = encode_condition(vlm_mod, processor,
                                                               gate_views(rec["images"], _ml, man_thr), prompt,
                                                               gate_views(history_for(rec, temporal_on), _ml, man_thr))
                        cond = cvec.unsqueeze(0).to(device); mem = mem.to(device); mem_mask = mem_mask.to(device)
                    pred_norm = _infer(rec, cond, mem, mem_mask)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if not warmup:                             # 예열 패스는 계측·집계 안 함
                        if i > 1:                              # ── 계측 끝(측정패스 1번째도 보수적 제외)
                            infer_times_ms.append((time.perf_counter() - t0) * 1000.0)
                        _accum(rec, pred_norm, pred_dir)       # ── 계측 밖: 지표 집계
    plan_agg = aggregate(plan_rows)                            # 각 지표 nanmean + NPS 유효 표본 수
    # 속도 통계(ms, 워밍업 제외): fastest=최솟값, slowest=최댓값, median=중앙값, mean=산술평균.
    t_sorted = sorted(infer_times_ms)
    n_t = len(t_sorted)
    if n_t:
        median_ms = (t_sorted[n_t // 2] if n_t % 2 else (t_sorted[n_t // 2 - 1] + t_sorted[n_t // 2]) / 2)
        speed_stats = {"n_timed": n_t, "fastest_ms": round(t_sorted[0], 2), "slowest_ms": round(t_sorted[-1], 2),
                      "median_ms": round(median_ms, 2), "mean_ms": round(sum(t_sorted) / n_t, 2)}
    else:
        speed_stats = {"n_timed": 0, "fastest_ms": None, "slowest_ms": None, "median_ms": None, "mean_ms": None}
    n_scored = len(ades)                                     # 실제 채점 표본 수(keyframe_eval이면 keyframe만)
    metrics = {"n_samples": n_scored, "n_recs": len(ade_recs), "keyframe_eval": keyframe_eval,
               "keyframe_select": list(keyframe_select), "model": cfg["model"],
               "ADE_mean": round(float(np.mean(ades)), 3), "FDE_mean": round(float(np.mean(fdes)), 3),
               "ADE_baseline_cv": round(float(np.mean(cv_ades)), 3),
               "FDE_baseline_cv": round(float(np.mean(cv_fdes)), 3), "ode_steps": ode_steps,
               "maneuver_lateral_thr": man_thr, "inference_speed_ms": speed_stats,
               **plan_agg}                                     # NC_mean..NPS_mean, n_nps
    if gate_mod is not None:                                  # 폐루프 게이트 예측 정확도(gate_direction GT 대비)
        metrics["gate_closedloop"] = True
        metrics["gate_pred_acc"] = round(gate_correct / gate_total, 3) if gate_total else None
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    # Table 3 스타일 블록을 콘솔에 출력(rank0). 높을수록 좋음(ADE만 미터·낮을수록 좋음).
    _kf = " [keyframe-only]" if keyframe_eval else ""
    print(f"\n=== planning metrics (nuReasoning Table 3 style, 5s horizon, n={n_scored}{_kf}) ===")
    print(format_table(plan_agg, ade=metrics["ADE_mean"]))
    print(f"(NPS 유효 표본 {plan_agg.get('n_nps', 0)}/{n_scored}; 미니 근사 — planning_metrics.py 주석 참고)\n")
    print(f"=== inference speed (ms/sample, VLM encode + DiT sample, n={speed_stats['n_timed']}, "
          f"1번째 워밍업 제외, selective-view thr={man_thr}) ===")
    if speed_stats["n_timed"]:
        print(f"  Fastest: {speed_stats['fastest_ms']:8.2f} ms")
        print(f"  Slowest: {speed_stats['slowest_ms']:8.2f} ms")
        print(f"  Median : {speed_stats['median_ms']:8.2f} ms")
        print(f"  Mean   : {speed_stats['mean_ms']:8.2f} ms\n")
    else:
        print("  (표본 부족 — n_samples<=1)\n")
    with open(out_dir / f"{tag}_predictions.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ② BEV 시각화 + Pred reasoning 생성: reasoning 보유 샘플 우선, 전체에서 균등 간격으로 viz_n개.
    #    Pred reasoning은 8뷰 가운데 요약에 표시(별도 QnA 파일 출력은 안 함 — VLM 백본 테스트가 아님).
    with_reas = [r for r in recs if r["output"].get("reasoning")]   # GT reasoning 있는 프레임만 시각화(폴백 없음)
    chosen = select_diverse(with_reas, viz_n)
    samples = []
    do_reas = cfg.get("vlm_mode") != "frozen"
    with torch.no_grad():
        for rec in chosen:
            prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
            _ml = rec_maneuver(rec)
            hist = gate_views(history_for(rec, temporal_on), _ml, man_thr)
            cvec, mem, mem_mask = encode_condition(vlm_mod, processor,
                                                   gate_views(rec["images"], _ml, man_thr), prompt, hist)
            cond = cvec.unsqueeze(0).to(device); mem = mem.to(device); mem_mask = mem_mask.to(device)
            ego = torch.tensor([ego_vec(rec, dit_mod.ego_dim)], dtype=torch.float32, device=device) \
                if dit_mod.ego_dim > 0 else None
            pred_wp = normalizer.denormalize(dit_mod.sample(cond, steps=ode_steps, deterministic=True, ego=ego,
                                                            mem=mem, mem_mask=mem_mask)[0].cpu()).numpy()
            gt_wp = np.asarray(rec["output"]["waypoints"])
            pred = upsample_waypoints(pred_wp)                # (51,2) @10Hz — 매끄러운 BEV + 일관 ADE
            gt = upsample_waypoints(gt_wp)                    # (51,2)
            d = np.linalg.norm(pred - gt, axis=1)
            gen = None
            if do_reas:                                   # reasoning generate는 KV캐시로 메모리↑ → OOM이면 건너뜀
                try:
                    gen = generate_reasoning(vlm_mod, processor, rec["images"], prompt, 256, hist)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gen = "(OOM: reasoning 생성 생략)"
            samples.append({"id": rec["id"], "gt": gt.tolist(), "pred": pred.tolist(),
                            "ade": float(d.mean()), "fde": float(d[-1]),
                            "images": rec["images"], "mission": rec.get("mission"),
                            "gt_reasoning": rec["output"].get("reasoning"),
                            "pred_reasoning": gen,        # 8뷰 가운데 요약에 표시
                            # selective-view: 실제 VLM에 입력된 뷰 이름 집합(None=게이팅 off, 전 8뷰 입력).
                            #   plot_8view가 이걸로 "꺼진 뷰"를 시각적으로 구분(단순 missing과 다름).
                            "gated_views": (sorted(im["view"] for im in gate_views(rec["images"], _ml, man_thr))
                                           if man_thr is not None and man_thr >= 0 else None)})
    save_visuals(samples, out_dir, tag, per_page,             # {tag}_trajectories_{i}.png + {tag}_8view/{i}_images/
                 reasoning_types=cfg.get("reasoning_types", []))   # 3뷰 Pred 열: 학습한 reasoning 종류만
    if vlm_was_train:
        vlm_mod.train()
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL), help="epoch eval용 val.jsonl(trajectory 샘플만 사용)")
    ap.add_argument("--out", default=None,
                    help="모델 저장 경로. 미지정 시 그 실행 폴더 results/traj_reas/<timestamp>/model/에 저장 "
                         "(실행별 보존 → 과거 실행도 평가 가능). 명시하면 그 경로에 저장.")
    ap.add_argument("--eval-every", type=int, default=1,
                    help="N epoch마다 val로 eval(평균 val loss). 0=eval 생략")
    ap.add_argument("--vlm-mode", choices=["full", "lora", "frozen"], default="full",
                    help="full=VLM+DiT 공동 full FT(논문 충실, OOM 위험) / lora=VLM QLoRA+DiT(24GB 안전) / frozen=DiT만")
    ap.add_argument("--epochs", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=1, help="full FT는 1 권장(VLM forward 그래프 누적)")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="그래디언트 누적 step 수 K. micro-batch를 K번 forward/backward해 grad를 누적한 뒤 "
                         "한 번만 optimizer.step() → 메모리는 batch-1 수준 그대로 두고 유효 배치만 키운다. "
                         "유효 배치 = batch_size × K × world_size(GPU 수). 논문 유효 배치 64 재현: 4GPU→16, 8GPU→8.")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader 워커 프로세스 수(load+preprocess 오프로드→prefetch로 GPU forward와 겹침). "
                         "0=메인 프로세스 동기 로드. 프로파일상 데이터준비가 step의 ~58%라 >0 권장(64코어면 8 안전).")
    ap.add_argument("--prefetch-factor", type=int, default=2,
                    help="워커당 미리 준비할 배치 수(num_workers>0일 때만 유효). 총 프리페치 = num_workers × 이 값.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--vlm-lr", type=float, default=5e-5, help="VLM 파라미터 학습률(보통 DiT보다 작게)")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (논문 0.01)")
    ap.add_argument("--warmup-ratio", type=float, default=0.05,
                    help="논문식 LR 스케줄: 전체 optimizer step의 이 비율만큼 linear warmup 후 cosine annealing→0. "
                         "0이면 warmup 없이 cosine만. --lr-scheduler none이면 상수 LR(스케줄 무시).")
    ap.add_argument("--lr-scheduler", choices=["cosine", "none"], default="cosine",
                    help="cosine=논문식 warmup+cosine annealing(기본) / none=상수 LR(스케줄러 없음, 디버그용).")
    ap.add_argument("--optimizer", choices=["adamw", "paged8bit"], default="adamw",
                    help="adamw=표준 32-bit(정밀도·속도) / paged8bit=PagedAdamW8bit(메모리↓, 8bit정밀도·paging비용). "
                         "⚠️ full 모드는 2B optimizer state가 커서 adamw면 24GB OOM 위험 → paged8bit 권장.")
    ap.add_argument("--reas-weight", type=float, default=1.0,
                    help="공동 손실 λ: total = flow_loss + λ·LM_loss(reasoning). frozen이면 무시")
    ap.add_argument("--reasoning-types", default="spatial,decision,counterfactual",
                    help="LM_loss로 supervision할 reasoning 종류(쉼표구분). 단계별 ablation: "
                         "'none'=baseline(궤적만) → 'spatial' → 'spatial,decision' → "
                         "'spatial,decision,counterfactual'(논문 최종). 선택 종류를 한 텍스트로 합쳐 학습.")
    ap.add_argument("--reasoning-only", dest="reasoning_only", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="논문 정렬: reasoning 주석 있는 프레임(clip당 30~150, 1Hz)만 학습/평가. "
                         "reasoning-free 프레임(clip 앞 0,10,20 + tail)은 제외. --no-reasoning-only면 전 프레임 사용. "
                         "⚠️ --reasoning-types와 독립 — baseline도 동일 프레임에서 돌아 ablation 공정.")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--beta-alpha", type=float, default=2.0,
                    help="flow-matching 시간 t~Beta(α,β)의 α (논문: t를 Beta에서 샘플, α,β수치 미명시 → 기본 2). "
                         "α=β=1이면 Uniform(구형 균등 샘플).")
    ap.add_argument("--beta-beta", type=float, default=2.0,
                    help="flow-matching 시간 t~Beta(α,β)의 β (기본 2, 대칭 mode 0.5).")
    ap.add_argument("--ode-steps", type=int, default=5,
                    help="추론(sample) 시 Euler ODE 적분 스텝 수 K (논문 K=5). final-eval/eval_trajectory.py "
                         "기본값도 동일 — flow-matching은 t~Beta로 학습되면 소수 스텝으로도 정확.")
    ap.add_argument("--cross-attn", dest="cross_attn", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="(기본 켜짐, 논문 nuVLA #9) DiT가 VLM hidden **시퀀스 전체**에 cross-attention. "
                         "scene은 cross-attn으로 유입, AdaLN은 timestep만. --no-cross-attn이면 구형(pooled cond → AdaLN). "
                         "⚠️ 메모리: VLM 시퀀스(~수백 토큰)를 KV로 유지 — full 모드면 grad도 흐름.")
    ap.add_argument("--maneuver-lateral-thr", dest="maneuver_lateral_thr", type=float, default=-1.0,
                    help="selective-view 게이팅 임계값(m). <0=off(전 8뷰=baseline). >0이면 maneuver_lateral로 "
                         "뷰 선택: 항상 전면3뷰(L0/F/R0), ml>+thr(좌)→L1/L2/B 추가, ml<−thr(우)→R1/R2/B 추가. "
                         "실험: 0.5 / 0.75 / 1.0. 평가도 이 값을 cfg에서 읽어 동일 적용.")
    ap.add_argument("--gate-weight", dest="gate_weight", type=float, default=0.0,
                    help="selective-view 1단계 게이트(cond→직진/좌/우 3-class) 손실 가중 μ. total = flow + λ·LM + "
                         "μ·gate_CE. 0(기본)=게이트 헤드 비활성. >0이면 GateHead 학습 — 타깃은 build_sft가 심은 "
                         "gate_direction(과거 결정 이월값, 인과 안전).")
    ap.add_argument("--gate-hidden", dest="gate_hidden", type=int, default=128,
                    help="GateHead 은닉 차원(작은 MLP: cond_dim→hidden→3).")
    ap.add_argument("--gate-focal-gamma", dest="gate_focal_gamma", type=float, default=2.0,
                    help="게이트 focal loss의 감쇠 지수 γ(Lin et al. 2017). 0=순수 class-weighted CE. "
                         "클수록 이미 잘 맞추는 다수클래스(직진) 손실을 더 강하게 억제해 소수클래스(좌/우)에 집중.")
    ap.add_argument("--temporal-clip", dest="temporal_clip", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="clip 단위 시간순 시퀀스 학습/추론(진짜 폐루프). True면 프레임을 clip별 시간 순서로 롤아웃하며 "
                         "**이전 프레임의 게이트 예측 dir(t-1)**로 현재 프레임 뷰를 게이팅(selective-view일 때). "
                         "False면 기존 프레임 독립 경로(DataLoader 셔플). DDP는 clip 단위 샤딩.")
    ap.add_argument("--selective-view", dest="selective_view", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="게이트 예측(dir(t-1))으로 현재 프레임 뷰를 실제로 게이팅할지. True=폐루프 뷰 선택(직진→전면3뷰, "
                         "좌/우→방향3뷰 추가). False=항상 전 8뷰(게이트는 --gate-weight>0이면 학습만 되고 뷰 선택엔 미사용). "
                         "3설정: baseline(--no-selective-view --gate-weight 0) / +decision(--no-selective-view --gate-weight 1) "
                         "/ selective+decision(--selective-view --gate-weight 1). ⚠️ --temporal-clip와 함께 사용.")
    ap.add_argument("--forcing", choices=["student", "teacher"], default="student",
                    help="학습 시 뷰 게이팅 방향의 출처(--temporal-clip --selective-view에서만 유효). "
                         "student(기본, 폐루프): 프레임 t 뷰를 게이트의 t-1 예측으로 게이팅 — 추론과 동일 분포지만 "
                         "게이트가 나쁘면 붕괴 자기강화. teacher: 프레임 t 뷰를 그 프레임 GT gate_direction으로 게이팅 "
                         "— 프레임 독립·항상 올바른 뷰로 학습해 안정적. ⚠️ 추론은 GT가 없어 항상 student 폐루프. "
                         "--no-selective-view면 forcing 무관하게 항상 전 8뷰(순차 입력만 유지).")
    ap.add_argument("--keyframe-eval", dest="keyframe_eval", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="(논문 정렬) 검증/최종평가를 **keyframe에서만** 채점. temporal-clip 폐루프는 전 프레임을 "
                         "순차 롤아웃(prev_dir 유지)하되 선택된 keyframe에서만 ADE/planning·속도를 집계. 학습은 영향 "
                         "없음(전 reasoning 프레임 사용). 기본 False(reasoning 프레임 전체에서 평가).")
    ap.add_argument("--keyframe-select", dest="keyframe_select", default="1",
                    help="--keyframe-eval 시 clip당 decision 프레임 3개 중 **순서(ordinal)**로 채점 대상 선택. "
                         "0=첫째(~50), 1=정중앙(~100), 2=마지막(~150). 기본 '1'(정중앙=논문식 clip당 keyframe 1개). "
                         "'0,1,2'=3개 전부. 절대 frame_index가 clip마다 50/100/150 또는 51/101/151로 어긋나 순서로 매칭.")
    ap.add_argument("--ego-state-token", dest="ego_state_token", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="(기본 켜짐, 논문 nuVLA) ego(현재 dynamics + 이동 history)를 DiT **state token**으로 시퀀스에 "
                         "넣어 waypoint가 self-attention으로 참조. --no-ego-state-token이면 구형(ego→AdaLN) 방식.")
    ap.add_argument("--limit", type=int, default=0, help="subset train for a quick run (0=all)")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke test); -1=use epochs")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"랜덤 시드(torch.manual_seed + DataLoader 셔플 generator + epoch별 clip 셔플). "
                         f"기본={SEED}(vlm.SEED). 재현성 확인·시드 스윕용으로 --seed로 override.")
    # 학습 종료 후 최종 평가(ADE/FDE + BEV 시각화)를 rank0이 자동 실행. 이미 메모리의 모델 재사용(재로딩 없음).
    ap.add_argument("--final-eval", dest="final_eval", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="(기본 켜짐) 학습 끝나면 ADE/FDE + BEV 시각화를 run_dir에 자동 저장. "
                         "--no-final-eval로 생략(smoke/디버그/GPU 빠듯할 때)")
    ap.add_argument("--final-eval-limit", type=int, default=0,
                    help="최종 ADE/FDE 평가 샘플 수(0=전체)")
    ap.add_argument("--final-eval-viz", type=int, default=18,
                    help="최종 BEV/8뷰 시각화 장면 수(전체에서 균등 간격 선택)")
    ap.add_argument("--final-eval-per-page", type=int, default=6,
                    help="격자 1장당 장면 수(_1/_2/_3 페이지 분할 + 8뷰 1/2/3 폴더 단위)")
    args = ap.parse_args()

    import torch
    # DDP: torchrun이면 프로세스 그룹 초기화 + 이 rank의 GPU 고정. 단일 실행이면 device만 잡고 폴백.
    device = ddp.setup()                                   # "cuda:{local_rank}"
    lr_idx = ddp.local_rank()
    main_proc = ddp.is_main()

    def log(*a):                                           # rank0만 출력(로그 중복 방지)
        if main_proc:
            print(*a)

    seed = args.seed                                       # vlm.SEED 기본값을 --seed로 override 가능
    torch.manual_seed(seed)
    ds = TrajDataset(Path(args.train), args.limit)
    if len(ds) == 0:
        raise SystemExit("no trajectory samples in train.jsonl — rebuild SFT data (build_sft.py) first.")
    # 논문 정렬(--reasoning-only): reasoning 주석 있는 프레임만 남긴다(reasoning-free clip 앞뒤 제외).
    #   ⚠️ args.limit 슬라이싱 뒤에 필터하면 limit 표본이 대부분 잘려나갈 수 있으나(초반 frame 0,10,20이
    #   reasoning-free) 스모크용이라 허용. 본학습은 limit=0이라 무관.
    train_n_before = len(ds.records)
    if args.reasoning_only:
        ds.records, train_n_before, _ = filter_reasoning_only(ds.records)
        if len(ds.records) == 0:
            raise SystemExit("--reasoning-only인데 reasoning 주석 있는 궤적 샘플이 0개 — "
                             "build_sft.py가 reasoning_parts를 심었는지 확인(또는 --no-reasoning-only).")
    n_points = len(ds.records[0]["output"]["waypoints"])   # 고정 길이(빌더가 보장). 논문 정렬=10(2Hz×5s)
    point_dim = len(ds.records[0]["output"]["waypoints"][0])  # waypoint 차원: 3=논문(fwd,left,θ), 2=구형(fwd,left)
    ego_dim = len(ds.records[0].get("ego_state") or [])    # ego 운동상태 차원(vx,vy,ax,ay=4; 없으면 0)
    prompt_tpl = TRAJ_PROMPT_FILE.read_text()
    train_vlm = args.vlm_mode != "frozen"                  # VLM이 학습에 참여하는가
    reasoning_types = parse_reasoning_types(args.reasoning_types)   # [](baseline)~[spatial,decision,counterfactual]
    keyframe_select = tuple(int(x) for x in str(args.keyframe_select).split(",") if x.strip() != "") \
        or (0, 1, 2)                                               # keyframe 채점 대상 순서(기본 (1,)=정중앙)

    # 시간 맥락(temporal): vlm.TEMPORAL 스위치가 켜졌고 + 데이터에 history_images가 실제로 박혀 있을 때만
    #   과거 1 timestep(8뷰)을 condition에 넣는다(빌드 때 offset이 이미지에 안 닿으면 데이터에 없을 수 있음).
    #   학습이 실제 쓴 값을 cfg에 저장 → 평가(eval_*)가 동일하게 따라 빌드·학습·평가가 일관된다.
    data_has_hist = any(r.get("history_images") for r in ds.records)
    temporal_on = bool(TEMPORAL and data_has_hist)
    if TEMPORAL and not data_has_hist:
        log("⚠️ vlm.TEMPORAL=True인데 train 데이터에 history_images가 없음 → 단일 프레임으로 학습. "
            "build_sft.py를 TEMPORAL=True로 재실행했는지/offset이 이미지(1Hz)에 닿는지 확인.")

    log(f"trajectory train samples: {len(ds)}  | waypoints/sample: {n_points}  | "
        f"vlm-mode: {args.vlm_mode}  | world_size: {ddp.world_size()}  | output: {args.out}")
    log(f"temporal: {'ON (과거 1 timestep, offset %d프레임)' % TEMPORAL_HISTORY_OFFSET if temporal_on else 'OFF (현재 프레임만)'}")
    log(f"reasoning-types: {reasoning_types if reasoning_types else 'none (baseline, 궤적만)'}")
    if args.reasoning_only:
        log(f"reasoning-only: ON (논문 정렬) — reasoning 주석 있는 프레임만 {len(ds)}/{train_n_before} "
            f"({100*len(ds)/max(1,train_n_before):.1f}%), reasoning-free {train_n_before-len(ds)}개 제외")
    else:
        log(f"reasoning-only: OFF — 전 궤적 프레임 {len(ds)}개 사용(reasoning-free 포함)")
    log(f"cross-attn: {'ON (논문 #9, VLM 시퀀스 cross-attention)' if args.cross_attn else 'OFF (구형 pooled cond → AdaLN)'}")
    if args.maneuver_lateral_thr is not None and args.maneuver_lateral_thr >= 0:
        log(f"maneuver-thr(구형): ON (thr={args.maneuver_lateral_thr}m) — 미래GT 게이팅(폐기 예정)")
    # temporal-clip(신규 폐루프) 상태 + 3설정 판별
    if args.temporal_clip:
        if args.selective_view and args.gate_weight <= 0:
            log("⚠️ --selective-view True인데 --gate-weight 0 → 게이트가 없어 방향 예측 불가. gate-weight>0 필요(전 8뷰로 폴백).")
        mode = ("selective+decision (폐루프 뷰 게이팅 + gate focal)" if args.selective_view and args.gate_weight > 0
                else "8view+decision (전 8뷰, gate focal만)" if args.gate_weight > 0
                else "8view baseline (전 8뷰, gate 없음)")
        log(f"temporal-clip: ON | 설정 = {mode} | selective_view={args.selective_view} gate_weight={args.gate_weight}")
        log("  clip 단위 시간순 롤아웃 — dir(t-1) 게이트 예측 → 프레임 t 뷰 게이팅(폐루프). DDP=clip 샤딩+sync_grads.")
    else:
        log("temporal-clip: OFF — 프레임 독립 경로(DataLoader 셔플)")

    # 1) waypoints 정규화기 fit(전 rank 동일 데이터 → 동일 통계, 별도 동기화 불필요)
    normalizer = TrajectoryNormalizer.fit(ds.waypoints())
    log(f"normalizer mean={[round(m,2) for m in normalizer.mean.tolist()]} "
        f"std={[round(s,2) for s in normalizer.std.tolist()]}")

    # 2) VLM(mode별, 이 rank의 GPU에 배치) + condition dim probe(grad 불필요)
    log(f"loading VLM {args.model} (mode={args.vlm_mode}) on rank {ddp.rank()} (cuda:{lr_idx}) ...")
    processor = load_processor(args.model)
    vlm = load_vlm(args.model, args.vlm_mode, device_index=lr_idx)

    with torch.no_grad():
        probe, _, _ = encode_condition(vlm, processor, ds.records[0]["images"],
                                       prompt_tpl.replace("{mission}", ds.records[0].get("mission") or "drive safely"),
                                       history_for(ds.records[0], temporal_on))
    cond_dim = probe.shape[0]
    log(f"condition dim (probed): {cond_dim}")

    # 3) DiT 헤드(fp32, 이 rank GPU) + optimizer. point_dim=3(논문 x,y,θ)/2(구형). ego_dim>0이면 ego 조건화 포함.
    dit = TrajectoryDiT(cond_dim=cond_dim, n_points=n_points, point_dim=point_dim,
                        d_model=args.d_model, n_layers=args.n_layers, ego_dim=ego_dim,
                        ego_as_state_token=args.ego_state_token,
                        beta_alpha=args.beta_alpha, beta_beta=args.beta_beta,
                        cross_attn=args.cross_attn).to(device)
    dit_mod = dit                                          # (표준 DDP는 TrajReasVLA를 감쌈; dit_mod=원본 별칭)

    # 3-a) **cond 정규화 통계 적합**(성능 핵심): VLM hidden mean-pool은 transformer massive-activation
    #   공통성분이 지배해 장면이 달라도 cond이 거의 동일(cosine~0.99)하다 → DiT가 cond을 무시하고
    #   데이터셋 평균 궤적으로 붕괴(ADE가 constant-velocity보다 나빠짐). 학습셋 subsample로 per-dim
    #   mean/std를 구해 DiT 내부 정규화에 넣으면 장면별 차이가 드러난다(진단: cosine 0.99→0.29).
    #   전 rank가 동일 subsample(앞 n_fit개)·동일 초기 VLM으로 적합 → DiT 버퍼가 rank 간 일치(동기화 불필요).
    #   ⚠️ full 모드는 VLM이 학습 중 드리프트하지만, massive-activation outlier 구조는 안정적이라
    #      초기 적합으로도 공통성분 제거 효과가 유지된다(버퍼는 고정).
    #   ⚠️ cross_attn이면 DiT가 pooled cond을 안 쓰지만(scene은 cross-attn으로 유입), 게이트 헤드(--gate-weight>0)는
    #      cross_attn 여부와 무관하게 **항상 pooled cond**를 쓰므로 게이트가 켜져 있으면 fit을 스킵하지 않는다.
    gate_on = args.gate_weight > 0
    if not args.cross_attn or gate_on:
        n_fit = min(256, len(ds))
        with torch.no_grad():
            fit_conds = torch.stack([
                encode_condition(vlm, processor, ds.records[i]["images"],
                                 prompt_tpl.replace("{mission}", ds.records[i].get("mission") or "drive safely"),
                                 history_for(ds.records[i], temporal_on))[0]
                for i in range(n_fit)
            ])                                                 # (n_fit, Dc) fp32
        if not args.cross_attn:
            dit_mod.set_cond_stats(fit_conds.mean(0), fit_conds.std(0))
        log(f"cond normalizer fit on {n_fit} samples | mean‖·‖={fit_conds.mean(0).norm():.1f} "
            f"std(mean)={fit_conds.std(0).mean():.3f}")
    else:
        fit_conds = None
        log("cross_attn=ON, gate=OFF → pooled cond 미사용, cond normalizer fit 스킵(scene은 cross-attention으로 유입)")

    # 3-b) ego 운동상태 정규화 통계 적합(ego_dim>0): 전체 학습셋의 [vx,vy,ax,ay] per-dim mean/std.
    #   ego 경로가 c에 직접 더해지므로 단위분산으로 맞춰 cond 경로와 균형을 잡는다.
    if ego_dim > 0:
        ego_all = torch.tensor([ego_vec(r, ego_dim) for r in ds.records], dtype=torch.float32)
        dit_mod.set_ego_stats(ego_all.mean(0), ego_all.std(0))
        log(f"ego normalizer fit | dim={ego_dim} mean={[round(m,2) for m in ego_all.mean(0).tolist()]} "
            f"std={[round(s,2) for s in ego_all.std(0).tolist()]}")
    # 3-c) selective-view 게이트 헤드(--gate-weight>0): pooled cond → 3-class(직진/좌/우). class-weighted
    #   focal loss로 불균형(직진 다수) 보정. alpha는 학습셋 gate_direction 분포의 역빈도(1/freq, 평균 1로 정규화).
    gate_mod = None
    gate_alpha = None
    if gate_on:
        gate_counts = Counter(r["output"].get("gate_direction") for r in ds.records)
        n_g = sum(v for k, v in gate_counts.items() if k is not None)
        if n_g == 0:
            raise SystemExit("--gate-weight>0인데 gate_direction 있는 trajectory 샘플이 0개 — "
                             "build_sft.py 재빌드(gate_direction 필드) 필요.")
        freq = torch.tensor([gate_counts.get(0, 0), gate_counts.get(1, 0), gate_counts.get(2, 0)],
                            dtype=torch.float32) / n_g
        inv = 1.0 / freq.clamp_min(1e-6)
        gate_alpha = inv / inv.mean()               # 평균 1로 정규화(전체 손실 스케일 유지)
        gate_mod = GateHead(cond_dim=cond_dim, hidden=args.gate_hidden).to(device)
        if fit_conds is not None:                   # DiT와 동일 cond 통계 재사용(별 fit 불필요)
            gate_mod.set_cond_stats(fit_conds.mean(0), fit_conds.std(0))
        log(f"gate head: ON | class dist(0=straight,1=left,2=right)={dict(gate_counts)} | "
            f"alpha(class weight)={[round(a,3) for a in gate_alpha.tolist()]} | "
            f"focal_gamma={args.gate_focal_gamma} | weight(μ)={args.gate_weight}")
        _train_gate = ("gate_direction GT로 뷰 게이팅(teacher-forcing)" if args.forcing == "teacher"
                       else "이전 프레임 게이트 예측 dir(t-1)로 뷰 게이팅(student-forcing 폐루프)")
        log(f"gate 뷰 게이팅: 학습={_train_gate} / 추론=게이트 예측으로 뷰 게이팅(closed-loop, GT 미사용) "
            "→ --maneuver-lateral-thr 무시  (forcing/게이팅은 --temporal-clip --selective-view에서만 유효)")
    else:
        log("gate head: OFF (--gate-weight 0)")

    # DDP 처리 메모: 흩어진 forward를 TrajReasVLA(nn.Module)로 묶고 표준 "DDP(model).forward()"를 쓴다.
    #    reasoning 유무로 step별 그래프가 달라져도(LM head used/unused) 0-더미(0·logits)로 lm_head가 매 step
    #    grad를 받게 해 rank 간 param-ready 집합을 고정 → find_unused_parameters=False로 hang 없이 동작한다.
    #    vlm_mod/dit_mod는 래핑 이후에도 원본 서브모듈을 가리켜(=model.vlm/model.dit) 저장·평가·probe에 그대로 쓴다.
    vlm_mod = vlm

    vlm_params = [p for p in vlm_mod.parameters() if p.requires_grad] if train_vlm else []
    # DiT와 VLM은 학습률을 분리(VLM은 보통 더 작게). 게이트는 DiT와 같은 lr 그룹(소형 헤드, 별도 lr 불필요).
    param_groups = [{"params": dit_mod.parameters(), "lr": args.lr}]
    if gate_mod is not None:
        param_groups.append({"params": gate_mod.parameters(), "lr": args.lr})
    if vlm_params:
        param_groups.append({"params": vlm_params, "lr": args.vlm_lr})
    # optimizer는 --optimizer로 선택(기본 adamw). full 모드 + adamw는 2B optimizer state(32-bit)가
    # ~16GB라 24GB OOM 위험 → paged8bit 권장. 경고만 띄우고 사용자 선택을 존중.
    if args.optimizer == "paged8bit":
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(param_groups, weight_decay=args.weight_decay)
    else:
        if args.vlm_mode == "full":
            log("⚠️ full + adamw(32-bit): 2B optimizer state ~16GB라 24GB에서 OOM 위험. "
                "OOM 나면 --optimizer paged8bit 사용.")
        opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    n_dit = sum(p.numel() for p in dit_mod.parameters())
    n_vlm = sum(p.numel() for p in vlm_params)
    log(f"trainable params: DiT {n_dit:,} + VLM {n_vlm:,}  (per-rank, grad는 all-reduce 동기화)")

    # 4) 커스텀 공동 학습 루프: 한 forward로 (condition, LM_loss) → flow_loss + λ·LM_loss → backward.
    #    DDP: 이 rank가 맡을 샘플만 보고(샤딩), grad는 backward에서 자동 all-reduce 동기화.
    my_indices = ddp.shard(list(range(len(ds))))            # rank별 분담 인덱스
    # ⚠️ DDP 데드락 방지: 1554를 4 rank로 나누면 rank별 샘플 수가 다름(389/389/388/388). step 수가
    #    다르면 epoch 끝에서 일부 rank만 backward(DDP all_reduce)/로깅 all_reduce를 더 호출해 collective
    #    불일치→데드락. → epoch당 step을 전 rank **최소값**으로 고정, 남는 샘플은 그 epoch에서만 스킵
    #    (매 epoch 셔플되므로 장기적으로 전 샘플이 고루 쓰임).
    my_steps = (len(my_indices) + args.batch_size - 1) // args.batch_size
    steps_per_epoch = ddp.all_reduce_min(my_steps)          # 전 rank 공통 step 수
    # temporal-clip: clip 단위 시퀀스 롤아웃 → step = clip 1개(내부에서 프레임 순차 처리). clip을 rank별 샤딩.
    my_clips = None
    if args.temporal_clip:
        all_clips = group_records_by_clip(ds.records)       # clip_token 정렬 → 전 rank 동일 순서(샤딩 재현)
        my_clip_idx = ddp.shard(list(range(len(all_clips))))
        my_clips = [all_clips[i] for i in my_clip_idx]
        steps_per_epoch = ddp.all_reduce_min(len(my_clips)) # clip 1개 = 1 step
    total_steps = args.max_steps if args.max_steps > 0 else int(args.epochs * steps_per_epoch)
    n_reas = sum(1 for r in ds.records if reasoning_target(r, reasoning_types))  # 선택 종류로 LM_loss 걸릴 샘플 수
    # 그래디언트 누적: K micro-step마다 한 번 optimizer.step(). 유효 배치 = batch_size × K × world_size.
    grad_accum = max(1, args.grad_accum)
    eff_batch = args.batch_size * grad_accum * ddp.world_size()

    # LR 스케줄(논문식): linear warmup(warmup_ratio) → cosine annealing→0. **optimizer step 단위**로 진행하므로
    #   total_opt_steps = 실제 optimizer.step() 호출 횟수(=grad_accum 경계 수)를 정확히 센다. epoch당 opt step은
    #   ceil(steps_per_epoch/grad_accum)(마지막 부분 그룹도 epoch 끝 경계에서 flush됨), 마지막 부분 epoch은 별도 계산.
    import math
    opt_steps_per_epoch = math.ceil(steps_per_epoch / grad_accum)
    n_full_epochs, rem_micro = divmod(total_steps, steps_per_epoch)
    total_opt_steps = n_full_epochs * opt_steps_per_epoch + (math.ceil(rem_micro / grad_accum) if rem_micro else 0)
    total_opt_steps = max(1, total_opt_steps)
    warmup_steps = int(args.warmup_ratio * total_opt_steps) if args.lr_scheduler == "cosine" else 0

    def _lr_factor(cur):                                    # cur = 0-based optimizer step 인덱스
        if cur < warmup_steps:                             # linear warmup
            return (cur + 1) / max(1, warmup_steps)
        prog = (cur - warmup_steps) / max(1, total_opt_steps - warmup_steps)   # cosine annealing 1→0
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = None
    if args.lr_scheduler == "cosine":
        # LambdaLR: 두 param group(dit/vlm) 각 base lr에 동일한 factor를 곱함(둘 다 warmup+cosine 형태 유지).
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_factor)

    # eval용 val(trajectory) 샘플 — 분산 eval이라 전 rank가 로드(각자 ddp.shard로 자기 몫만 forward).
    val_records = []
    if args.eval_every > 0:
        val_records = TrajDataset(Path(args.val)).records
        if args.reasoning_only:                            # 학습과 동일 프레임 집합에서 val_loss 측정(일관)
            val_records, _vn0, _vn1 = filter_reasoning_only(val_records)

    # 실행별 산출물/로깅 = results/traj_reas/<timestamp>/. rank0만 실제 기록(나머지 no-op).
    img_tok = IMAGE_MAX_PIXELS // (28 * 28)               # 비전토큰 상한(이미지당). 28×28px=토큰1.
    logger = RunLogger("traj_reas", is_main=main_proc)
    # 설정 기록: argparse 전체를 한 줄씩(가독성) + 실행 시점에만 정해지는 파생값 몇 개를 별도 섹션으로.
    #   argparse 원본을 빠짐없이 남겨 어떤 인자로 돌렸는지 로그만 봐도 재현 가능하게 한다.
    cfg_lines = ["=== train_traj_reas config ===", "[argparse]"]
    cfg_lines += [f"  {k} = {v}" for k, v in vars(args).items()]
    cfg_lines += [
        "[derived]",
        f"  world_size = {ddp.world_size()}",
        f"  eff_batch = {eff_batch}  (= batch_size {args.batch_size} × grad_accum {grad_accum} × world_size {ddp.world_size()})",
        f"  steps_per_epoch = {steps_per_epoch}",
        f"  total_steps = {total_steps}  (micro-steps)",
        f"  total_opt_steps = {total_opt_steps}  (optimizer updates)",
        f"  lr_schedule = {args.lr_scheduler}  (warmup {warmup_steps}/{total_opt_steps} opt-steps → cosine→0)",
        f"  weight_decay = {args.weight_decay}",
        f"  cond_dim = {cond_dim}",
        f"  image_tokens = {img_tok}  (<= {IMAGE_MAX_PIXELS} px)",
        f"  temporal_on = {temporal_on}  (offset {TEMPORAL_HISTORY_OFFSET} frames)",
        f"  reasoning_types(parsed) = {reasoning_types if reasoning_types else 'none (baseline)'}",
        f"  forcing = {args.forcing}  (뷰 게이팅 출처: student=폐루프예측 / teacher=GT; --temporal-clip --selective-view에서만 유효)",
        f"  train_samples = {len(ds)}",
        f"  reasoning_samples = {n_reas}/{len(ds)}",
        f"  val_samples = {len(val_records)}",
    ]
    # forcing은 clip 시퀀스 + 뷰 게이팅이 켜진 경우에만 의미. 그 외 조합에선 무시됨을 명시(혼동 방지).
    if not (args.temporal_clip and args.selective_view):
        cfg_lines.append(f"  ⚠️ forcing={args.forcing} 무시됨 — --temporal-clip --selective-view 아님 "
                         f"(temporal_clip={args.temporal_clip}, selective_view={args.selective_view}) → "
                         f"{'순차 입력이되 항상 전 8뷰' if args.temporal_clip else '프레임 독립(구형) 경로'}.")
    logger.info("\n".join(cfg_lines) + "\n")

    dit.train()
    # 학습 파라미터(전 rank 동일 순서) = grad clipping 대상(표준 DDP가 grad all-reduce는 자동 처리).
    clip_params = list(dit_mod.parameters()) + vlm_params
    # joint = VLM에 reasoning LM_loss를 걸지 여부. frozen(VLM 미학습)이거나 --reasoning-types none(baseline)이면
    #   LM_loss를 아예 안 건다(순수 궤적 학습). 그 외엔 선택된 reasoning 종류를 공동 supervision.
    joint = train_vlm and bool(reasoning_types)

    # DataLoader(프리페치): 이 rank의 shard를 워커가 load+preprocess(TrajTrainSet)→collate_traj로 배치화.
    #   워커가 다음 배치를 미리 만들어(prefetch_factor) GPU forward/backward와 겹친다(데이터준비 ~58% 은닉).
    #   shard 단위로 감싸고 shuffle=True(rank 내 셔플) + generator(SEED)로 epoch마다 재현가능 재셔플.
    #   drop_last=False: 배치 수 = ceil(shard/batch) ≥ steps_per_epoch → 아래 루프가 공통 step만큼만 소비.
    #   fork 워커가 CUDA를 건드리지 않으므로(전부 CPU 텐서) 안전. 토크나이저 병렬 경고는 모듈 상단서 off.
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id or 0
    # temporal-clip 경로는 DataLoader/DDP-wrapper를 쓰지 않는다(clip 내 순차 롤아웃 + prev_dir stateful이라
    #   워커 오프로드·표준 DDP forward-once 가정과 맞지 않음). 대신 clip을 직접 순회하고 backward 후 sync_grads로
    #   grad를 수동 all-reduce한다. 프레임 독립 경로(기존)는 DataLoader+prefetch+표준 DDP 그대로.
    loader = ddp_model = None
    sync_params = None
    if not args.temporal_clip:
        train_set = TrajTrainSet([ds.records[i] for i in my_indices], processor, prompt_tpl,
                                 temporal_on, reasoning_types, joint, ego_dim,
                                 maneuver_thr=args.maneuver_lateral_thr,
                                 gate_teacher_forcing=(args.gate_weight > 0))
        loader_kwargs = dict(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             collate_fn=partial(collate_traj, pad_id=pad_id),
                             pin_memory=torch.cuda.is_available(), drop_last=False,
                             generator=torch.Generator().manual_seed(seed))
        if args.num_workers > 0:                                # 워커 있을 때만 유효한 옵션
            loader_kwargs.update(persistent_workers=True, prefetch_factor=args.prefetch_factor)
        loader = DataLoader(train_set, **loader_kwargs)

        # VLM+DiT를 한 nn.Module로 래핑. 학습 루프의 forward를 model(batch) 단일 진입점으로 통일.
        model = TrajReasVLA(vlm_mod, dit_mod, normalizer, args.reas_weight, train_vlm, device,
                           gate=gate_mod, gate_weight=args.gate_weight, gate_alpha=gate_alpha,
                           gate_focal_gamma=args.gate_focal_gamma)
        # 표준 DDP: backward에서 grad 자동 all-reduce. 0-더미로 lm_head/gate가 매 step 그래프에 연결 →
        #   find_unused_parameters=False로 데드락 없음. broadcast_buffers=False(버퍼는 전 rank 동일).
        if ddp.is_dist():
            from torch.nn.parallel import DistributedDataParallel as DDP
            ddp_model = DDP(model, device_ids=[lr_idx], output_device=lr_idx,
                            find_unused_parameters=False, broadcast_buffers=False)
        else:
            ddp_model = model                                    # 단일 GPU: 래퍼 그대로
    else:
        # clip 경로: 수동 grad 동기화 대상 파라미터(전 rank 동일 순서). ddp.sync_grads가 None-grad는 0으로 채워
        #   collective에 참여시키므로 프레임별 그래프가 달라도 안전.
        sync_params = list(dit_mod.parameters()) + (list(gate_mod.parameters()) if gate_mod is not None else []) + vlm_params
        if gate_mod is not None:
            gate_mod.train()

    step = 0
    done = False
    epoch = 0
    # best 모델 선정: 검증마다 val_loss를 이전 최소와 비교해 더 낮으면 그 시점 가중치를 **CPU 스냅샷**한다.
    # 학습 종료 후 이 best 가중치를 복원해 저장 + 최종 eval에 쓴다(마지막 epoch이 항상 best는 아님).
    # 전 rank가 동일 val_loss(all_reduce)를 보지만 스냅샷/복원은 저장·eval을 맡는 rank0만(RAM 절약).
    # ⚠️ full 모드 스냅샷은 2B 가중치(~4GB)를 CPU로 복사 → GPU 메모리엔 영향 없음(24GB 안전).
    best_val = float("inf")
    best_epoch = -1
    best_dit_sd = None
    best_vlm_sd = None
    best_gate_sd = None
    while not done:
        epoch += 1
        ep_loss_sum, ep_n = 0.0, 0                          # 이 epoch 평균 train loss 집계
        step_in_ep = 0                                       # 이 epoch 안에서의 step(로깅용 1-base)
        opt.zero_grad(set_to_none=True)                      # 누적 그룹 시작 전 grad 초기화(이후엔 경계에서만)
        # DataLoader가 shard를 shuffle+prefetch로 흘려준다(워커가 다음 배치 load+preprocess를 미리 수행).
        # 전 rank 공통 steps_per_epoch 만큼만 소비(drop_last=False라 배치 수 ≥ steps_per_epoch → 남는 배치는
        # 이 epoch 스킵). 셔플은 loader의 generator(SEED)가 epoch마다 재현가능하게 재추출한다.
        # ── 두 학습 경로 분기: temporal-clip(clip 시퀀스 롤아웃) vs 프레임 독립(DataLoader) ──
        if args.temporal_clip:
            # clip 1개 = 1 step. clip_rollout_forward가 프레임 순차 처리 + 프레임별 backward(grad 로컬 누적).
            #   경계에서 ddp.sync_grads로 수동 all-reduce → clip/opt-step 수가 전 rank 동일이라 collective 일치.
            # ⚠️ 매 epoch **clip 순서만** 셔플(클립 내 frame 순서는 유지 → 인과 보존). clip끼리는 독립이라
            #   순서를 섞어도 폐루프에 문제없다. SEED+epoch 기반 재현가능 셔플(rank별 shard는 원래 다름).
            import random as _random
            order = list(range(len(my_clips)))
            _random.Random(seed + epoch).shuffle(order)     # epoch마다 다르되 재현가능
            batch_iter = ((s, my_clips[order[s]]) for s in range(steps_per_epoch))
        else:
            batch_iter = enumerate(loader)
        for s, item in batch_iter:
            if s >= steps_per_epoch:                         # rank별 step 수 차이 → 데드락 방지(공통 최소만 소비)
                break
            at_boundary = ((s + 1) % grad_accum == 0) or (s == steps_per_epoch - 1) or ((step + 1) >= total_steps)
            if args.temporal_clip:
                # clip 롤아웃: 프레임별 즉시 backward(grad_scale=1/grad_accum). loss는 로깅용 mean으로 재구성.
                st = clip_rollout_forward(vlm_mod, dit_mod, gate_mod, normalizer, processor, prompt_tpl,
                                          item, device, 1.0 / grad_accum, train_vlm=train_vlm, joint=joint,
                                          reasoning_types=reasoning_types, ego_dim=ego_dim, temporal_on=temporal_on,
                                          reas_weight=args.reas_weight, gate_weight=args.gate_weight,
                                          gate_alpha=gate_alpha, gate_focal_gamma=args.gate_focal_gamma,
                                          selective_view=args.selective_view, forcing=args.forcing)
                flow = st["flow_sum"] / max(st["n"], 1)
                lm_loss = (st["lm_sum"] / st["lm_n"]) if st["lm_n"] > 0 else None
                gate_ce = (st["gate_sum"] / st["gate_n"]) if st["gate_n"] > 0 else None
                loss = flow + (args.reas_weight * lm_loss if lm_loss is not None else 0.0) \
                            + (args.gate_weight * gate_ce if gate_ce is not None else 0.0)
                if at_boundary:                              # clip 경계: 수동 grad all-reduce 후 step
                    ddp.sync_grads(sync_params)
                    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                    opt.step()
                    if scheduler is not None:
                        scheduler.step()
                    opt.zero_grad(set_to_none=True)
            else:
                batch = item
                # 표준 DDP + grad accum: 비경계 step은 no_sync로 grad 로컬 누적, 경계 backward에서 all-reduce.
                use_no_sync = ddp.is_dist() and not at_boundary
                with (ddp_model.no_sync() if use_no_sync else nullcontext()):
                    loss, flow, lm_loss, gate_ce = ddp_model(batch)   # (total, flow(detached), LM_loss, gate_CE)
                    (loss / grad_accum).backward()           # 1/K 스케일 → K번 누적하면 K-배치 평균 grad
                loss = float(loss)
                if at_boundary:                              # 경계: DDP가 grad를 rank 간 평균 완료 → clip + step
                    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                    opt.step()
                    if scheduler is not None:                # 논문식 warmup+cosine: optimizer step 단위로 1회
                        scheduler.step()
                    opt.zero_grad(set_to_none=True)
            has_lm = 1.0 if lm_loss is not None else 0.0
            has_gate = 1.0 if gate_ce is not None else 0.0
            step += 1
            step_in_ep += 1
            # 로깅 집계: 모든 rank가 동일하게 호출(조건부 호출 금지 — collective 불일치 hang 방지).
            g_flow = ddp.all_reduce_mean(float(flow))
            g_lm_sum = ddp.all_reduce_mean(float(lm_loss) if lm_loss is not None else 0.0)
            g_has = ddp.all_reduce_mean(has_lm)              # reasoning 보유 rank 비율
            g_gate_sum = ddp.all_reduce_mean(float(gate_ce) if gate_ce is not None else 0.0)
            g_has_gate = ddp.all_reduce_mean(has_gate)       # gate 라벨 보유 rank 비율
            ep_loss_sum += g_flow; ep_n += 1                # epoch 평균은 flow 기준(전 rank 공통)
            live = {"loss": float(loss), "flow": g_flow}
            if g_has > 0:                                    # 이번 step에 reasoning 가진 rank가 하나라도
                live["lm"] = g_lm_sum / g_has               # 보유 rank들 평균 lm
            if g_has_gate > 0:                                # 이번 step에 gate 라벨 가진 rank가 하나라도
                live["gate"] = g_gate_sum / g_has_gate       # 보유 rank들 평균 gate_CE
            logger.step(epoch, step_in_ep, steps_per_epoch, **live)
            if step >= total_steps:
                done = True
                break

        # epoch 종합: 평균 train loss + (eval-every 주기면) val eval. 터미널 영구 줄 + 로그파일 기록.
        ep_avg = ep_loss_sum / max(ep_n, 1)
        ev = None
        if args.eval_every > 0 and (epoch % args.eval_every == 0 or done):
            # 분산 eval: 전 rank가 자기 val 샤드를 forward하고 내부에서 all_reduce로 집계 →
            # 모든 rank가 동일 collective를 호출하므로 별도 barrier 없이 동기화된다(데드락 없음).
            ev = evaluate(vlm_mod, dit_mod, processor, val_records, prompt_tpl,
                          normalizer, args.reas_weight, device, joint, temporal_on, reasoning_types,
                          maneuver_thr=args.maneuver_lateral_thr,
                          gate_mod=gate_mod, gate_weight=args.gate_weight, gate_alpha=gate_alpha,
                          gate_focal_gamma=args.gate_focal_gamma,
                          temporal_clip=args.temporal_clip, selective_view=args.selective_view,
                          forcing=args.forcing, keyframe_eval=args.keyframe_eval,
                          keyframe_select=keyframe_select)
        # best 갱신: val_loss가 이전 최소보다 낮으면 이 epoch을 best로 기록하고 가중치를 CPU 스냅샷.
        #   전 rank가 동일 ev["val_loss"]를 보므로 best_epoch 판단은 일치(스냅샷만 rank0).
        is_best = ev is not None and ev["val_loss"] < best_val
        if is_best:
            best_val = ev["val_loss"]
            best_epoch = epoch
            if main_proc:                                    # 저장·최종eval을 맡는 rank만 스냅샷 보관
                best_dit_sd = {k: v.detach().to("cpu", copy=True)
                               for k, v in dit_mod.state_dict().items()}
                if train_vlm:                                # VLM은 **학습 대상 파라미터만** 스냅샷
                    #   full=전체 가중치, lora=어댑터만. lora의 frozen 4-bit base를 CPU 복사하면
                    #   bitsandbytes Params4bit와 충돌·낭비 → requires_grad 파라미터만 담는다.
                    best_vlm_sd = {n: p.detach().to("cpu", copy=True)
                                   for n, p in vlm_mod.named_parameters() if p.requires_grad}
                if gate_mod is not None:
                    best_gate_sd = {k: v.detach().to("cpu", copy=True)
                                   for k, v in gate_mod.state_dict().items()}
        # epoch 요약에 best 표시(★ = 이번이 best). 로그파일에도 best 추적이 남는다.
        extra = f"total {step}/{total_steps}" + (f"  ★best(val {best_val:.4f})" if is_best else "")
        logger.epoch(epoch, ep_avg, eval=ev, extra=extra)  # 전체 진행률

    # 학습 종료: optimizer state(full FT는 2B분 → 수 GB)와 grad를 즉시 해제하고 캐시를 비운다.
    # ⚠️ 안 그러면 full 모드에서 GPU가 거의 꽉 찬 채로 저장·cleanup·final-eval에 들어가 OOM 난다
    #    (실제로 134254/134418가 학습 후 OOM). 저장/eval은 추론만 하므로 optimizer는 더 이상 불필요.
    opt.zero_grad(set_to_none=True)
    del opt
    for p in clip_params:
        p.grad = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5) 저장: rank0만 기록(가중치는 전 rank 동기화돼 동일). 나머지 rank는 완료까지 대기.
    #    모델(DiT+VLM+config)을 **실행 폴더 안 model/**(run_dir/model)에 저장 → 실행마다 모델이 보존돼
    #    과거 실행도 그대로 평가 가능(이전엔 models/trajectory에 덮어써서 과거 모델이 사라졌었음).
    #    --out을 명시하면 그 경로로(하위호환). 로그/run_info는 run_dir 직속.
    if main_proc:
        def _rel(p):                                       # REPO 하위면 상대경로, 아니면(예: /tmp) 절대경로
            try:
                return str(Path(p).relative_to(REPO))
            except ValueError:
                return str(p)
        # best 가중치 복원: 검증을 했다면(스냅샷 존재) val_loss 최소 epoch의 가중치를 in-place 로드
        #   → 이후 저장·최종 eval이 **best 모델**로 수행된다. load_state_dict는 기존 CUDA 파라미터에
        #   값만 복사(새 할당 없음 → GPU 안전). 검증 안 했으면(--eval-every 0) 마지막 epoch 모델 사용.
        if best_dit_sd is not None:
            dit_mod.load_state_dict(best_dit_sd)
            if train_vlm and best_vlm_sd is not None:
                vlm_mod.load_state_dict(best_vlm_sd, strict=False)  # 학습 파라미터만 제공 → strict=False
            if gate_mod is not None and best_gate_sd is not None:
                gate_mod.load_state_dict(best_gate_sd)
            logger.info(f"best model = epoch {best_epoch} (val_loss {best_val:.4f}) — 이 가중치로 저장+최종 eval")
        else:
            logger.info("best 스냅샷 없음(--eval-every 0) — 마지막 epoch 모델로 저장+최종 eval")

        out = Path(args.out) if args.out else (logger.run_dir / "model")
        out.mkdir(parents=True, exist_ok=True)
        torch.save(dit_mod.state_dict(), out / "dit_head.pt")
        if gate_mod is not None:
            torch.save(gate_mod.state_dict(), out / "gate_head.pt")
        if train_vlm:
            vlm_mod.save_pretrained(str(out / "vlm"))      # full=전체 가중치, lora=어댑터만
            processor.save_pretrained(str(out / "vlm"))
        cfg = {
            "model": args.model, "vlm_mode": args.vlm_mode,
            "n_points": n_points, "point_dim": point_dim, "cond_dim": cond_dim, "ego_dim": ego_dim,
            "ego_as_state_token": args.ego_state_token,    # ego를 state token으로 썼는지(평가 시 DiT 재구성용)
            # flow-matching t~Beta(α,β) (학습 전용, sample()엔 무영향이나 재구성 일관 위해 기록).
            "beta_alpha": args.beta_alpha, "beta_beta": args.beta_beta,
            "ode_steps": args.ode_steps,                   # 추론 K(논문=5). final_eval/eval_trajectory가 이 값 사용.
            "cross_attn": args.cross_attn,                 # DiT가 VLM 시퀀스에 cross-attention 했는지(평가 재구성용).
            "maneuver_lateral_thr": args.maneuver_lateral_thr,  # (구형) 미래GT 게이팅 thr. 평가도 동일 적용.
            "temporal_clip": args.temporal_clip,           # clip 시퀀스 폐루프 여부(평가도 clip 순차 폐루프).
            "selective_view": args.selective_view,         # 게이트 예측으로 뷰 게이팅 여부(폐루프).
            "forcing": args.forcing,                       # 학습 뷰 게이팅 출처(student=폐루프예측/teacher=GT). 추론은 항상 student.
            "keyframe_eval": args.keyframe_eval,           # 검증/최종평가를 keyframe(decision, 0.2Hz)만 채점(논문 정렬).
            "keyframe_select": list(keyframe_select),      # clip당 decision 3개 중 채점 순서(기본 [1]=정중앙).
            "seed": seed,                                  # 랜덤 시드(--seed override, 기본 vlm.SEED).
            # 1단계 게이트(cond→직진/좌/우): μ(gate_weight)>0이면 GateHead 학습(gate_head.pt 저장됨).
            "gate_weight": args.gate_weight, "gate_hidden": args.gate_hidden,
            "gate_focal_gamma": args.gate_focal_gamma,
            "gate_alpha": gate_alpha.tolist() if gate_alpha is not None else None,
            # 시간 맥락: 학습이 실제로 과거 1 timestep(8뷰)을 condition에 넣었는지 + 그 offset(프레임).
            # 평가 스크립트가 이 값을 읽어 동일하게 처리 → 빌드·학습·평가 일관.
            "temporal": temporal_on, "history_offset": TEMPORAL_HISTORY_OFFSET,
            "d_model": args.d_model, "n_layers": args.n_layers,
            "normalizer": normalizer.state_dict(),
            "prompt": TRAJ_PROMPT_FILE.name, "steps": step, "lr": args.lr, "vlm_lr": args.vlm_lr,
            "optimizer": args.optimizer,
            "batch_size": args.batch_size, "grad_accum": grad_accum, "eff_batch": eff_batch,
            "reas_weight": args.reas_weight, "n_reasoning_samples": n_reas,
            # 단계별 ablation: 이 실행이 LM_loss로 supervision한 reasoning 종류(baseline이면 []).
            "reasoning_types": reasoning_types,
            # 논문 정렬: reasoning 주석 프레임만 학습/평가했는지. eval_trajectory.py가 읽어 동일 필터 적용.
            "reasoning_only": args.reasoning_only,
            "world_size": ddp.world_size(), "run_dir": _rel(logger.run_dir),
            # best 모델 추적: val_loss 최소 epoch과 그 값(없으면 마지막 epoch 사용 → -1/None).
            "best_epoch": best_epoch, "best_val_loss": (round(best_val, 4) if best_dit_sd is not None else None),
        }
        (out / "traj_config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
        # 실행 폴더엔 설정 스냅샷 + 모델 위치 포인터(나중에 어떤 실행이 어떤 모델을 냈는지 추적).
        (logger.run_dir / "run_info.json").write_text(json.dumps(
            {"tag": "traj_reas", "args": vars(args), "steps": step,
             "model_out": _rel(out)}, ensure_ascii=False, indent=2))
        logger.info(f"saved DiT head" + (" + VLM" if train_vlm else "") + f" + config -> {out}")
        logger.info(f"run logs -> {_rel(logger.run_dir)}")

        # 6) 최종 자동 평가(rank0): 이미 메모리의 모델 재사용 → ADE/FDE + BEV 시각화/reasoning을 run_dir에 저장.
        #    다른 rank는 아래 ddp.cleanup()의 barrier에서 대기(eval은 collective 없음).
        if args.final_eval:
            logger.info("running final eval (ADE/FDE + BEV) ...")
            try:
                fm = run_final_eval(vlm_mod, dit_mod, processor, normalizer, cfg, prompt_tpl,
                                    device, logger.run_dir, args.val, args.final_eval_limit,
                                    viz_n=args.final_eval_viz, per_page=args.final_eval_per_page,
                                    gate_mod=gate_mod)      # 게이트 있으면 폐루프 추론(예측→뷰게이팅→planning)
                if fm:
                    # ADE/FDE(m, 거리지표)는 소수 셋째자리 그대로. NC/DA/EP/CF/HL/NPS(0~1 저장값)는
                    # 표시 시에만 ×100(논문 %표기와 동일 스케일) + 소수 둘째자리로 변환(저장값 자체는 불변).
                    logger.info(f"final eval | ADE {fm['ADE_mean']:.3f}m FDE {fm['FDE_mean']:.3f}m "
                                f"(cv {fm['ADE_baseline_cv']:.3f}/{fm['FDE_baseline_cv']:.3f}) "
                                f"-> {_rel(logger.run_dir)}")
                    # Table 3 planning 점수(높을수록 좋음; NPS=종합). 로그파일에도 남긴다.
                    logger.info("final eval | Table3 "
                                f"NC {fm.get('NC_mean', float('nan')) * 100:.2f} "
                                f"DA {fm.get('DA_mean', float('nan')) * 100:.2f} "
                                f"EP {fm.get('EP_mean', float('nan')) * 100:.2f} "
                                f"CF {fm.get('CF_mean', float('nan')) * 100:.2f} "
                                f"HL {fm.get('HL_mean', float('nan')) * 100:.2f} "
                                f"NPS {fm.get('NPS_mean', float('nan')) * 100:.2f}")
                    # 추론 속도(ms/sample, VLM encode+DiT sample, 워밍업 제외) — 콘솔뿐 아니라 train.log에도 남김.
                    sp = fm.get("inference_speed_ms") or {}
                    if sp.get("n_timed"):
                        logger.info(f"final eval | speed(ms) n={sp['n_timed']} "
                                    f"fastest={sp['fastest_ms']:.2f} slowest={sp['slowest_ms']:.2f} "
                                    f"median={sp['median_ms']:.2f} mean={sp['mean_ms']:.2f} "
                                    f"(thr={fm.get('maneuver_lateral_thr')})")
                    if fm.get("gate_closedloop"):             # 폐루프 게이트: 예측→뷰게이팅→planning
                        logger.info(f"final eval | gate closed-loop | pred_acc={fm.get('gate_pred_acc')} "
                                    "(게이트 방향 예측 → 그 뷰로 planning)")
            except Exception as e:                            # eval 실패해도 학습 산출물은 보존
                logger.info(f"final eval 실패(건너뜀): {type(e).__name__}: {e}")
    logger.close()
    ddp.cleanup()


if __name__ == "__main__":
    main()
