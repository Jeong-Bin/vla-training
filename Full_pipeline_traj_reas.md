# 학습 파이프라인 — objective=trajectory (flow-matching DiT)

이 문서는 **궤적 planning 경로**(`train_traj_reas.py`)를 다룬다. 8뷰 이미지를 받아 **ego 차량의 미래
궤적(연속 waypoints)**을 생성하고, **동시에 reasoning을 함께 supervise**한다(논문 nuVLA의 "궤적 +
reasoning 공동 학습" 재현). 출력이 텍스트 토큰이 아니라 연속 좌표열이라, 텍스트 SFT와 학습 메커니즘이
근본적으로 다르다 — VLM 백본 위에 **flow-matching DiT 헤드**를 올린다(논문 nuVLA 구조의 핵심).

> 🧭 **궤적 + reasoning 공동 supervision:** 손실은 `total = flow_loss + λ·LM_loss(reasoning)`이며,
> VLM은 `--vlm-mode {full(기본)·lora·frozen}`으로 학습 범위를 고른다. 자세한 내용은 ⑤ 절을 보라.

> 🔀 **objective 분기:** 학습 경로는 둘이다. 이 문서는 그중 trajectory. 텍스트 결정/인식 경로는
> [Full_pipeline_text_sft.md](Full_pipeline_text_sft.md)를 보라. 분기 개요·공통 단계는
> [Full_pipeline.md](Full_pipeline.md)(허브)에 있다.
>
> 데이터셋 자체(구조·궤적 주석·정식 논문 대비)는 [DATASET_INFO.md](DATASET_INFO.md) 참고.

## 왜 별도 경로인가 — text_sft와의 근본적 차이

| | text_sft | **trajectory (이 문서)** |
|---|---|---|
| 출력 | 이산 라벨/객체 텍스트 | **연속 궤적** (N×2 [fwd,left]) **+ reasoning** |
| 헤드 | LM head (다음 토큰 분류) | **flow-matching DiT** (좌표 회귀) **+ LM head**(reasoning) |
| 손실 | cross-entropy | **`flow_loss + λ·LM_loss`** (rectified-flow MSE + reasoning CE) |
| 학습 루프 | HF `Trainer` | **커스텀 루프** |
| VLM | QLoRA로 학습 | **`--vlm-mode` full(기본)·lora·frozen** (full=VLM+DiT 공동학습) |
| GT 출처 | `reasoning/<ts>.json` | **`ego_state.trajectory_future`** (+ reasoning_trace) |
| 평가 | accuracy / macro-F1 | **ADE / FDE** (미터) |

연속 궤적 회귀에는 LM head·CE가 부적합하므로 DiT 헤드와 커스텀 루프가 필요하다. 그래서 `if/else`가 아니라
**별도 모듈**로 분리했다. 단, reasoning supervision은 같은 VLM의 LM head로 동시에 학습한다(⑤ 참고).

## 전체 파이프라인 한눈에 보기

```
┌─────────────────────────────────────────────────────────────────────┐
│ ① 원본 nuReasoning 클립          /home/etri/DATASET/nureasoning/clips/<clip>/              │
│    ├─ cameras/CAM_*/*.jpg   (8뷰 카메라 이미지)                       │
│    └─ ego_state/*.pkl  →  trajectory_future (50×3) [x,y,yaw] global! │
└─────────────────────────────────────────────────────────────────────┘
                          │  parse_clip()  [nureasoning/loader.py]
                          ▼   경로만 들고 lazy Frame 생성
┌─────────────────────────────────────────────────────────────────────┐
│ ② Clip → Frame 객체   (ego_state는 첫 접근 시 lazy 로드)             │
└─────────────────────────────────────────────────────────────────────┘
                          │  select_keyframes(policy="spatial")  (이미지 보유 프레임)
                          ▼   trajectory_future 있는 프레임(로컬 80%)
┌─────────────────────────────────────────────────────────────────────┐
│ ③ 키프레임 선별 + 좌표 변환                                          │
│    global UTM (x,y,yaw)  ──future_waypoints_ego()──▶  ego-frame      │
│    [geometry.py]  fwd=전방+, left=좌측+  (center_3d_ego와 동일 컨벤션)│
└─────────────────────────────────────────────────────────────────────┘
                          │  frame_to_trajectory_sft()  [build_sft.py]
                          ▼   고정 길이(50) 미달 프레임 제외
┌─────────────────────────────────────────────────────────────────────┐
│ ④ trajectory "내용" 레코드   →   data/sft/train.jsonl / val.jsonl    │
│    {id, images(8뷰), mission, task:"trajectory",                     │
│     output{waypoints:[[fwd,left] × 50], reasoning?}}                 │
│    (reasoning은 0.2Hz라 ~23% 프레임에만 존재 → 있으면 함께 저장)     │
│    ⚠ train/val은 clip_token 기준 분리(text_sft와 동일 split)         │
└─────────────────────────────────────────────────────────────────────┘
                          │  encode_and_lm_loss()  [train_traj_reas.py]
                          ▼   VLM 8뷰+프롬프트(+reasoning) → 1 forward
┌─────────────────────────────────────────────────────────────────────┐
│ ⑤ 학습 단계   [training/train_traj_reas.py]                          │
│    VLM(--vlm-mode full/lora/frozen) forward(이미지+프롬프트+reasoning)│
│         ├─ 프롬프트까지 hidden mean-pool → condition (B,2048)        │
│         └─ reasoning 토큰에 LM_loss (프롬프트는 -100 마스킹)         │
│         │                                                            │
│    flow = TrajectoryDiT.flow_loss(정규화 waypoints, condition)       │
│    total = flow + λ·LM_loss(reasoning)   ← 공동 손실 (λ=--reas-weight)│
│         │  ← 커스텀 루프(full/lora면 VLM+DiT 함께 backward)          │
│         ▼                                                            │
│    models/trajectory/  (dit_head.pt + traj_config.json + vlm/)       │
└─────────────────────────────────────────────────────────────────────┘
```

## 단계별 요약표

| 단계 | 하는 일 | 코드 위치 | 핵심 포인트 |
|---|---|---|---|
| ① 원본 | 8뷰 이미지 + ego_state | `/home/etri/DATASET/nureasoning/clips/` | **`trajectory_future` (50,3)=[x,y,yaw] global UTM**. 로컬 프레임 **80%(1665개)** 보유 — driving decision(33개)보다 훨씬 풍부 |
| ② 파싱 | 폴더 → `Clip`/`Frame` | `parse_clip` (`loader.py`) | 경로만 들고 lazy 로드 |
| ③ 선별+변환 | 프레임 → ego-frame waypoints | `future_waypoints_ego` (`nureasoning/geometry.py`) | global UTM → ego-frame 회전·평행이동. fwd=+x 전방, left=+y 좌측 |
| ④ 레코드 생성 | 프레임 → 내용 레코드 + split | `frame_to_trajectory_sft` (`build_sft.py`) | 고정 길이 50 보장(미달 제외). reasoning 있으면 `output.reasoning`에 포함. clip 단위 누수 방지(text_sft와 동일) |
| ⑤ 학습 | VLM condition → DiT flow-matching **+ reasoning LM_loss** | `train_traj_reas.py` + `dit_head.py` | `--vlm-mode`(full 기본·lora·frozen). 공동 손실 `flow+λ·LM`. 커스텀 루프 |

## ③ 좌표 변환 — 왜 필요한가

`trajectory_future`는 **global UTM 좌표**(예: x≈664472, y≈3996658 — 지구 좌표계)다. 모델이 학습할 수
있는 형태는 **ego 차량 기준 상대 좌표**이므로 변환이 필수다.

```python
# nureasoning/geometry.py
fwd  =  dx*cos(yaw) + dy*sin(yaw)   # 전방 거리(+)
left = -dx*sin(yaw) + dy*cos(yaw)   # 좌측 거리(+)   (dx,dy = waypoint - ego위치)
```

**검증(2026-06-23):** 변환 후 wp[0]≈(0.7, 0)으로 ego 원점 근처에서 시작, fwd 단조 증가, 50 waypoint
총거리 44m가 velocity 6.84 m/s × ~6.5초와 일관. 컨벤션은 spatial 태스크의 `center_3d_ego`와 동일(fwd=+x, left=+y).

## ⑤ flow-matching DiT 헤드 (`dit_head.py`)

논문 nuVLA가 VLM 위에 올린 flow-matching DiT 궤적 헤드를 **미니 규모로 재현**한다.

### rectified flow (직선 보간 flow-matching)

```
x0 ~ N(0, I)                      # 가우시안 노이즈
x1 = 정규화된 GT waypoints         # 목표 궤적
xt = (1−t)·x0 + t·x1   (t∈[0,1])  # 직선 보간
target velocity v = x1 − x0       # 상수 속도장
loss = MSE( DiT(xt, t, cond),  v )
```

추론은 x0~N(0,I)에서 시작해 **Euler ODE 적분**(steps번 `x += v̂·dt`)으로 궤적을 생성한다.

### 구조 (작은 규모, 단일 24GB GPU)

- 입력: noised waypoints (B,N,2) + flow time t (B,) + **VLM condition (B,2048)**
- waypoint 임베딩 + 순서 임베딩 → **AdaLN-Zero** 조건화 transformer 블록 × 4 → velocity (B,N,2)
- **AdaLN-Zero:** gate·final layer를 0으로 초기화 → 학습 초반엔 출력이 정확히 0(identity 시작) → 안정적 수렴
- **TrajectoryNormalizer:** waypoints를 평균0·표준편차1로 정규화(flow-matching 안정화). 평가 시 역정규화
- 파라미터 **5.6M** (VLM 2B 대비 매우 작음)

## ⑤ 궤적 + reasoning 공동 supervision (`train_traj_reas.py`)

논문의 핵심은 **궤적 회귀와 reasoning 생성을 한 VLM에서 함께 학습**하는 것이다. 우리도 이를 그대로 재현한다.

```
8뷰 이미지 + planning 프롬프트 (+ reasoning_trace, 있으면)
   │  processor.apply_chat_template (뷰 캡션 인터리브)
   ▼
VLM(Qwen3-VL-2B, --vlm-mode) forward(이미지+프롬프트+reasoning, output_hidden_states=True)
   ├─ (a) 프롬프트까지 hidden을 attention mask로 mean-pool → condition (B,2048)
   └─ (b) reasoning 토큰에 LM_loss(CE).  프롬프트 토큰은 labels=-100으로 마스킹
   │
   ├─ flow  = TrajectoryDiT.flow_loss(normalize(waypoints), condition)   [dit_head.py]
   └─ total = flow + λ·LM_loss(reasoning)        (λ = --reas-weight, 기본 0.5)
   ▼
total.backward()  →  optimizer.step()
   ← full/lora면 VLM+DiT 함께 갱신 / frozen이면 DiT만
```

### "한 forward" 트릭 — 학습 condition == 평가 condition

Qwen-VL은 **causal decoder**라 "이미지+프롬프트"의 hidden state는 뒤에 reasoning이 붙든 말든 **불변**이다
(미래를 엿보지 않음). 이를 이용해 **단 1회 forward**(이미지+프롬프트+reasoning)에서 두 가지를 동시에 얻는다:

- **condition**: `attention_mask`를 프롬프트 경계(`plen`) 이후로 0 처리한 뒤 mean-pool → 프롬프트까지만 반영
- **LM_loss**: `labels`의 프롬프트 구간(`:plen`)을 −100으로 마스킹 → reasoning 토큰에만 CE

이렇게 하면 평가 시(이미지+프롬프트만 입력)와 **정확히 같은 condition**이 보장된다(`encode_condition`도
`add_generation_prompt=True`로 통일). reasoning이 없는 샘플(~77%)은 `flow_loss`만, 있는 샘플(~23%)만
`LM_loss`를 더한다.

### `--vlm-mode` — VLM 학습 범위 (논문 충실도 ↔ 자원)

| 모드 | VLM | reasoning supervision | 자원 메모 |
|---|---|---|---|
| **full** (기본, 논문 충실) | 전체(2.1B) + DiT 공동학습 | ✅ 유효 | gradient checkpointing + PagedAdamW8bit + batch=1로 단일 24GB 통과(실증) |
| **lora** | QLoRA(17M) + DiT | ✅ 유효 | 24GB 안전 마진. DiT는 `--lr`, VLM은 `--vlm-lr` 별도 |
| **frozen** | DiT만 | ⚠️ 무의미(VLM에 grad 미전달) | 가장 가벼움. reasoning은 무시됨 |

> ⚠️ **frozen에서는 reasoning supervision이 동작하지 않는다** — VLM 파라미터로 grad가 흐르지 않아 LM_loss를
> 더해도 학습되는 게 없다. 논문의 "궤적+reasoning 공동"을 의도하면 **full(기본) 또는 lora**를 써라.

학습된 VLM 가중치는 `models/trajectory/vlm/`에 저장되고, 평가가 `traj_config.json`의 `vlm_mode`를 보고
복원한다(full=전체 로드, lora=베이스+어댑터, frozen=베이스만). HF `Trainer`는 LM head/CE 전제라 연속 궤적
회귀와 공동 손실에 부적합 → **커스텀 루프**를 직접 돈다.

### 로그의 손실 항 읽기 — `flow` / `lm` / `loss`

학습 로그(실시간 step 줄, epoch 종합의 `EVAL`)에 찍히는 세 값의 의미:

| 항목 | 무엇을 재나 | 손실 종류 | 평균 분모 |
|---|---|---|---|
| **flow** | DiT의 **미래 궤적 예측** 정확도 (메인 목표) | rectified-flow MSE | 전체 trajectory 샘플 |
| **lm** | VLM의 **reasoning 텍스트 생성** 정확도 (보조 supervision) | language-modeling CE | **reasoning 보유 샘플만(~23%)** |
| **loss** | 공동 손실 `flow + λ·lm` (λ=`--reas-weight`, 기본 0.5) | 합성 | 전체 샘플 |

- **낮을수록 좋다.** `flow ↓`= 궤적을 더 정확히 예측(핵심 지표), `lm ↓`= 장면 reasoning을 더 잘 설명.
- epoch 종합 줄의 `val_*`(예: `val_flow`, `val_lm`)는 **검증(val) 셋** 기준 같은 지표다(실시간 `flow`/`lm`은
  학습 step 기준).

> ⚠️ **`loss ≠ flow + 0.5·lm`로 안 맞아 보이는 이유**(분모가 다름): `lm`은 reasoning 보유 샘플(~23%)로만
> 평균 내지만, `loss`는 전체 샘플 평균이라 lm 없는 77% 샘플에는 flow만 더해진다. 즉 정확한 관계는
> `loss = flow + 0.5·(lm_sum / 전체샘플수)` 이고, `lm = lm_sum / reasoning샘플수`(다른 분모)다. 검산:
> reasoning≈23%면 `loss ≈ flow + 0.5·(0.23·lm)`. (예: `val_flow 0.90, val_lm 1.15` → `0.90 + 0.5·0.23·1.15 ≈ 1.03` = `val_loss`.)

## 평가 — ADE / FDE (`eval_trajectory.py`)

text_sft의 정확도/F1에 대응하는 궤적 표준 지표.

| 지표 | 정의 |
|---|---|
| **ADE** (Average Displacement Error) | 전 waypoint의 예측↔GT L2 거리 평균 (미터) |
| **FDE** (Final Displacement Error) | 마지막 waypoint의 L2 거리 (미터) |
| **constant-velocity baseline** | GT 첫 변위를 등속 연장한 naive 궤적의 ADE/FDE (비교 기준) |

학습된 VLM(`vlm_mode`로 복원)+DiT로 미래 궤적을 ODE 샘플링 → 역정규화 → GT와 비교. (평가는 궤적 지표만
보므로 reasoning 생성 품질은 별도로 다루지 않는다 — text_sft 경로의 reasoning 평가와 구분.)

## 파이프라인 동작 검증 (smoke test, 2026-06-23)

전 경로가 **OOM 없이 통과**함을 작은 데이터로 확인:

| 항목 | 결과 |
|---|---|
| VLM 로드 (Qwen3-VL-2B, full/lora/frozen) | ✅ |
| condition dim (probe) | ✅ **2048** |
| DiT params | ✅ 5,601,538 |
| 공동 손실 분해 (lora, reasoning 4/4) | ✅ `loss 3.08 = flow 1.88 + 0.5×lm 2.40` (합산 정확) |
| reasoning LM_loss (CE) | ✅ reasoning 샘플에만 동작 |
| VLM+DiT 공동 backward | ✅ lora(17M)+DiT(5.6M) 동시 갱신 |
| 저장 | ✅ `dit_head.pt`(22MB) + `traj_config.json` + `vlm/` |
| eval 복원(vlm_mode) + ADE/FDE + cv baseline | ✅ 산출·저장 |

> ⚠️ smoke는 **소수 step만** 학습했으므로 ADE가 baseline보다 나쁜 게 정상이다(DiT 거의 미학습). 목적은
> **성능이 아니라 파이프라인 동작 검증**. 의미 있는 ADE/FDE는 본 학습(충분한 epoch) 후 나온다.

## 실제로 돌리는 순서

```bash
# (사전) 클립 다운로드 → /home/etri/DATASET/nureasoning/clips/
python scripts/download_clips.py

# ④ SFT 데이터 생성 → data/sft/{train,val}.jsonl (trajectory 샘플 포함, reasoning 동반)
PYTHONPATH=src python -m sft_data.build_sft --clips-root "$PWD//home/etri/DATASET/nureasoning/clips"

# ⑤ 궤적+reasoning 공동 학습 → models/trajectory/  (full=논문 충실, 단일 24GB 통과)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m training.train_traj_reas \
    --vlm-mode full --reas-weight 0.5
#   ↳ 24GB 안전 마진을 원하면: --vlm-mode lora

# 평가(ADE/FDE) → results/trajectory_metrics.json  (vlm_mode 자동 복원)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluation.eval_trajectory \
    --adapter models/trajectory
```

## 논문 대비 정직성

| 요소 | 논문 nuVLA | 우리 trajectory 경로 |
|---|---|---|
| VLM 백본 | Qwen3-VL-2B | ✅ 동일 (Qwen3-VL-2B) |
| 궤적 헤드 | flow-matching DiT | ✅ 재현(소형, 5.6M) |
| VLM 학습 | full fine-tune | ✅ **full**(기본, VLM+DiT 공동학습) — lora·frozen도 선택 가능 |
| reasoning supervision | 궤적과 공동 학습 | ✅ 재현 (`flow_loss + λ·LM_loss`, 한 forward) |
| 궤적 길이 | (논문 설정) | 50 waypoint (~6.5초) |
| 데이터 규모 | 대규모 | ⚠️ 미니(수백~수천) — 8뷰·주석 프레임만 |
| 안전 게이트 | NPS(충돌·주행가능영역) | 미구현 (ADE/FDE만) |

즉 **VLM full fine-tune + DiT 궤적 + reasoning 공동 supervision**이라는 논문의 핵심 학습 구조를 재현한다.
남은 차이는 데이터 규모(미니)와 NPS 안전 게이트 미구현뿐이다.
([DECISIONS.md](DECISIONS.md) "학습 objective 분기" 절 참고.)

## 한 줄 요약

원본 클립 → 파싱 → **trajectory_future를 ego-frame으로 변환** → trajectory(+reasoning) 레코드 → VLM
condition + reasoning LM_loss → **flow-matching DiT 학습(공동 손실 `flow+λ·LM`)** → ADE/FDE 평가 순서이며,
③ 변환은 `geometry.py`, ④는 `build_sft.py`, DiT는 `dit_head.py`, ⑤ 공동 학습은 `train_traj_reas.py`,
평가는 `eval_trajectory.py`가 담당한다. VLM 학습 범위는 `--vlm-mode`(full 기본·lora·frozen)로 고른다.
텍스트 결정/인식 경로는 [Full_pipeline_text_sft.md](Full_pipeline_text_sft.md)를 보라.
