"""VLA SFT 데이터 생성 (멀티태스크: driving + spatial + trajectory).

nuReasoning 클립을 세 종류의 학습 샘플로 변환한다 — 논문 nuVLA의 "1Hz spatial supervision +
0.2Hz decision supervision + 궤적 planning"을 미니 규모로 재현. 태스크별 프레임 밀도가 다르다:
  - **driving** (클립당 ~3, Driving decision 있는 1Hz 프레임): (8뷰) → (longitudinal/lateral 결정 + reasoning)
  - **spatial** (클립당 ~13, 객체 주석 있는 1Hz 프레임): (8뷰) → (가까운 객체들의 category + ego-frame 위치)
  - **trajectory** (**10Hz 전 프레임을 --traj-stride로**, 기본 stride 1 = 클립당 ~150): (8뷰) →
    (ego-frame 미래 waypoints N×2) ← DiT planning 학습용. 10Hz 전체 이미지를 받았으므로 프레임마다 생성 가능.
입력 8뷰 surround는 공통이고 디스크에 존재하는 뷰만 담되 front는 필수다. task 필드로 셋을 구분한다.
driving/spatial은 텍스트 타깃(chat_format 어댑터가 렌더), trajectory는 수치 waypoints 타깃(DiT 헤드가 소비).
→ 학습은 `--objective`로 분기: text_sft(driving+spatial) | trajectory(DiT). 한 JSONL에 task로 공존.

이 모듈은 **데이터 내용**만 만든다(원본 GT 텍스트를 그대로 보존). 모델 학습 포맷(Qwen-VL chat)으로
바꾸는 일은 `chat_format.py` 어댑터가 담당한다 — 스펙 §7.4의 "내용과 형식 분리".

train/val 분리(누수 방지):
  val = 평가셋(`zeroshot_manifest.jsonl`)에 들어간 클립들(clip_token로 식별).
  train = 그 외 모든 클립. → 파인튜닝은 val에서 zero-shot 대비 개선을 측정하므로, SFT train이
  val 클립을 포함하면 누수가 된다. clip 단위로 가르므로 한 클립이 양쪽에 걸치지 않는다.

완료 기준(스펙 §7): data/sft/{train,val}.jsonl + 샘플수·스킵률·라벨분포 리포트.

Run:  python -m sft_data.build_sft [--clips-root data/raw]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nureasoning import (  # noqa: E402
    CAMERA_VIEWS,
    SFT_DIR,
    TEMPORAL,
    TEMPORAL_HISTORY_OFFSET,
    Clip,
    Frame,
    future_waypoints_ego,
    iter_clips,
    past_waypoints_ego,
    select_keyframes,
)
from evaluation import taxonomy as T  # noqa: E402

# 클립 데이터 루트. 이제 대용량 10Hz 데이터를 REPO 밖 큰 디스크(/home/etri/DATASET/nureasoning/clips)에
# 두므로 그 경로를 기본값으로 한다(--clips-root로 오버라이드 가능). 이미지 경로는 절대경로로 저장하므로
# 클립이 REPO 안/밖 어디에 있든 relative_to(REPO) 실패 없이 학습·평가가 그대로 로드한다.
DEFAULT_CLIPS_ROOT = Path("/home/etri/DATASET/nureasoning/clips")
VAL_MANIFEST = REPO / "data" / "eval" / "zeroshot_manifest.jsonl"
OUT_DIR = SFT_DIR     # 기본 출력 = vlm.SFT_DIR(중앙 설정, 현재 data/sft_v2). --out으로 override.

# 스펙 §7 예시의 instruction(미션 + 간단 질의). 사람이 읽는 content 레벨 문구이며,
# 실제 학습에 쓰는 프롬프트는 chat_format 어댑터가 평가와 동일한 템플릿으로 렌더한다.
INSTRUCTION = ("Mission: {mission}. Given the scene, what longitudinal and lateral "
               "driving action should the ego take, and why?")

# spatial supervision 타깃에 담을 최대 객체 수(가까운 순). 8뷰 합 ~54개라 타깃이 길어지므로
# 캡해 시퀀스 길이를 통제(논문 A의 1Hz spatial supervision을 미니 규모로 재현).
SPATIAL_MAX_OBJECTS = 15

# 궤적 planning 타깃 스펙(논문 nuVLA action expert 정렬): T=10 waypoints @ 2Hz, 5초 horizon, (x,y,θ) ego-frame.
#   원본 trajectory_future는 (50,3)[x,y,yaw] @10Hz라, TRAJ_STRIDE_2HZ=5로 서브샘플해 미래프레임 +5,+10,…,+50
#   (=0.5,1.0,…,5.0s)에서 10개를 취하고, heading θ를 3번째 채널로 유지한다. DiT는 고정 길이가 필요하므로
#   5초 미래가 확보 안 되는 클립 끝 근처 프레임(짧은 궤적)은 제외한다.
TRAJ_N_POINTS = 10                                          # 논문 T=10 (2Hz × 5s)
TRAJ_STRIDE_2HZ = 5                                         # 10Hz 원본을 2Hz로 서브샘플(5번째마다)
# ego 이동 history(논문 nuVLA state token 구성): 과거 ego pose를 2Hz로 T_H개 → ego_state 뒤에 flatten.
# 6 = 과거 3초 @2Hz(trajectory_history 30@10Hz 전량). 0이면 history 없이 dynamics만(하위호환).
TRAJ_HISTORY_N = 6


# val_clip_tokens: 평가셋 manifest에서 val로 고정된 클립 토큰 집합을 읽음(누수 방지 기준).
def val_clip_tokens(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    return {json.loads(l)["clip_token"]
            for l in manifest_path.read_text().splitlines() if l.strip()}


# frame_to_sft: 한 프레임을 SFT 레코드로 변환. 결격(결정/필드/이미지 누락)이면 (None, 사유) 반환.
def frame_to_sft(f: Frame) -> tuple[Optional[dict], Optional[str]]:
    dec = f.driving_decision()
    if not dec:
        return None, "no_decision"
    lon, lat = (dec.get("Longitudinal") or "").strip(), (dec.get("Lateral") or "").strip()
    if not lon or not lat:
        return None, "empty_decision_field"
    reasoning = (f.reasoning_trace() or "").strip()
    if not reasoning:
        return None, "no_reasoning"
    if not f.has_camera("front"):
        return None, "no_front_image"
    # 디스크에 실제로 존재하는 뷰만 CAMERA_VIEWS 순서로 수집(front는 위에서 보장). 결측 뷰는 건너뜀.
    # 절대경로로 저장 → 클립이 REPO 밖(/home/etri/DATASET)이어도 학습·평가가 그대로 로드(resolve_path).
    images = [{"view": v, "path": str(f.camera_path(v))}
              for v in CAMERA_VIEWS if f.has_camera(v)]
    return {
        "id": f.sample_id,
        "clip_token": f.clip_token,
        "task": "driving",                           # 행동 결정 태스크
        "images": images,                            # [{view, path(레포 루트 기준 상대경로)}] 순서 보존
        "mission": f.mission_command,
        "instruction": INSTRUCTION.format(mission=f.mission_command or "drive safely"),
        "output": {"longitudinal": lon, "lateral": lat, "reasoning": reasoning},
    }, None


# _frame_images: 디스크에 존재하는 뷰만 CAMERA_VIEWS 순서로 [{view,path(절대경로)}] 수집(공용).
#   절대경로 저장 → 클립이 REPO 밖이어도 소비측(resolve_path)이 그대로 로드.
def _frame_images(f: Frame) -> list[dict]:
    return [{"view": v, "path": str(f.camera_path(v))}
            for v in CAMERA_VIEWS if f.has_camera(v)]


# _history_images: vlm.TEMPORAL이 켜졌을 때 현재 키프레임 f에서 TEMPORAL_HISTORY_OFFSET 프레임 전
# (같은 클립)의 8뷰를 [{view,path}]로 수집해 반환(논문 nuVLA의 "과거 1 timestep" 재현). 없으면 None.
#   - offset 적용은 클립 프레임 배열(10Hz 전 프레임) 인덱스로 한다(f.index - offset). 클립 시작 이전이면 None.
#   - ✅ 이제 10Hz 전체 이미지(--policy all)를 받으므로 **어떤 offset이든 과거 프레임에 이미지가 있다**
#     (과거 1Hz-only 제약 없음). offset=10=1.0초 전, 5=0.5초 전 등 임의 정수 가능. has_camera 가드는
#     혹시 그 뷰가 결측이거나 1Hz-only 데이터로 빌드하는 경우를 위한 안전장치로 남겨둔다.
def _history_images(clip: Clip, f: Frame) -> Optional[list[dict]]:
    if not TEMPORAL:
        return None
    j = f.index - TEMPORAL_HISTORY_OFFSET
    if j < 0:                                   # 클립 시작 이전 → 과거 프레임 없음
        return None
    hf = clip.frames[j]
    if not hf.has_camera("front"):              # 과거 프레임 이미지 결측(10Hz 전체면 거의 안 걸림) → 안전 가드
        return None
    return _frame_images(hf)


# _nearest_objects: 8뷰 GT 객체(Spatial.per_camera_results)를 track_token으로 중복 제거 + ego 거리순
#   정렬해 가까운 max_n개를 {category,dist_m,fwd_m,left_m}로 반환(spatial 태스크·spatial reasoning 공용).
def _nearest_objects(f: Frame, max_n: int = SPATIAL_MAX_OBJECTS) -> list[dict]:
    best: dict[str, dict] = {}                       # track_token → 가장 가까운 관측
    for v in CAMERA_VIEWS:
        for o in f.camera_objects(v):
            c3d = (o.get("detection_bbox_3d") or {}).get("center_3d_ego") or {}
            x, y = c3d.get("x"), c3d.get("y")
            if x is None or y is None:
                continue
            dist = (x * x + y * y) ** 0.5
            cat = o.get("detection_label") or (o.get("category", "") or "").split(".")[-1] or "object"
            key = o.get("track_token") or f"{v}:{id(o)}"
            if key not in best or dist < best[key]["dist_m"]:
                best[key] = {"category": cat, "dist_m": round(dist, 1),
                             "fwd_m": round(x, 1), "left_m": round(y, 1)}
    return sorted(best.values(), key=lambda d: d["dist_m"])[:max_n]


# frame_to_spatial_sft: 한 프레임을 spatial supervision 샘플로 변환(논문 A의 1Hz 인식 태스크).
def frame_to_spatial_sft(f: Frame) -> tuple[Optional[dict], Optional[str]]:
    if not f.has_camera("front"):
        return None, "no_front_image"
    objs = _nearest_objects(f)
    if not objs:
        return None, "no_spatial_objects"
    return {
        "id": f.sample_id + "_sp",                   # driving 샘플과 id 충돌 방지
        "clip_token": f.clip_token,
        "task": "spatial",                           # 객체 인식 태스크
        "images": _frame_images(f),
        "mission": f.mission_command,
        "output": {"objects": objs},
    }, None


# ─── 3종 reasoning 텍스트 직렬화(train_traj_reas의 --reasoning-types LM_loss 타깃) ──────────────
# 논문의 spatial·decision·counterfactual reasoning을 각각 자연어/구조 텍스트로 직렬화한다. 프레임에
# 해당 reasoning이 없으면 None(대부분 프레임은 reasoning 파일이 없어 셋 다 None → LM_loss 없이 궤적만).
#   빈도: spatial ~1Hz(reasoning 파일 있는 프레임), decision·counterfactual ~0.2Hz(그 부분집합).

# _spatial_reasoning_text: 가까운 객체들을 "category d m ahead, l m left" 목록 텍스트로.
def _spatial_reasoning_text(f: Frame, max_n: int = 8) -> Optional[str]:
    objs = _nearest_objects(f, max_n)
    if not objs:
        return None
    parts = [f"{o['category']} at {o['fwd_m']}m ahead {o['left_m']}m left ({o['dist_m']}m)" for o in objs]
    return "Nearby objects: " + "; ".join(parts) + "."


# _decision_reasoning_text: Driving 블록의 "Reasoning trace"(행동 결정 근거 자연어).
def _decision_reasoning_text(f: Frame) -> Optional[str]:
    return (f.reasoning_trace() or "").strip() or None


# GATE_LABEL: selective-view 게이팅용 3분류(정수). 0=직진(뷰 게이팅 없음), 1=좌, 2=우.
#   Driving decision.Lateral(7종 자유값)을 방향 3종으로 통합(회전/차선변경/미세이동을 좌·우로 묶음) —
#   게이팅이 필요한 건 "카메라를 어느 방향으로 켤까"뿐이라 세부 기동 종류는 불필요.
GATE_LABEL = {"straight": 0, "left": 1, "right": 2}


# _lateral_to_gate: Driving decision.Lateral 원문 문자열 → {straight,left,right}(없으면 None).
#   analysis/maneuver_from_traj.py의 lateral_dir과 동일 매핑(검증된 3분류).
def _lateral_to_gate(lateral: str) -> Optional[str]:
    s = (lateral or "").strip().lower()
    if not s:
        return None
    if "no lateral" in s:
        return "straight"
    if "left" in s:
        return "left"
    if "right" in s:
        return "right"
    return None


# _gate_directions_for_clip: clip의 **전 프레임(10Hz 순서)**을 순회하며 Lateral을 관측된 시점부터
#   다음 관측까지 이월(propagation)한 {frame_index: "straight"|"left"|"right"} 매핑을 만든다.
#   ⚠️ 인과 유지: 이 값은 "과거에 관측된 가장 최근 결정"이라 미래 정보가 섞이지 않는다(oracle 아님).
#   첫 관측 이전 구간(clip 시작~최초 키프레임)은 관측된 과거 결정이 없으므로 안전측 기본값 "straight".
def _gate_directions_for_clip(frames: list) -> dict:
    out: dict = {}
    current = "straight"                      # 첫 결정 관측 전 기본값(안전측 = 뷰 게이팅 없음)
    for f in frames:
        dec = f.driving_decision()             # {"Longitudinal","Lateral"} or None(spatial-only 프레임)
        if dec is not None:
            g = _lateral_to_gate(dec.get("Lateral", ""))
            if g is not None:
                current = g                    # 새 관측 → 이월값 갱신(다음 관측 전까지 유지)
        out[f.frame_index] = current
    return out


# _counterfactual_reasoning_text: Alternative/Top safety-critical actions를 "행동 → 위험등급: 이유" 텍스트로.
def _counterfactual_reasoning_text(f: Frame) -> Optional[str]:
    cf = f.counterfactual()
    if not cf:
        return None
    lines = []
    for key in ("Alternative actions", "Top safety-critical actions"):
        for a in cf.get(key, []) or []:
            if not isinstance(a, dict):
                continue
            lon, lat = (a.get("Longitudinal") or "").strip(), (a.get("Lateral") or "").strip()
            risk, reason = (a.get("Risk level") or "").strip(), (a.get("Reason") or "").strip()
            lines.append(f"{lon} / {lat} → {risk}: {reason}")
    return "Counterfactual actions: " + " | ".join(lines) if lines else None


# _reasoning_parts: 한 프레임의 3종 reasoning 텍스트 dict(있는 것만). train이 --reasoning-types로 선택.
def _reasoning_parts(f: Frame) -> dict:
    parts = {}
    for name, fn in (("spatial", _spatial_reasoning_text),
                     ("decision", _decision_reasoning_text),
                     ("counterfactual", _counterfactual_reasoning_text)):
        txt = fn(f)
        if txt:
            parts[name] = txt
    return parts


# frame_to_trajectory_sft: 한 프레임을 궤적 planning 샘플로 변환(DiT 헤드 + reasoning 공동학습용).
# 논문 nuVLA 정렬: ego_state.trajectory_future(global UTM 50×3 @10Hz)를 **2Hz로 서브샘플**(stride 5)하고
# **heading θ를 유지**해 ego-frame (fwd, left, θ) 고정 길이 T=10 waypoints로 변환(=5초 horizon @2Hz).
# 5초 미래가 확보 안 돼 길이가 TRAJ_N_POINTS 미만이면(클립 끝 근처) 제외해 DiT 고정 시퀀스 길이를 보장.
# reasoning이 있는 프레임이면 3종(spatial/decision/counterfactual) 텍스트를 output.reasoning_parts에 실어
# 논문식 "궤적+reasoning 공동 supervision"을 가능케 한다 — 학습기(train_traj_reas)가 --reasoning-types로
# 어떤 종류를 LM_loss 타깃으로 쓸지 단계별 선택한다(baseline=無 → +spatial → +decision → +counterfactual).
def frame_to_trajectory_sft(f: Frame, n_points: int = TRAJ_N_POINTS,
                           gate_direction: Optional[str] = None) -> tuple[Optional[dict], Optional[str]]:
    if not f.has_camera("front"):
        return None, "no_front_image"
    wp = future_waypoints_ego(f.ego_state(), n_points,   # (T,3) [fwd,left,θ] ego-frame, 2Hz(stride 5)
                              stride=TRAJ_STRIDE_2HZ, with_heading=True)
    if wp is None:
        return None, "no_trajectory"
    if len(wp) < n_points:                               # 고정 길이 미달 → 제외(패딩 대신 단순 제외)
        return None, "short_trajectory"
    out = {"waypoints": [[round(float(x), 2), round(float(y), 2), round(float(th), 3)] for x, y, th in wp]}
    # selective-view용 기동 신호(연속값, 이산화는 학습/평가 시 thr로): ego-frame 미래궤적의 lateral(=left) 성분.
    #   maneuver_lateral     = 최종(5초 뒤) lateral 변위(m). 부호=방향(+좌/−우), 크기=정도.
    #   maneuver_max_lateral = 5초 내 |lateral| 최대(m). 차선변경(갔다 유지)은 둘 다 큼, 회피(갔다 복귀)는 max만 큼.
    #   ⚠️ thr은 여기서 고정하지 않는다 — 원본 연속값만 심어 학습·평가에서 abs(·)<thr로 자유롭게 이산화(재빌드 불필요).
    #   검증: Lateral 텍스트 라벨과 최종변위 3분류 일치 82~86%(analysis/maneuver_from_traj.py, thr 0.75~2.0).
    lat = [float(row[1]) for row in wp]
    out["maneuver_lateral"] = round(lat[-1], 3)
    out["maneuver_max_lateral"] = round(max(lat, key=abs), 3)
    # gate_direction(정수 0/1/2): 과거에 관측된 가장 최근 Driving decision.Lateral을 이월한 값(인과 안전,
    #   미래 GT 미사용). selective-view 게이트 분류기(1단계 판단)의 학습 타깃 + 뷰 게이팅 인덱스로 쓴다.
    #   None이면(이 clip에 아직 Lateral 관측이 없거나 build 구버전) 필드 자체를 생략(하위호환).
    if gate_direction is not None:
        out["gate_direction"] = GATE_LABEL[gate_direction]
    parts = _reasoning_parts(f)                          # {spatial?,decision?,counterfactual?} 있는 것만
    if parts:
        out["reasoning_parts"] = parts
        if "decision" in parts:                          # 하위호환: 기존 필드는 decision reasoning 유지
            out["reasoning"] = parts["decision"]
    return {
        "id": f.sample_id + "_tj",                       # driving/spatial 샘플과 id 충돌 방지
        "clip_token": f.clip_token,
        "task": "trajectory",                            # 궤적 planning 태스크(DiT + reasoning 공동)
        "images": _frame_images(f),
        "mission": f.mission_command,
        # ego 상태(DiT의 state token, 논문 nuVLA): **현재 운동상태 [vx,vy,ax,ay]** + **ego 이동 history**
        #   (과거 ego pose 2Hz T_H개를 ego-frame [fwd,left,θ]로 flatten해 뒤에 이어붙임). 단일 프레임 이미지로는
        #   속도를 알 수 없어 현재 속도(=최강 예측변수)가 핵심이고(진단: ego속도만 선형회귀해도 ADE 0.97m),
        #   history는 최근 운동 추세를 준다. 앞 4차원이 항상 현재 dynamics → 하위호환·해석 일관.
        "ego_state": _ego_state_vec(f),
        "output": out,
    }, None


# _ego_motion: ego-frame 현재 [vx, vy, ax, ay] (forward/left 속도·가속도). 키 없으면 0.0.
def _ego_motion(f: Frame) -> list:
    es = f.ego_state()
    v = getattr(es, "velocity", None) or {}
    a = getattr(es, "acceleration", None) or {}
    return [round(float(v.get("vx", 0.0)), 4), round(float(v.get("vy", 0.0)), 4),
            round(float(a.get("ax", 0.0)), 4), round(float(a.get("ay", 0.0)), 4)]


# _ego_state_vec: DiT state token용 ego 상태 = [vx,vy,ax,ay] + flatten(과거 pose T_H×3). 논문 nuVLA의
#   "ego dynamics + ego motion history를 flatten한 state token". history는 trajectory_history(과거 pose)를
#   2Hz(TRAJ_STRIDE_2HZ)로 서브샘플한 recent→old T_H개 [fwd,left,θ]. history 없으면 dynamics만(하위호환).
def _ego_state_vec(f: Frame) -> list:
    vec = _ego_motion(f)                                   # [vx,vy,ax,ay] (항상 앞 4차원)
    if TRAJ_HISTORY_N > 0:                                 # 고정 차원 보장: 항상 T_H×3 값을 이어붙임
        hist = past_waypoints_ego(f.ego_state(), TRAJ_HISTORY_N,
                                  stride=TRAJ_STRIDE_2HZ, with_heading=True)   # (T_H,3) or None
        if hist is not None:
            vec += [round(float(x), 3) for row in hist for x in row]           # flatten recent→old
        else:                                              # 클립 초반(과거 3초 미만) → history 0 패딩
            vec += [0.0] * (TRAJ_HISTORY_N * 3)            #   (reasoning 프레임 30~150은 전부 실제 history 보유)
    return vec


def build(clips_root: Path, out_dir: Path, traj_stride: int = 10) -> dict:
    val_tokens = val_clip_tokens(VAL_MANIFEST)
    splits: dict[str, list[dict]] = {"train": [], "val": []}
    skips = Counter()
    seen: set[str] = set()

    # 멀티태스크 샘플 생성(논문 A) — 태스크별로 프레임 밀도가 다르다:
    #   - driving 샘플: Driving decision이 있는 1Hz 프레임(클립당 ~3) → 행동 결정 (주석에 묶임)
    #   - spatial 샘플: 객체 주석이 있는 1Hz 프레임(클립당 ~13) → 인식 (주석에 묶임)
    #   - trajectory 샘플: planning을 supervise할 **프레임**을 traj_stride로 뽑는다(=클립당 planning 샘플 수).
    #     논문은 주석(reasoning) 프레임에서 planning+reasoning을 공동 supervise → traj_stride=10(1Hz)이 기본
    #     (reasoning 주석 밀도와 일치, 클립당 ~16). ⚠️ 이 traj_stride는 "어느 프레임을 샘플로 쓰나"이고,
    #     각 샘플의 **waypoint 자체**는 frame_to_trajectory_sft가 2Hz(TRAJ_STRIDE_2HZ)로 서브샘플한 T=10이다(별개).
    def _emit(rec, reason):
        if rec is None:
            skips[reason] += 1
            return
        if rec["id"] in seen:                   # 같은 클립이 두 추출 폴더에 겹칠 때 중복 제거
            return
        seen.add(rec["id"])
        split = "val" if rec["clip_token"] in val_tokens else "train"
        splits[split].append(rec)

    # _emit_with_hist: 레코드 생성 → (TEMPORAL이면) 과거 8뷰를 history_images로 주입 → _emit.
    #   과거뷰 주입은 빌드 시점에만 가능(클립 프레임 구조 필요). 학습·평가는 이 필드만 보고 시간 맥락을 켠다.
    def _emit_with_hist(built, hist):
        rec, reason = built
        if rec is not None and hist:
            rec["history_images"] = hist        # [{view,path}] 과거 1 timestep(8뷰)
        _emit(rec, reason)

    for clip in iter_clips(clips_root):
        # driving + spatial: 주석에 묶인 1Hz 프레임(더 못 늘림)
        for f in select_keyframes(clip, policy="spatial"):
            hist = _history_images(clip, f)     # TEMPORAL=False면 항상 None(현재 프레임만)
            _emit_with_hist(frame_to_sft(f), hist)             # driving (결정 없는 프레임은 no_decision으로 스킵)
            _emit_with_hist(frame_to_spatial_sft(f), hist)     # spatial (1Hz 프레임 전체)
        # gate_direction 이월(propagation): clip **전 프레임(10Hz, 순서대로)**을 한 번 훑어 frame_index별
        # "가장 최근 관측된 Lateral"을 계산(인과 안전 — 미래 GT 미사용). trajectory 샘플이 이 값을 참조.
        all_frames = select_keyframes(clip, policy="all")
        gate_map = _gate_directions_for_clip(all_frames)
        # trajectory: 10Hz 전 프레임을 traj_stride로 → planning 밀도 ↑ (short_trajectory/이미지결측은 스킵)
        for f in all_frames[:: max(1, traj_stride)]:
            hist = _history_images(clip, f)
            gd = gate_map.get(f.frame_index)
            _emit_with_hist(frame_to_trajectory_sft(f, gate_direction=gd), hist)  # trajectory (10Hz 밀도)

    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"skips": dict(skips), "splits": {}}
    for name, recs in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        drv = [r for r in recs if r.get("task", "driving") == "driving"]   # 행동 라벨 분포는 driving만
        lon = Counter(T.map_longitudinal(r["output"]["longitudinal"]) for r in drv)
        lat = Counter(T.map_lateral(r["output"]["lateral"]) for r in drv)
        tasks = Counter(r.get("task", "driving") for r in recs)
        n_views = Counter(len(r["images"]) for r in recs)   # 샘플당 보유 뷰 수 분포(8뷰 정상)
        n_hist = sum(1 for r in recs if r.get("history_images"))  # 과거 1 timestep(8뷰) 주입된 샘플 수
        traj = [r for r in recs if r.get("task") == "trajectory"]           # 궤적 통계는 trajectory만
        report["splits"][name] = {
            "path": str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path),
            "n_samples": len(recs),
            "n_by_task": dict(tasks.most_common()),         # {driving, spatial, trajectory}
            "n_clips": len({r["clip_token"] for r in recs}),
            "views_per_sample": dict(n_views.most_common()),
            # 시간 맥락(temporal): TEMPORAL=True일 때 과거 8뷰가 주입된 샘플 수(이미지 ~1Hz라 offset이
            # 10배수가 아니거나 클립 시작 근처면 0일 수 있음 → 0이면 vlm.py의 TEMPORAL_HISTORY_OFFSET 점검).
            "n_with_history": n_hist,
            "long_dist": dict(lon.most_common()),
            "lat_dist": dict(lat.most_common()),
        }
        if traj:                                            # 궤적 GT 평균 전방 도달거리(검증용 sanity)
            finals = [r["output"]["waypoints"][-1][0] for r in traj]        # 마지막 waypoint의 fwd(m)
            report["splits"][name]["traj_final_fwd_m_mean"] = round(sum(finals) / len(finals), 1)
            gd = Counter(r["output"]["gate_direction"] for r in traj if "gate_direction" in r["output"])
            if gd:                                          # {0:straight,1:left,2:right} 이월 분포(selective-view)
                report["splits"][name]["gate_direction_dist"] = {
                    {0: "straight", 1: "left", 2: "right"}[k]: v for k, v in gd.most_common()}
    report["val_clip_tokens_known"] = len(val_tokens)
    # 시간 맥락 설정 스냅샷(빌드 시점 vlm.py 값) — 데이터가 어떤 temporal 설정으로 만들어졌는지 추적.
    report["temporal"] = {"enabled": TEMPORAL, "history_offset": TEMPORAL_HISTORY_OFFSET}
    report["traj_stride"] = traj_stride     # trajectory 프레임 밀도(1=10Hz, 5=2Hz, 10=1Hz)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", default=str(DEFAULT_CLIPS_ROOT))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--traj-stride", type=int, default=10,
                    help="trajectory 샘플을 뽑을 10Hz 프레임 간격(=클립당 planning 샘플 수). 10=1Hz(기본, 논문의 "
                         "주석 프레임 밀도와 일치, 클립당 ~16), 5=2Hz, 1=10Hz 전부(과밀). driving/spatial은 "
                         "주석에 묶여 영향 없음. ⚠️ 각 샘플의 waypoint 자체는 항상 2Hz T=10(별개).")
    args = ap.parse_args()

    report = build(Path(args.clips_root), Path(args.out), args.traj_stride)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["splits"]["train"]["n_samples"] == 0:
        print("\nℹ️  train split is empty — only val(=eval) clips are present. "
              "Download disjoint clips to populate train.")
    # 시간 맥락 켰는데 과거뷰가 한 샘플도 안 붙으면 명확히 알린다.
    if TEMPORAL:
        nh = report["splits"]["train"]["n_with_history"]
        nt = report["splits"]["train"]["n_samples"]
        print(f"\n🕑 TEMPORAL=ON (history offset {TEMPORAL_HISTORY_OFFSET} frames = "
              f"{TEMPORAL_HISTORY_OFFSET/10:.1f}s @10Hz) — train: {nh}/{nt} samples got a past 8-view frame.")
        if nh == 0:
            print("⚠️  과거뷰가 0건 — 10Hz 전체 이미지(--policy all)로 받았는지, offset이 클립 길이보다 "
                  "크지 않은지 확인. (1Hz-only 데이터면 offset이 10의 배수여야 과거 프레임에 이미지가 있음.)")


if __name__ == "__main__":
    main()
