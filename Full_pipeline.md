# nuReasoning 학습 파이프라인 — 개요 (분기 허브)

nuReasoning 원본 클립이 학습에 들어가는 흐름을 정리한 진입점이다. 학습 **objective가 둘**이라
공통 단계만 여기서 요약하고, 태스크별 상세는 두 문서로 분리했다.

> 데이터셋 자체(구조·reasoning 주석·라벨 빈도·정식 논문 대비)는 [DATASET_INFO.md](DATASET_INFO.md)로
> 분리했다. 이 문서는 **학습 파이프라인**에 집중한다.

## 🔀 두 학습 경로 (objective 분기)

논문 nuVLA는 한 모델이 **(가) 텍스트 reasoning/결정**과 **(나) 연속 궤적 planning**을 모두 낸다.
우리는 이를 두 경로로 나눠 각각 재현한다 — 같은 `data/sft/train.jsonl`을 `task` 필드로 공유한다.

| | **text_sft** | **trajectory** |
|---|---|---|
| 상세 문서 | [Full_pipeline_text_sft.md](Full_pipeline_text_sft.md) | [Full_pipeline_trajectory.md](Full_pipeline_trajectory.md) |
| 태스크 | driving(행동 결정) + spatial(객체 인식) | 미래 궤적 (waypoints) |
| 출력 | 텍스트 토큰 (이산 라벨/객체) | 연속 좌표 (N×2) |
| 헤드 | LM head | flow-matching **DiT** |
| 손실 | cross-entropy | rectified-flow MSE |
| 학습 | HF `Trainer` (QLoRA) | 커스텀 루프 (VLM frozen + DiT 학습) |
| GT 출처 | `reasoning/<ts>.json` | `ego_state.trajectory_future` |
| 학습 코드 | `train_text_sft.py --tasks driving,spatial` | `train_traj_reas.py` |
| 평가 | accuracy / macro-F1 (`run_zeroshot.py`) | ADE / FDE (`eval_trajectory.py`) |

## 공통 단계 ①~④ (두 경로 공유)

```
① 원본 클립          /home/etri/DATASET/nureasoning/clips/<clip>/  (8뷰 이미지 + ego_state/annotation/reasoning)
        │  parse_clip()                       [nureasoning/loader.py]   raw 안 읽고 lazy
② Clip → Frame       (이미지·pkl·reasoning은 첫 접근 시 lazy 로드)
        │  select_keyframes(policy="spatial") 주석/주행 있는 1Hz 프레임
③ 키프레임 선별       ← reasoning이 희소해서 라벨/주석 있는 프레임만 사용
        │  build_sft.py  (task별 빌더: driving / spatial / trajectory)
④ SFT 레코드          data/sft/{train,val}.jsonl  (한 파일에 task로 공존, clip 단위 누수 방지)
        │
        ├─▶ ⑤ text_sft   to_qwen_chat → QwenVLCollator → Trainer(QLoRA)   [text_sft 문서]
        └─▶ ⑤ trajectory geometry 변환 → VLM condition → DiT flow-matching [trajectory 문서]
```

| 단계 | 코드 위치 | 핵심 포인트 |
|---|---|---|
| ① 원본 | `/home/etri/DATASET/nureasoning/clips/` | ego/카메라는 전 프레임(10Hz), `reasoning`은 sparse(1Hz/0.2Hz). `trajectory_future`는 프레임 80% 보유 |
| ② 파싱 | `parse_clip` (`loader.py`) | 경로만 들고 **lazy 로드** |
| ③ 선별 | `select_keyframes` (`loader.py`) | `policy="spatial"` → 주석 있는 1Hz 프레임 |
| ④ 생성 | `build_sft.py` | 프레임 → task별 레코드(driving/spatial/trajectory) + clip 단위 train/val split |

> ⑤ 학습부터 두 경로가 갈라진다. **각 상세 문서를 보라:**
> - 텍스트 결정·인식 + reasoning 3층(B 학습/C 안전평가) → [Full_pipeline_text_sft.md](Full_pipeline_text_sft.md)
> - 궤적 좌표 변환 + flow-matching DiT + ADE/FDE → [Full_pipeline_trajectory.md](Full_pipeline_trajectory.md)

## 실제로 돌리는 순서 (요약)

```bash
python scripts/download_clips.py                                    # 클립 다운로드
python -m sft_data.build_sft --clips-root data/raw   # ④ SFT 생성(3 task 공존)

# ⑤ 둘 중 택1 (또는 둘 다)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m training.train_text_sft --tasks driving,spatial
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m training.train_traj_reas
```

## 한 줄 요약

원본 클립 → 파싱 → 키프레임 선별 → SFT 레코드(task별) 까지는 **공통**이고, ⑤ 학습에서
**text_sft(텍스트 결정/인식, QLoRA)** 와 **trajectory(연속 궤적, flow-matching DiT)** 로 갈라진다.
상세는 위 두 문서를 참고하라.
