# nuReasoning 데이터셋 정리

이 프로젝트가 쓰는 **nuReasoning 데이터셋**의 출처·구조·주석·라벨 빈도와, 정식 논문(nuVLA)의
학습 방식 및 우리 미니프로젝트의 단순화 대비를 정리한 문서다.
학습 파이프라인(text_sft + trajectory 두 경로)은 [Full_pipeline.md](Full_pipeline.md)를 참고.

---

## 1. 개요 / 출처

- **논문:** *nuReasoning: A Reasoning-Centric Dataset and Benchmark for Long-Tail Autonomous Driving*,
  arXiv:2605.31572 (UCLA + Motional). 프로젝트 홈: https://nureasoning.github.io/
- **HF repo:** `qixuewei/nuReasoning` (라이선스 표기 `apache-2.0`, 실제 COMMERCIAL/NONCOMMERCIAL 듀얼 — 연구·분석 용도는 NONCOMMERCIAL로 충분).
- **계보:** nuScenes / nuPlan 라인을 perception·planning에서 **reasoning 중심**으로 확장.
- **규모(논문 전체 데이터셋 기준):** 20K clips, 각 **≈20초 @ 10Hz**, 다도시(Las Vegas·Pittsburgh·LA·Boston). split = **train 17K / val 2K / private test 1K** (§3.1). ⚠ local에 받은 건 이 중 일부.
- **클립 1개 구성:** 동기화된 멀티모달 — 8뷰 카메라 + LiDAR + **HD map(3D 벡터)** + ego state + 3D object annotation + traffic light + route + navigation command + **reasoning annotation**. (HD map 상세는 §2.1)
- **long-tail 중심:** 사내 fleet 로그를 VLM 평가기(Gemini)로 난이도 1~10 채점 → 점수>5만 + 인간 검증 + **decision-critical 키프레임** 선택. 키프레임 ±10초를 잘라 20초 클립 구성(§3.2).

### 1.1 센서·수집 플랫폼 (논문 명시 범위)

논문이 **공식적으로 기재한** 센서/수집 정보만 적는다. 기재되지 않은 항목은 아래에서 "논문 미기재"로 명확히 구분한다(추정치를 사실처럼 적지 않기 위함).

**수집 플랫폼·지역·샘플링:**

| 항목 | 논문 명시 내용 | 출처 |
|---|---|---|
| 차량/출처 | **Motional 사내 자율주행(AV) fleet** 주행 로그 | §3.2 *"Motional's internal AV fleet driving logs"* |
| 수집 지역 | 미국 다도시 — **Las Vegas, Pittsburgh, Los Angeles, Boston** | §3.4 |
| 샘플링 | 모든 모달리티(센서·객체 트랙·신호등 상태·ego state) **10Hz 동기화** | §3.3 |
| 차량 모델(제조사/차종) | **논문 미기재** | — |


**센서 구성:**

| 센서 | 논문 명시 내용 | 미기재 항목 |
|---|---|---|
| 카메라 | **8뷰 surround**(front, front-left, front-right, left, right, back, back-left, back-right) — §C.1 | 제조사/모델 논문 미기재. 원본 해상도는 논문 미기재이나, 실측(front 이미지 PIL 디코드)으로 **2816×1856 px** 확인([availability_report.md §3](data/availability_report.md)). 모델 입력 시 448×448 리사이즈. |
| LiDAR | "LiDAR point clouds" 모달리티로 포함 — §3.1, Fig.2 | **브랜드·모델·채널 수 전부 미기재** |
| 기타 | ego state(pose/velocity/accel), 3D object annotation, HD vector map, traffic-light state, route path, navigation command, **센서 캘리브레이션**(camera intrinsic + sensor2lidar) — §3.1 | — |

> ⚠️ **논문에 공식적으로 기재되지 않은 항목**
>
> | 항목 | 상태 |
> |---|---|
> | LiDAR 브랜드 / 모델명 / 채널 수 | **논문 미기재** — §3.1·Fig.2에 "LiDAR point clouds"라고만 언급, 세부 사양 없음 |
> | 수집 차량 모델 (제조사/차종) | **논문 미기재** — "Motional's internal AV fleet"까지만 기재 |
> | 카메라 네이티브 해상도 | **논문 미기재** — 단, 실측(front 이미지 PIL 디코드)으로 **2816×1856 px** 확인; 모델 입력 시 448×448 리사이즈 |
> | 카메라 제조사·모델 | **논문 미기재** |
>
> 위 항목은 논문 전체(본문·부록·supplementary)를 통틀어 기재되지 않는다. 따라서 외부 추정이나 Motional 공개 차량 정보 등을 끌어다 채우지 않는다.
> (참고로 우리가 받은 클립에선 LiDAR `point_cloud_path`가 전부 비어 있어 실물 포인트클라우드도 없음 — [DECISIONS.md](DECISIONS.md) KNOWN_ISSUES.)

## 2. 클립 디렉토리 구조

```
/home/etri/DATASET/nureasoning/clips/<clip>/
├─ metadata.json           # 프레임 목록(202개) + 카메라/센서 경로 + mission_goal
├─ cameras/CAM_*/*.jpg     # 8뷰 카메라 이미지 (CAM_M_F = front …)
├─ *.pkl                   # ego_state / annotations (repo-root data_schema.py 객체)
└─ reasoning/<ts_us>.json  # reasoning 주석 ← 희소!
```

- metadata의 프레임 레코드 키: `token, frame_index, timestamp_us, relative_time_s, ego_state, sensors, annotations, reasoning, mission_goal`.
- **모든 프레임(~202, 10Hz)이 `ego_state`·`sensors`(8뷰)·`annotations`를 보유**(실측). sparse한 건 `reasoning`뿐(클립당 ~13개 파일).
- ⚠ 카메라 경로는 metadata의 `sensors.cameras` 값을 그대로 써야 함 — 카메라 파일명 타임스탬프와 프레임 `timestamp_us`가 다름(재구성 금지).
- `map.pkl`은 **클립당 1개(static map)** — 프레임마다가 아님(주행 중 정적 지도는 불변).

### 2.1 HD map = 3D 벡터(폴리라인/폴리곤) 맵

`map.pkl`은 `data_schema.nuReasoningStaticMap` 객체로, **BEV 래스터 이미지가 아니라
nuScenes/nuPlan 계열의 3D 벡터 맵**이다. 모든 요소의 geometry가 `[[x, y, z], ...]`
좌표열(폴리라인/폴리곤)로 정의됨([data_schema.py](data_schema.py) §Static map).

구성 요소: `lanes`(polygon + centerline + speed_limit), `baseline_paths`, `boundaries`,
`crosswalks`, `intersections`, `stop_polygons`, `road_blocks`, `traffic_lights`,
`lane_connectors`.

실측(클립 `2024.02.27...`의 `map.pkl`, point 차원 직접 확인):

| 요소 | 개수 | point 차원 | 샘플 좌표 |
|---|---|---|---|
| lanes(polygon) | 64 | **3** | `[664514.41, 3996665.22, 0.0]` |
| lanes(centerline) | 64 | **3** | `[664514.36, 3996667.47, 0.0]` |
| boundaries | 299 | **3** | `[664410.14, 3996656.30, 0.0]` |
| baseline_paths | 120 | **3** | `[664492.34, 3996659.75, 0.0]` |
| traffic_lights | 33 | **3** | `[664442.89, 3996655.30, **2.6**]` |

- **3D지만 노면은 사실상 평면:** 차로/경계의 z는 대부분 `0.0`, **신호등만 z=2.6m**처럼 실제
  높이를 가짐 → "z 정보가 있는 3D 벡터 맵"으로 이해하면 정확.
- **좌표계 = 절대 map 좌표(UTM류 global frame):** `664514, 3996665` 같은 큰 값. ego 상대좌표가
  아니라서 모델에 쓰려면 ego pose로 변환 필요. (참고: `mission_goal.route_path`는 `[[x, y]]` **2D**.)
- **로더 노출 위치:** `Clip.map`이 `map_annotation`(기본 `map.pkl`)을 lazy unpickle
  ([loader.py:245-251](src/nureasoning/loader.py#L245-L251)).
- ⚠ **우리 미니파이프라인에선 미사용** — SFT는 front 이미지+텍스트만 입력. 로더에 노출만 돼 있음.
  (논문 nuVLA도 map을 모델 명시 입력으로 쓰진 않음 — 라벨 생성·평가 재료 쪽.)

## 3. reasoning 주석 3종

| 종류 | 내용 | 주요 필드 |
|---|---|---|
| **Spatial** (공간) | 장면/객체 인식·관계 | `per_camera_results`(카메라별 객체: `track_token`, `category`, `detection_bbox_2d`[이미지 픽셀], `detection_bbox_3d`[ego 프레임 center/velocity]) |
| **Driving** (의사결정) | 그 시점 주행 결정 + 근거 | `Driving decision` {`Longitudinal`(9종), `Lateral`(7종)}, `Reasoning trace`(1문장 인과), `Scene description`, `Critical components` |
| **Counterfactual** (반사실) | "만약 ~했다면"의 대안 행동 위험도 | `Alternative actions`(Safe), `Top safety-critical actions`(각 항목 `Risk level`∈{Safe, Suboptimal, Unsafe} + reason) |

- **액션 taxonomy:** Longitudinal 9종(Remain stopped / Quickly·Gently come to a stop / Slow down quickly·gently / Quickly·Gently accelerate / Maintain speed / Reverse), Lateral 7종(Slightly move left·right / Left·Right lane change / Turn left·right / No lateral action). (논문 Fig S5)
- **생성 방식:** Spatial = Gemini 3 Flash 2D 검출 + 3D annotation projection/IoU 매칭. Driving/CF = Gemini 3.1 Pro + 2단 인간 검증(decision 84.69% / CF 76.11% auto-human agreement 후 교정).

JSON 구조(한 파일 = 한 프레임):

```jsonc
{
  "frame_index": 30,
  "Spatial":        { ... },   // 거의 항상 채워짐 — 카메라별 객체 인식/2D·3D bbox/관계
  "Driving":        { },       // 대부분 비어 있음 — 주행 의사결정/행동
  "Counterfactual": { }        // 대부분 비어 있음 — 대안 행동의 위험도
}
```

⚠ **데이터 함정:** `Driving`/`Counterfactual`은 채워지면 dict, 비면 빈 str/{} — 파싱 시 타입 체크 필요.

## 4. reasoning 라벨 빈도 (희소성)

reasoning은 **2단 주기**로 sparse하게 달린다(논문 §3.3):

- **Spatial:** 클립 3초 지점부터 **1Hz** → 클립당 ~13개.
- **Driving / Counterfactual:** 키프레임 5초 전부터 **0.2Hz** → 클립당 ~3개.
- **0.2Hz인 이유:** 어노테이션 누락이 아니라 *"matching the lower operating frequency of the
  decision-making module"* — 주행 의사결정 모듈 자체가 그 주기로 동작하므로 더 자주 달면 중복.

실측(클립 `2024.02.27.12.52.42_...`): 202프레임 중 reasoning json 13개, 그중 `Driving`/`CF`까지 채워진 건 3개:

| frame_index | 채워진 key |
|---|---|
| 30, 40 | Spatial |
| 50 | Spatial, Driving, Counterfactual ✅ |
| 60, 70, 80, 90 | Spatial |
| 100 | Spatial, Driving, Counterfactual ✅ |
| 110, 120, 130, 140 | Spatial |
| 150 | Spatial, Driving, Counterfactual ✅ |

→ 13개 중 3개(frame 50/100/150)만 세 가지 다 채워짐. 나머지는 `Spatial`만.
**전체 규모(논문 전체 데이터셋 기준, §3.4):** spatial **247K** 프레임(1Hz) + decision/CF **57K** 프레임(0.2Hz) (≈4.3:1, 위 13:3과 일치). ⚠ local 다운로드분은 그 일부.

## 5. 프레임 종류별 역할 (reasoning 없는 프레임의 용도)

reasoning 주석이 없어도 그 프레임은 **버려지지 않는다** — ego 궤적·객체 박스·이미지가 10Hz로 있어
입력 컨텍스트·supervision·평가 GT로 쓰인다.

| 프레임 종류 | reasoning 라벨 | 역할 |
|---|---|---|
| 키프레임 (0.2Hz, 클립당 ~3) | `Driving`/`Counterfactual` 있음 | reasoning supervision + planning 평가 지점 |
| Spatial 프레임 (1Hz, 클립당 ~13) | `Spatial`만 | spatial supervision (+ history/future 재료) |
| 비주석 프레임 (나머지 ~189) | 없음 | ① 입력 history ② 미래 궤적 GT ③ 충돌/DA GT ④ 라벨 생성 재료 |

- **① 입력 history:** 학습 샘플 = "현재 + 직전 1초" 멀티뷰(§5.1, C.1) → history 프레임은 비주석이지만 시각 컨텍스트로 입력.
- **② 미래 궤적 GT:** planning 타깃(5초·2Hz)은 **미래 비주석 프레임의 ego 궤적**에서 나옴. §3.2: *"10s preceding context + 10s subsequent data ... future scene evolution"*.
- **③ 충돌/DA GT:** planning 평가(부록 D)는 예측 궤적을 **미래 프레임의 객체 박스**와 겹쳐 채점.
- **④ 라벨 재료:** Spatial의 future-motion/TTC/conflict는 *subsequent annotations*로 계산(부록 B.1), Decision/CF는 future ego trajectory에 정렬해 생성(부록 B.2).

> §5.1의 *"주석 없는 샘플은 학습에서 제외"*는 **reasoning 타깃으로 직접 학습하지 않는다**는 뜻이지
> 데이터를 안 쓴다는 뜻이 아니다.

## 6. perception 라벨(3D/2D bbox)이 모델에서 쓰이는 경로

3D bounding box는 **모델 입력이 아니라** 학습 타깃·평가 GT·라벨 재료로 쓰인다.

| 경로 | 3D bbox 사용? | 비고 |
|---|---|---|
| 모델 **입력**(추론 시) | ❌ | 이미지 + 텍스트 + ego dynamics만; 객체는 모델이 스스로 인식 |
| 학습 **타깃**(Spatial supervision) | ✅ | "ego-frame 3D 위치/2D bbox/distance/TTC" 등을 **출력**하도록 학습(부록 C.1) |
| **평가** GT | ✅ | reasoning 좌표 정답(Table S2) + planning 충돌·DA 채점(부록 D) |
| **라벨 생성** 재료 | ✅ | 2D 검출 ↔ 3D annotation projection/IoU 매칭으로 Spatial 라벨 구성(부록 B.1) |

→ nuReasoning의 핵심 주장과 연결: reasoning supervision(3D 위치·관계 포함)을 넣으면 **추론 시 그
reasoning 출력을 꺼도** planning이 개선된다(§5.3).

### "그럼 3D bbox 검출 없이 바로 planning 하면 안 되나?"

된다 — **추론 때는** 검출 없이 바로 결정/궤적을 내도 되고(위 표 입력 ❌, §5.3에서 reasoning
출력을 꺼도 planning↑), 그게 오히려 권장이다. 그런데도 **학습 때** 3D bbox를 타깃으로 넣는 이유:

- **shortcut 방지(grounding):** trajectory만 주면 "이런 장면이면 사람이 이렇게 운전" 같은 표면
  통계를 외워버리기 쉽다(객체 위치를 실제로 이해 안 해도 됨). 3D 위치를 **출력하게** 강제하면
  visual encoder가 공간 구조를 실제로 뽑는다.
- **VLM 기하 감각 보강:** Qwen-VL backbone은 웹 이미지 사전학습이라 미터 단위 거리·속도 감각이
  약함 → spatial supervision이 그 gap을 메운다.
- **dense supervision:** 궤적 GT는 float 몇 개로 신호가 약하지만, spatial은 이미지 1장당 객체별
  위치/거리/TTC로 신호가 풍부 → 특히 long-tail에서 sample efficiency↑.
- **chain-of-thought:** spatial=전제("보행자 접근 중"), driving decision=결론("Gently come to a
  stop"). 전제를 명시 학습시키면 결론이 더 신뢰성 있고 해석 가능.

⚠ **착각 주의:** 여기서 3D bbox 검출은 모듈형 AD의 *detector → tracker → planner* 별도 단계가
**아니다.** nuVLA는 하나의 VLA가 reasoning trace의 일부로 객체 위치를 언어/토큰으로 서술하도록
학습될 뿐 — bbox는 "파이프라인 단계"가 아니라 "보조 supervision 타깃". "이미지→planning"만
시키는 건 곧 pure end-to-end imitation이고, reasoning supervision을 더하면 그보다 나아진다는 게
논문 main claim(§5.3).

## 7. 정식 논문(nuVLA) 학습 방식 & 우리 미니프로젝트 대비

논문의 핵심 학습 규칙:

> **"Only samples with reasoning annotations are used for training."** (§5.1)

이 규칙은 우리 코드(`select_keyframes` 선별 + `frame_to_sft` skip)와 철학이 같다. 차이는 **"무엇을
얼마나 쓰느냐"의 스코프**다:

| 항목 | 논문 nuVLA | 이 미니 파이프라인 |
|---|---|---|
| 카메라 | 8뷰 전부 | **8뷰 surround 전부**(2026-06-22 front 1장 → 8뷰 확장) |
| 시간 맥락 | 현재 + 직전 1초(2 timestep) = 최대 16장 | 단일 프레임(키프레임 1장당 8뷰) |
| 사용 프레임 | `Spatial`(1Hz, 247K) **+** `Driving`/`CF`(0.2Hz, 57K) 모두 | **`Spatial`(1Hz, ~13/클립) + `Driving`(0.2Hz, ~3/클립)**(2026-06-22 spatial 추가) |
| 태스크 | 궤적 planning(주) + reasoning supervision(보조) | **멀티태스크 SFT** — driving(행동+근거) + spatial(객체 인식·위치) 텍스트 생성 |
| 구조 | Qwen3-VL-2B + flow-matching DiT 궤적 헤드 | Qwen-VL QLoRA(텍스트만) |

- **"성분이 일부만 있는 프레임" 처리:** 논문은 마스킹이 아니라 *있는 reasoning 성분만 타깃에 결합*하고
  언어 손실을 assistant 토큰에만 건다 → `Spatial`만 있는 프레임은 spatial 타깃만 학습. 궤적 손실은
  GT trajectory가 10Hz라 모든 학습 프레임에서 계산.
- **이 미니 파이프라인의 단순화(2026-06-22 갱신):** 이제 **8뷰 surround + 1Hz spatial supervision**까지
  쓴다 — driving(0.2Hz, 행동+근거) **+** spatial(1Hz, 객체 인식·ego 위치) **멀티태스크 SFT**.
  뷰당 비전토큰 예산은 160으로(8뷰+spatial이 단일 GPU 24GB OOM → 하향), [DECISIONS.md](DECISIONS.md) 참고.
  **남은 단순화:** 궤적 planning 헤드·시간 맥락(직전 1초)·비주석 프레임의 미래 궤적/충돌 GT(위 §5 ①~④)·
  LiDAR는 여전히 미사용(LiDAR는 데이터 자체가 없음 — §1.1). spatial 타깃도 가까운 15객체로 캡한 축소판.
