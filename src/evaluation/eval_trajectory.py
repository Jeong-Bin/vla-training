"""(objective=trajectory) — DiT 궤적 planning 평가 (ADE/FDE).

`train_traj_reas.py`가 학습한 VLM(traj_config의 vlm_mode로 복원: full/lora/frozen) + DiT 헤드로 val의
trajectory 샘플에 대해 미래 궤적을 샘플링하고, ego-frame GT waypoints와 비교해 표준 planning 지표를 낸다:
  - **ADE** (Average Displacement Error): 전 waypoint L2 거리 평균(미터).
  - **FDE** (Final Displacement Error): 마지막 waypoint L2 거리(미터).
참고 baseline으로 **constant-velocity**(첫 스텝 변위를 등속 연장) 궤적의 ADE/FDE도 함께 내 비교 기준 제공.

text_sft 경로의 `run_zeroshot.py`(분류 정확도/F1)에 대응하는 trajectory 경로 평가 하니스다.

Run:  CUDA_VISIBLE_DEVICES=0,1,2,3 python -m evaluation.eval_trajectory \
        --adapter results/traj_reas/<timestamp>/model [--manifest data/sft/val.jsonl] [--limit N] [--steps 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nureasoning import SFT_VAL, load_processor, upsample_waypoints             # noqa: E402
from training.dit_head import TrajectoryDiT, TrajectoryNormalizer  # noqa: E402
from training.train_traj_reas import (encode_condition, ego_vec, history_for, load_vlm,  # noqa: E402
                                       has_reasoning_annotation, gate_views, rec_maneuver)  # 학습=평가 동일 인코더/ego/시간맥락/필터/게이팅

DEFAULT_MANIFEST = SFT_VAL     # vlm.SFT_DIR 중앙 설정 따름(현재 data/sft_v2/val.jsonl). --manifest로 override.
TRAJ_PROMPT_FILE = REPO / "prompts" / "trajectory_plan_v1.txt"
RESULTS = REPO / "results"


# eval_out_dir: 평가 산출물을 둘 디렉터리 결정(학습·평가 일관). 학습이 traj_config.json에 적어둔
# `run_dir`(예: results/traj_reas/<timestamp>)이 있으면 **그 학습 실행 폴더**에 저장 → 평가 결과가
# "어느 학습이 낸 모델인지"와 항상 같은 폴더에 묶인다. 없으면(구버전 모델) results/로 폴백.
def eval_out_dir(cfg: dict) -> Path:
    rd = cfg.get("run_dir")
    out = (REPO / rd) if rd else RESULTS
    out.mkdir(parents=True, exist_ok=True)
    return out


# displacement_errors: 예측·GT waypoints (N,2)에서 ADE(전 점 L2 평균), FDE(마지막 점 L2) 반환.
def displacement_errors(pred, gt):
    import numpy as np

    pred, gt = np.asarray(pred), np.asarray(gt)
    d = np.linalg.norm(pred - gt, axis=1)                  # 각 waypoint의 L2 거리(m)
    return float(d.mean()), float(d[-1])


# constant_velocity: GT 첫 변위(wp[1]-wp[0])를 등속 연장한 참조 궤적(naive baseline). (N,2).
def constant_velocity(gt):
    import numpy as np

    gt = np.asarray(gt)
    step = gt[1] - gt[0] if len(gt) > 1 else gt[0]
    return gt[0] + step * np.arange(len(gt))[:, None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="train_traj_reas.py 출력 디렉터리(dit_head.pt + traj_config.json)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N samples (0=all)")
    ap.add_argument("--steps", type=int, default=5, help="ODE 적분 스텝 수(논문 K=5). --steps로 override 가능")
    ap.add_argument("--tag", default="trajectory", help="output filename tag")
    args = ap.parse_args()

    import numpy as np
    import torch

    adapter = Path(args.adapter)
    cfg = json.loads((adapter / "traj_config.json").read_text())
    normalizer = TrajectoryNormalizer.from_state_dict(cfg["normalizer"])

    # val에서 trajectory 샘플만
    recs = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("task") == "trajectory"]
    if cfg.get("reasoning_only", False):                   # 학습이 reasoning 주석 프레임만 썼으면 평가도 동일 필터(일관)
        n_all = len(recs)
        recs = [r for r in recs if has_reasoning_annotation(r)]
        print(f"reasoning_only: reasoning 주석 프레임만 {len(recs)}/{n_all} 평가 (cfg 정렬)")
    if args.limit:
        recs = recs[: args.limit]
    if not recs:
        raise SystemExit(f"no trajectory samples in {args.manifest}")

    prompt_tpl = TRAJ_PROMPT_FILE.read_text()
    # 학습 때의 vlm_mode를 복원: full=학습된 전체 가중치 로드 / lora=베이스+어댑터 / frozen=베이스만.
    vlm_mode = cfg.get("vlm_mode", "frozen")
    print(f"loading VLM {cfg['model']} (mode={vlm_mode}) + DiT head {adapter.name} ...")
    processor = load_processor(cfg["model"])                 # 학습과 동일 비전토큰 캡
    if vlm_mode == "full":
        from transformers import AutoModelForImageTextToText
        vlm = AutoModelForImageTextToText.from_pretrained(
            str(adapter / "vlm"), dtype=torch.bfloat16, device_map="cuda:0").eval()
    elif vlm_mode == "lora":
        from peft import PeftModel
        vlm = load_vlm(cfg["model"], "frozen")              # 베이스 4-bit
        vlm = PeftModel.from_pretrained(vlm, str(adapter / "vlm")).eval()
    else:
        vlm = load_vlm(cfg["model"], "frozen")
    dit = TrajectoryDiT(cond_dim=cfg["cond_dim"], n_points=cfg["n_points"],
                        point_dim=cfg.get("point_dim", 2),   # 3=논문(fwd,left,θ)/2=구형(하위호환)
                        d_model=cfg["d_model"], n_layers=cfg["n_layers"],
                        ego_dim=cfg.get("ego_dim", 0),
                        ego_as_state_token=cfg.get("ego_as_state_token", False),
                        beta_alpha=cfg.get("beta_alpha", 2.0),   # 학습 전용(sample 무영향)이나 재구성 일관
                        beta_beta=cfg.get("beta_beta", 2.0),
                        cross_attn=cfg.get("cross_attn", False)).to("cuda:0")  # VLM 시퀀스 cross-attention 여부
    dit.load_state_dict(torch.load(adapter / "dit_head.pt", map_location="cuda:0"))
    dit.eval()
    temporal_on = cfg.get("temporal", False)              # 학습이 과거뷰를 썼으면 평가도 동일하게(일관)
    man_thr = cfg.get("maneuver_lateral_thr", -1.0)       # selective-view 게이팅(학습과 동일 thr)
    print(f"evaluating {len(recs)} trajectory samples (ODE steps={args.steps}, "
          f"temporal={'ON' if temporal_on else 'OFF'}, "
          f"selective-view={'thr='+str(man_thr)+'m' if man_thr is not None and man_thr >= 0 else 'OFF(8뷰)'})\n")

    from evaluation.planning_metrics import planning_scores, aggregate, format_table

    # 추론 속도 계측(샘플당 VLM encode + DiT ODE sample, 즉 "1 프레임 planning 추론" 전체 wall-clock).
    #   ⚠️ GPU는 비동기 실행이라 커널 큐잉만으로는 시간이 부정확 → 구간 앞뒤로 cuda.synchronize() 필수.
    #   워밍업(cuDNN 알고리즘 탐색·커널 컴파일 등)으로 첫 샘플이 비정상적으로 느릴 수 있어 통계에서 제외
    #   (표준 벤치마크 관례). 게이팅(selective-view) 중이면 뷰 수가 샘플마다 달라 속도 자체가 변동하는 게
    #   정상 — 이 분포(Fastest/Slowest/Median/Mean)가 곧 "게이팅이 속도에 주는 실효"의 근거가 된다.
    import time
    infer_times_ms = []                                    # 워밍업(첫 샘플) 제외 나머지 전부(초→ms)

    rows, ades, fdes, cv_ades, cv_fdes, plan_rows = [], [], [], [], [], []
    for i, rec in enumerate(recs, 1):
        prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():                              # 평가: VLM forward도 grad 불필요
            _ml = rec_maneuver(rec)
            cvec, mem, mem_mask = encode_condition(vlm, processor,
                                                   gate_views(rec["images"], _ml, man_thr), prompt,
                                                   gate_views(history_for(rec, temporal_on), _ml, man_thr))
            cond = cvec.unsqueeze(0).to("cuda:0"); mem = mem.to("cuda:0"); mem_mask = mem_mask.to("cuda:0")
        ego = torch.tensor([ego_vec(rec, dit.ego_dim)], dtype=torch.float32, device="cuda:0") \
            if dit.ego_dim > 0 else None
        pred_norm = dit.sample(cond, steps=args.steps, deterministic=True, ego=ego,
                               mem=mem, mem_mask=mem_mask)[0]  # (T,point_dim) 정규화(결정론)
        if torch.cuda.is_available():
            torch.cuda.synchronize()                        # GPU 큐 비우고 나서 시간 확정(비동기 실행 보정)
        if i > 1:                                           # 첫 샘플=워밍업 → 통계 제외
            infer_times_ms.append((time.perf_counter() - t0) * 1000.0)
        pred_wp = normalizer.denormalize(pred_norm.cpu()).numpy()  # 미터 공간으로 역정규화 (T,3)[fwd,left,θ]@2Hz
        gt_wp = np.asarray(rec["output"]["waypoints"])
        # 논문 ADE: 예측/GT를 Δt=0.1s(10Hz)로 51 poses(현재 pose 포함) 보간 후 평가(planning_metrics도 10Hz 가정).
        pred = upsample_waypoints(pred_wp)                 # (51,2)[fwd,left]@10Hz
        gt = upsample_waypoints(gt_wp)                     # (51,2)
        ade, fde = displacement_errors(pred, gt)
        cv_ade, cv_fde = displacement_errors(constant_velocity(gt), gt)
        ades.append(ade); fdes.append(fde); cv_ades.append(cv_ade); cv_fdes.append(cv_fde)
        ps = planning_scores(rec, pred, gt)                # NC/DA/EP/CF/HL/NPS (Table 3, 10Hz)
        plan_rows.append(ps)
        rows.append({"id": rec["id"], "ade": round(ade, 3), "fde": round(fde, 3),
                     "cv_ade": round(cv_ade, 3), "cv_fde": round(cv_fde, 3),
                     **{k: round(v, 3) for k, v in ps.items()},
                     "pred_final": [round(float(x), 2) for x in pred[-1]],
                     "gt_final": [round(float(x), 2) for x in gt[-1]]})
        t_tag = f"  t={infer_times_ms[-1]:.1f}ms" if i > 1 else "  t=(warmup, 통계제외)"
        print(f"[{i}/{len(recs)}] {rec['id']}  ADE={ade:.2f}m FDE={fde:.2f}m  (cv: {cv_ade:.2f}/{cv_fde:.2f}){t_tag}")

    # adapter가 REPO 하위면 상대경로로, 아니면(예: 상대경로 입력) 그대로 문자열화(ValueError 방지).
    try:
        adapter_str = str(adapter.resolve().relative_to(REPO))
    except ValueError:
        adapter_str = str(adapter)
    plan_agg = aggregate(plan_rows)                       # NC_mean..NPS_mean + n_nps
    # 추론 속도 통계(ms, 워밍업 제외 n=len(recs)-1개 기준). 정렬 후 fastest=최솟값, slowest=최댓값,
    #   median=중앙값(짝수개면 두 중앙값 평균), mean=산술평균.
    t_sorted = sorted(infer_times_ms)
    n_t = len(t_sorted)
    if n_t:
        median_ms = (t_sorted[n_t // 2] if n_t % 2 else (t_sorted[n_t // 2 - 1] + t_sorted[n_t // 2]) / 2)
        speed_stats = {
            "n_timed": n_t,                                # 워밍업(첫 샘플) 제외 표본 수
            "fastest_ms": round(t_sorted[0], 2),
            "slowest_ms": round(t_sorted[-1], 2),
            "median_ms": round(median_ms, 2),
            "mean_ms": round(sum(t_sorted) / n_t, 2),
        }
    else:                                                  # 샘플 1개뿐이면 워밍업만 있고 계측 표본 0
        speed_stats = {"n_timed": 0, "fastest_ms": None, "slowest_ms": None, "median_ms": None, "mean_ms": None}
    metrics = {
        "n_samples": len(recs),
        "model": cfg["model"], "adapter": adapter_str,
        "ADE_mean": round(float(np.mean(ades)), 3), "FDE_mean": round(float(np.mean(fdes)), 3),
        "ADE_baseline_cv": round(float(np.mean(cv_ades)), 3),
        "FDE_baseline_cv": round(float(np.mean(cv_fdes)), 3),
        "ode_steps": args.steps,
        "maneuver_lateral_thr": man_thr,                   # 이 속도 통계가 어떤 게이팅 조건에서 나왔는지 기록
        "inference_speed_ms": speed_stats,                 # 샘플당 VLM encode+DiT sample wall-clock(ms), 워밍업 제외
        **plan_agg,
    }

    out_dir = eval_out_dir(cfg)
    pred_path = out_dir / f"{args.tag}_predictions.jsonl"
    with open(pred_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics_path = out_dir / f"{args.tag}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n=== trajectory metrics (meters) ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n=== planning metrics (nuReasoning Table 3 style, 5s horizon, n={len(recs)}) ===")
    print(format_table(plan_agg, ade=metrics["ADE_mean"]))
    print(f"(NPS 유효 표본 {plan_agg.get('n_nps', 0)}/{len(recs)}; 미니 근사 — planning_metrics.py 주석 참고)")
    print(f"\n=== inference speed (ms/sample, VLM encode + DiT sample, n={speed_stats['n_timed']}, "
          f"1번째 워밍업 제외, selective-view thr={man_thr}) ===")
    if speed_stats["n_timed"]:
        print(f"  Fastest: {speed_stats['fastest_ms']:8.2f} ms")
        print(f"  Slowest: {speed_stats['slowest_ms']:8.2f} ms")
        print(f"  Median : {speed_stats['median_ms']:8.2f} ms")
        print(f"  Mean   : {speed_stats['mean_ms']:8.2f} ms")
    else:
        print("  (표본 부족 — n_samples<=1)")
    print(f"\nwrote {metrics_path.relative_to(REPO)} and {pred_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
