"""(objective=trajectory) — 정성 평가: 궤적 시각화 + reasoning 생성 예시.

`eval_trajectory.py`가 ADE/FDE **숫자**를 낸다면, 이 스크립트는 "모델이 실제로 말이 되는 추론을
하는가"를 **눈으로** 확인하게 해준다. val의 trajectory 샘플 몇 개를 골라 두 가지를 낸다:

  1) **궤적 시각화**(PNG): 학습된 VLM+DiT로 미래 궤적을 ODE 샘플링 → ego-frame(전방=위, 좌측=왼쪽)에
     **예측(파랑) vs GT(검정)** waypoints를 겹쳐 격자 그림으로 저장. 궤적이 GT를 따라가는지 직관 확인.
  2) **reasoning 생성**(jsonl + 콘솔): 같은 장면에 대해 VLM이 `generate`로 만든 reasoning 텍스트를
     GT reasoning과 나란히 출력. full/lora 모드(VLM이 reasoning도 학습)에서 의미 있다.

`eval_trajectory.py`와 동일한 모델 복원·인코더를 재사용한다(학습=평가 일관).

Run:  CUDA_VISIBLE_DEVICES=4 conda run -n vla python -m evaluation.eval_qualitative \
        --adapter "$PWD/results/traj_reas/<timestamp>/model" [--n 18] [--per-page 6] [--steps 50]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nureasoning import SFT_VAL, load_image, load_processor, resolve_path, upsample_waypoints  # noqa: E402
from sft_data.chat_format import view_caption                         # noqa: E402
from evaluation.eval_trajectory import eval_out_dir                   # noqa: E402  (저장 위치 일관)
from evaluation.bev import render_bev                                # noqa: E402  (HD맵+객체박스+궤적 BEV)
from training.dit_head import TrajectoryDiT, TrajectoryNormalizer     # noqa: E402
from training.train_traj_reas import encode_condition, ego_vec, history_for, load_vlm   # noqa: E402

# 8뷰 3×3 surround 배치(view_8cam.py와 동일). 중앙(1,1)은 GT 텍스트용.
GRID_POS = {
    "front_left": (0, 0), "front": (0, 1), "front_right": (0, 2),
    "left": (1, 0),                          "right": (1, 2),
    "back_left": (2, 0), "back": (2, 1), "back_right": (2, 2),
}

DEFAULT_MANIFEST = SFT_VAL     # vlm.SFT_DIR 중앙 설정 따름(현재 data/sft_v2/val.jsonl). --manifest로 override.
TRAJ_PROMPT_FILE = REPO / "prompts" / "trajectory_plan_v1.txt"


# generate_reasoning: VLM에 (과거 8뷰 옵션+)현재 8뷰+프롬프트를 주고 reasoning 텍스트를 그리디 생성
#   (학습=평가 동일 프롬프트/시간맥락). history_recs가 있으면 train의 _build_views_content와 동일하게
#   과거 1 timestep을 현재 앞에 둔다 → 학습 때 본 시퀀스와 일치.
def generate_reasoning(vlm, processor, image_recs, prompt, max_new_tokens, history_recs=None):
    import torch

    from training.train_traj_reas import _build_views_content  # 학습과 동일 인터리브(과거/현재 마커 포함)

    content, images = _build_views_content(image_recs, prompt, history_recs)
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(vlm.device)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]        # 프롬프트 부분 잘라내고 생성분만
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# plot_grid: 샘플별 **BEV**(HD맵 + 객체 GT 박스 + pred/GT 궤적)를 격자로 그려 PNG 저장.
# 각 칸은 render_bev가 그 장면의 차선·경계·횡단보도·신호등 + 주변 객체 위에 궤적을 올린다
# (맵 매칭 실패해도 궤적은 폴백으로 그려짐). 전방=위(+x), 좌측=왼쪽(+y).
def plot_grid(samples, out_path):
    import math
    import matplotlib
    matplotlib.use("Agg")                                  # 헤드리스(파일 저장 전용)
    import matplotlib.pyplot as plt

    n = len(samples)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 5.0 * rows), squeeze=False)
    for k, s in enumerate(samples):
        ax = axes[k // cols][k % cols]
        rec = {"id": s["id"], "images": s["images"]}       # render_bev가 클립/프레임 추적에 쓰는 최소 레코드
        render_bev(ax, rec, s["gt"], s["pred"], s["ade"], s["fde"])
    for k in range(n, rows * cols):                        # 빈 칸 숨김
        axes[k // cols][k % cols].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# plot_8view: 한 샘플을 Figure 5 스타일로 그린다.
#   왼쪽 3×3 = 8뷰 카메라(라벨 없이 셀을 꽉 채워 크게), 가운데(1,1)=예측 요약, 오른쪽 세로 패널=BEV.
#   가운데 요약: **가운데 정렬**, 항목 사이 빈 줄, 라벨(mission/ADE/FDE/GT·Pred reasoning)만 굵게(mathtext).
# sample dict: id, images, mission, gt_reasoning, pred_reasoning, ade, fde, gt, pred.
def plot_8view(sample, out_path):
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    by_view = {im["view"]: im["path"] for im in sample["images"]}
    fig = plt.figure(figsize=(22, 12))
    # 열 구성: [카메라0,1,2] [spacer] [BEV]. spacer 열(폭 0.05)을 둬 HD맵을 카메라에서 오른쪽으로 띄운다.
    # left/right/top/bottom 여백 최소화 + wspace/hspace 작게 → 카메라가 공간을 꽉 채운다.
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 0.05, 1.25],
                          left=0.005, right=0.975, top=0.955, bottom=0.04,
                          wspace=0.03, hspace=0.03)
    # 8뷰 카메라(왼쪽 3열). 라벨(title) 없음. aspect="auto"로 셀을 꽉 채워 여백 제거(약간 늘어남 감수).
    for v, (r, c) in GRID_POS.items():
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")
        if v in by_view:
            ax.imshow(Image.open(resolve_path(by_view[v], REPO)).convert("RGB"), aspect="auto")
        else:
            ax.text(0.5, 0.5, f"{v}\n(missing)", ha="center", va="center")

    # 가운데(1,1): 예측 요약 — mathtext로 라벨만 굵게, 값은 보통 굵기. 가운데 정렬, 항목 사이 빈 줄.
    def bold(s):                                            # mathtext 굵게(공백은 \ 로 escape)
        return r"$\mathbf{" + s.replace(" ", r"\ ") + r"}$"

    def clean(s):                                          # 값에서 mathtext 깨는 $ 제거
        return (s or "").replace("$", "")

    def wrapc(s):                                          # 가독성 위해 줄바꿈(폭 넓혀 셀 좌우 여백 축소)
        return "\n".join(textwrap.wrap(s, 52)) or s

    trace = clean(sample.get("reasoning_trace") or "(none)")   # GT "Reasoning Trace"(Driving.Reasoning trace) 1개만
    blocks = [                                              # mission은 제목줄(suptitle)에 있으니 여기선 생략
        f"{bold('ADE')}: {sample['ade']:.2f}m, {bold('FDE')}: {sample['fde']:.2f}m",
        f"{bold('Reasoning Trace')}\n{wrapc(trace)}",
    ]
    centre = fig.add_subplot(gs[1, 1]); centre.axis("off")
    centre.text(0.5, 1.0, "\n\n".join(blocks), ha="center", va="top", fontsize=13,
                linespacing=1.35, transform=centre.transAxes)

    # 오른쪽 세로 패널(전 행 span, spacer 열 다음): BEV. 좌우(lat_rng)를 좁혀 세로로 길게(Figure 5).
    bev_ax = fig.add_subplot(gs[:, 4])
    rec = {"id": sample["id"], "images": sample["images"]}
    render_bev(bev_ax, rec, sample["gt"], sample["pred"], sample["ade"], sample["fde"],
               fwd_lo=-20.0, fwd_hi=60.0, lat_rng=22.0, pred_color="orange")   # 전방 60m~후방 -20m, pred=주황
    # 제목줄: id + "mission: <v>"(전체 굵게, mathtext). mission 값은 _ 등 escape 필요.
    miss = (sample.get("mission") or "—").replace("\\", "").replace("$", "").replace("_", r"\_")
    fig.suptitle(f"{sample['id']}  |  " + bold(f"mission: {miss}"), fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# select_diverse: pool에서 n개를 **균등 간격**으로 골라 다양한 장면을 확보(앞쪽 n개만 쓰면 비슷한
# 씬에 몰릴 수 있음). 인덱스를 0..len-1에 고르게 펼친 뒤 중복 제거(순서 보존). pool<=n이면 전체 반환.
def select_diverse(pool, n):
    if len(pool) <= n:
        return list(pool)
    if n <= 1:
        return [pool[0]]
    seen, out = set(), []
    for i in range(n):
        idx = round(i * (len(pool) - 1) / (n - 1))
        if idx not in seen:
            seen.add(idx)
            out.append(pool[idx])
    return out


# save_visuals: samples를 per_page개씩 페이지로 나눠 저장. 한 프레임당 **2종**의 이미지를 같은 폴더에 낸다:
#   1) {base}_3view.png    = 노트북 "Preview One Frame" 스타일(front_left/front/front_right 3카메라 +
#                            DRIVING/COUNTERFACTUAL reasoning 텍스트 + Spatial 객체 박스 + ego 오버레이).
#   2) {base}_8view_tj.png = 기존 8뷰 콜라주(8카메라 + 가운데 요약 + 우측 세로 BEV[pred=주황]). 가운데 요약의
#                            'Pred reasoning'은 GT 'Reasoning Trace'(Driving.Reasoning trace)로 대체.
#   base = 샘플 id에서 끝의 '_tj'를 뗀 것(기존 파일명 ..._tj_8view → ..._3view / ..._8view_tj).
#   그리고 페이지별 BEV 격자 {tag}_trajectories_{i}.png(개요)는 그대로 유지. 반환: 생성한 페이지 수.
def save_visuals(samples, out_dir, tag, per_page=6, reasoning_types=()):
    import math
    from evaluation.reasoning_view import (frame_context_from_sample, render_3view,
                                           reasoning_trace_of, parse_pred_reasoning)
    views_root = out_dir / f"{tag}_8view"
    n_pages = math.ceil(len(samples) / per_page) if samples else 0
    for pi in range(n_pages):
        page = samples[pi * per_page:(pi + 1) * per_page]
        idx = pi + 1
        plot_grid(page, out_dir / f"{tag}_trajectories_{idx}.png")     # 페이지 개요(BEV 격자)
        images_dir = views_root / f"{idx}_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        for s in page:
            # 원본 클립/프레임/reasoning JSON 복구(front 이미지 basename 매칭, 1회) → 3뷰 GT + Reasoning Trace 공용.
            clip_dir, frame, rjson, total = frame_context_from_sample(s)
            s["reasoning_trace"] = reasoning_trace_of(rjson)           # 8뷰 콜라주 가운데 요약(GT 1개)
            pred_parts = parse_pred_reasoning(s.get("pred_reasoning"))  # 생성 reasoning → {spatial,decision,cf}
            base = s["id"][:-3] if s["id"].endswith("_tj") else s["id"]
            render_3view(clip_dir, frame, rjson, total, images_dir / f"{base}_3view.png",   # 1) 3뷰(GT|Pred)
                         pred_parts=pred_parts, reasoning_types=reasoning_types)
            plot_8view(s, images_dir / f"{base}_8view_tj.png")                              # 2) 8뷰+BEV
    return n_pages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="train_traj_reas.py 출력 디렉터리")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--n", type=int, default=18, help="시각화·reasoning 예시 샘플 수(균등 간격 선택)")
    ap.add_argument("--per-page", type=int, default=6, help="격자 1장당 장면 수(페이지 분할 단위)")
    ap.add_argument("--steps", type=int, default=5, help="ODE 적분 스텝(논문 K=5)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="reasoning 생성 최대 토큰")
    ap.add_argument("--no-reasoning", action="store_true", help="reasoning 생성 건너뛰고 궤적 시각화만")
    ap.add_argument("--tag", default="trajectory")
    args = ap.parse_args()

    import numpy as np
    import torch

    adapter = Path(args.adapter)
    cfg = json.loads((adapter / "traj_config.json").read_text())
    normalizer = TrajectoryNormalizer.from_state_dict(cfg["normalizer"])

    # val에서 reasoning 있는 trajectory 샘플 우선 선택(정성 비교가 의미 있으려면 GT reasoning 필요).
    recs = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    recs = [r for r in recs if r.get("task") == "trajectory"]
    with_reas = [r for r in recs if r["output"].get("reasoning")]   # GT reasoning 있는 프레임만(폴백 없음)
    chosen = select_diverse(with_reas, args.n)                # 그중 균등 간격으로 다양하게
    if not chosen:
        raise SystemExit(f"no trajectory samples in {args.manifest}")

    # 모델 복원(eval_trajectory.py와 동일 규칙: vlm_mode별).
    vlm_mode = cfg.get("vlm_mode", "frozen")
    print(f"loading VLM {cfg['model']} (mode={vlm_mode}) + DiT {adapter.name} ...")
    processor = load_processor(cfg["model"])
    if vlm_mode == "full":
        from transformers import AutoModelForImageTextToText
        vlm = AutoModelForImageTextToText.from_pretrained(
            str(adapter / "vlm"), dtype=torch.bfloat16, device_map="cuda:0").eval()
    elif vlm_mode == "lora":
        from peft import PeftModel
        vlm = PeftModel.from_pretrained(load_vlm(cfg["model"], "frozen"), str(adapter / "vlm")).eval()
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
    prompt_tpl = TRAJ_PROMPT_FILE.read_text()
    do_reas = not args.no_reasoning and vlm_mode != "frozen"   # frozen은 VLM 미학습 → reasoning 무의미
    temporal_on = cfg.get("temporal", False)              # 학습이 과거뷰를 썼으면 평가도 동일하게(일관)
    print(f"qualitative eval on {len(chosen)} samples (reasoning={'on' if do_reas else 'off'}, "
          f"temporal={'on' if temporal_on else 'off'})\n")

    samples = []
    for i, rec in enumerate(chosen, 1):
        prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
        hist = history_for(rec, temporal_on)              # 과거 8뷰(temporal_on일 때만)
        # 1) 궤적 예측
        with torch.no_grad():
            cvec, mem, mem_mask = encode_condition(vlm, processor, rec["images"], prompt, hist)
            cond = cvec.unsqueeze(0).to("cuda:0"); mem = mem.to("cuda:0"); mem_mask = mem_mask.to("cuda:0")
        ego = torch.tensor([ego_vec(rec, dit.ego_dim)], dtype=torch.float32, device="cuda:0") \
            if dit.ego_dim > 0 else None
        pred_wp = normalizer.denormalize(dit.sample(cond, steps=args.steps, deterministic=True, ego=ego,
                                                    mem=mem, mem_mask=mem_mask)[0].cpu()).numpy()
        gt_wp = np.asarray(rec["output"]["waypoints"])
        pred = upsample_waypoints(pred_wp)                # (51,2) @10Hz — 매끄러운 BEV + 일관 ADE
        gt = upsample_waypoints(gt_wp)                    # (51,2)
        d = np.linalg.norm(pred - gt, axis=1)
        gt_reas = rec["output"].get("reasoning")
        # 2) Pred reasoning 생성(8뷰 가운데 요약에 사용). frozen/--no-reasoning이면 None.
        gen = generate_reasoning(vlm, processor, rec["images"], prompt, args.max_new_tokens, hist) if do_reas else None
        samples.append({"id": rec["id"], "gt": gt.tolist(), "pred": pred.tolist(),
                        "ade": float(d.mean()), "fde": float(d[-1]),
                        "images": rec["images"], "mission": rec.get("mission"),
                        "gt_reasoning": gt_reas,
                        "pred_reasoning": gen})          # 8뷰 가운데 요약에 표시
        print(f"[{i}/{len(chosen)}] {rec['id']}  ADE={d.mean():.2f}m FDE={d[-1]:.2f}m")
        if do_reas:
            print(f"    PRED: {gen[:200]}")
            print(f"    GT  : {(gt_reas or '(없음)')[:200]}\n")

    # 저장: 궤적 격자 PNG(페이지별) + 8뷰 콜라주(페이지 하위폴더) → 학습 실행 폴더(run_dir).
    # n개 장면을 per_page개씩 나눠 {tag}_trajectories_{i}.png + {tag}_8view/{i}_images/ 로 묶는다.
    out_dir = eval_out_dir(cfg)
    n_pages = save_visuals(samples, out_dir, args.tag, args.per_page,
                           reasoning_types=cfg.get("reasoning_types", []))   # Pred 열: 학습한 종류만

    views_root = out_dir / f"{args.tag}_8view"
    print(f"\nwrote {(out_dir / args.tag).relative_to(REPO)}_trajectories_1..{n_pages}.png  "
          f"({len(samples)}개 장면, {n_pages}페이지)")
    print(f"wrote {views_root.relative_to(REPO)}/1_images../  (8뷰 + 가운데 요약 + 우측 세로 BEV)")


if __name__ == "__main__":
    main()
