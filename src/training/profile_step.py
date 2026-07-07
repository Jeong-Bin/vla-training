"""한 학습 step의 시간 구성 프로파일 — 데이터로딩 / 전처리 / forward / backward / 통신(all-reduce).

DataLoader(num_workers)·표준 DDP 래퍼 전환이 값어치 있는지 판단하려면 "한 step에서 무엇에 시간을
쓰나"를 먼저 재야 한다. 이 스크립트는 train_traj_reas의 실제 빌딩블록(load_vlm/_build_views_content/
encode 경로/dit.flow_loss/ddp.sync_grads)을 그대로 써서 step을 재현하되, 5개 구간에 타이머를 심는다.

⚠️ GPU 커널은 **비동기**라 각 구간 경계에서 torch.cuda.synchronize()로 완료를 기다려야 시간이 정확하다.
   (sync 없이 재면 커널 '실행'이 아니라 '런치'만 재서 forward가 비현실적으로 짧게 나온다.)

구간:
  load       = 이미지 디스크 read + PIL 디코드 (_build_views_content) — DataLoader 워커가 offload 가능한 부분
  preprocess = Qwen 프로세서(chat_template + tokenize + 이미지 전처리) + .to(device) H2D 복사 — CPU 위주
  forward    = VLM forward(+LM) + DiT flow_loss (GPU)
  backward   = loss.backward() (GPU)
  comm       = ddp.sync_grads = rank 간 grad all-reduce (멀티-GPU에서만 >0; DDP 전환 시 이게 backward와 겹쳐짐)

Run(멀티-GPU, 통신까지 측정):
  cd src && CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      -m training.profile_step --vlm-mode lora --steps 20
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from nureasoning import DEFAULT_MODEL, TEMPORAL, TEMPORAL_HISTORY_OFFSET, load_processor  # noqa: E402
from training import ddp  # noqa: E402
from training.dit_head import TrajectoryDiT  # noqa: E402
from training.train_traj_reas import (  # noqa: E402
    DEFAULT_TRAIN, TRAJ_PROMPT_FILE, TrajDataset, _build_views_content, ego_vec,
    history_for, load_vlm, parse_reasoning_types, reasoning_target,
)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# _encode: encode_and_lm_loss와 동일 경로를 재현하되 (load, preprocess, forward)를 분리 계측하기 위해
#   인라인으로 편다. 반환: (cond (Dc,), lm_loss or None, {t_load, t_prep, t_fwd_vlm}).
def _encode_timed(model, processor, image_recs, prompt, reasoning, hist, device):
    t0 = time.perf_counter()
    content, images = _build_views_content(image_recs, prompt, hist)     # LOAD: 디스크→PIL
    t1 = time.perf_counter()

    user_msg = {"role": "user", "content": content}
    if reasoning:
        msgs = [user_msg, {"role": "assistant", "content": reasoning}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    else:
        text = processor.apply_chat_template([user_msg], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)   # PREPROCESS(+H2D)
    _sync()
    t2 = time.perf_counter()

    labels = inputs["input_ids"].clone() if reasoning else None          # 타이밍용(마스킹 생략)
    out = model(**inputs, labels=labels, output_hidden_states=True)      # VLM forward
    hs = out.hidden_states[-1]
    m = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)
    cond = ((hs * m).sum(1) / m.sum(1).clamp_min(1.0)).squeeze(0).float()
    lm = out.loss if reasoning else None
    return cond, lm, (t1 - t0, t2 - t1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--vlm-mode", default="lora", choices=["full", "lora", "frozen"])
    ap.add_argument("--reasoning-types", default="spatial,decision,counterfactual")
    ap.add_argument("--steps", type=int, default=20, help="계측 step 수")
    ap.add_argument("--warmup", type=int, default=3, help="계측 전 워밍업 step(CUDA init/autotune 제외)")
    args = ap.parse_args()

    device = ddp.setup()
    lr_idx = ddp.local_rank()
    main_proc = ddp.is_main()

    def log(*a):
        if main_proc:
            print(*a, flush=True)

    reasoning_types = parse_reasoning_types(args.reasoning_types)
    ds = TrajDataset(Path(args.train))
    if len(ds) == 0:
        raise SystemExit("no trajectory samples in train.jsonl — build SFT first.")
    prompt_tpl = TRAJ_PROMPT_FILE.read_text()
    n_points = len(ds.records[0]["output"]["waypoints"])
    ego_dim = len(ds.records[0].get("ego_state") or [])
    temporal_on = bool(TEMPORAL and any(r.get("history_images") for r in ds.records))
    train_vlm = args.vlm_mode != "frozen"

    log(f"profile | vlm-mode={args.vlm_mode} world_size={ddp.world_size()} steps={args.steps} "
        f"warmup={args.warmup} samples={len(ds)} temporal={'ON' if temporal_on else 'OFF'} "
        f"reasoning-types={reasoning_types or 'none'}")

    processor = load_processor(args.model)
    vlm = load_vlm(args.model, args.vlm_mode, device_index=lr_idx)

    # cond_dim probe (겸 첫 워밍업)
    with torch.no_grad():
        c0, _, _ = _encode_timed(vlm, processor, ds.records[0]["images"],
                                 prompt_tpl.replace("{mission}", ds.records[0].get("mission") or "drive safely"),
                                 None, history_for(ds.records[0], temporal_on), device)
    cond_dim = c0.shape[0]
    log(f"cond_dim (probed): {cond_dim}")

    dit = TrajectoryDiT(cond_dim=cond_dim, n_points=n_points, ego_dim=ego_dim).to(device)
    dit.set_cond_stats(torch.zeros(cond_dim), torch.ones(cond_dim))       # 타이밍용 항등 정규화(값 정확도 무관)
    if ego_dim > 0:
        dit.set_ego_stats(torch.zeros(ego_dim), torch.ones(ego_dim))
    dit.train()

    vlm_params = [p for p in vlm.parameters() if p.requires_grad] if train_vlm else []
    clip_params = list(dit.parameters()) + vlm_params
    opt = torch.optim.AdamW([{"params": dit.parameters(), "lr": 1e-4},
                             *([{"params": vlm_params, "lr": 5e-5}] if vlm_params else [])])

    ph = defaultdict(list)      # phase → [seconds per step]
    n_reas_steps = 0
    total = args.warmup + args.steps
    for s in range(total):
        # rank별로 다른 레코드(대표성) — sync_grads는 어떤 레코드든 collective 횟수만 맞으면 됨
        rec = ds.records[(s * ddp.world_size() + ddp.rank()) % len(ds)]
        prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
        reasoning = reasoning_target(rec, reasoning_types) if train_vlm else None
        hist = history_for(rec, temporal_on)

        opt.zero_grad(set_to_none=True)
        _sync(); t_start = time.perf_counter()

        # forward (내부에서 load/preprocess 분리 계측)
        if train_vlm:
            cond, lm, (t_load, t_prep) = _encode_timed(vlm, processor, rec["images"], prompt, reasoning, hist, device)
        else:
            with torch.no_grad():
                cond, lm, (t_load, t_prep) = _encode_timed(vlm, processor, rec["images"], prompt, None, hist, device)
        ego = torch.tensor([ego_vec(rec, ego_dim)], dtype=torch.float32, device=device) if ego_dim > 0 else None
        x1 = torch.tensor([rec["output"]["waypoints"]], dtype=torch.float32, device=device)
        flow = dit.flow_loss(x1, cond.unsqueeze(0), ego)
        loss = flow + (0.5 * lm if lm is not None else 0.0)
        _sync(); t_fwd_end = time.perf_counter()

        # backward
        if train_vlm or True:
            loss.backward()
        _sync(); t_bwd_end = time.perf_counter()

        # comm: rank 간 grad all-reduce (단일 GPU면 사실상 no-op → ~0)
        ddp.sync_grads(clip_params)
        _sync(); t_comm_end = time.perf_counter()

        opt.step()

        if s >= args.warmup:        # 워밍업 제외하고 집계
            ph["load"].append(t_load)
            ph["preprocess"].append(t_prep)
            ph["forward"].append(t_fwd_end - t_start - t_load - t_prep)   # forward = 총 - (load+prep)
            ph["backward"].append(t_bwd_end - t_fwd_end)
            ph["comm"].append(t_comm_end - t_bwd_end)
            ph["step_total"].append(t_comm_end - t_start)
            if reasoning:
                n_reas_steps += 1

    # 집계(rank0). 각 구간 평균 ms + step 대비 비중.
    if main_proc:
        order = ["load", "preprocess", "forward", "backward", "comm"]
        tot = st.mean(ph["step_total"])
        print("\n=== per-step time breakdown (rank0, mean over "
              f"{args.steps} steps, {n_reas_steps} had reasoning) ===")
        print(f"{'phase':<12}{'mean(ms)':>10}{'median(ms)':>12}{'% of step':>11}")
        for k in order:
            mean_ms = st.mean(ph[k]) * 1e3
            med_ms = st.median(ph[k]) * 1e3
            pct = 100.0 * st.mean(ph[k]) / tot
            print(f"{k:<12}{mean_ms:>10.1f}{med_ms:>12.1f}{pct:>10.1f}%")
        print(f"{'step_total':<12}{tot * 1e3:>10.1f}{st.median(ph['step_total']) * 1e3:>12.1f}{100.0:>10.1f}%")
        data_pct = 100.0 * (st.mean(ph["load"]) + st.mean(ph["preprocess"])) / tot
        print(f"\n데이터 준비(load+preprocess) = {data_pct:.1f}%  |  "
              f"GPU(forward+backward) = {100.0*(st.mean(ph['forward'])+st.mean(ph['backward']))/tot:.1f}%  |  "
              f"통신(comm) = {100.0*st.mean(ph['comm'])/tot:.1f}%")

    ddp.cleanup()


if __name__ == "__main__":
    main()
