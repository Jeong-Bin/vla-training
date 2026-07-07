"""텍스트 SFT 학습 (objective=text_sft) — 베이스 VLM(기본 Qwen3-VL-2B, 후보 Qwen2.5-VL-3B)을
SFT 데이터로 4-bit QLoRA 파인튜닝.

(파일명 메모: 학습 *방법*(QLoRA)이 아니라 *objective*(text_sft)로 명명한다 — QLoRA는 trajectory 경로의
`--vlm-mode lora`에서도 쓰이므로, 두 학습기는 objective(text_sft↔trajectory)로 구분한다.)

학습 objective 분기(논문 nuVLA의 두 출력 형태에 대응):
  - **text_sft** (이 모듈): driving(action+reasoning) + spatial(객체) — 텍스트 토큰 생성 SFT.
  - **trajectory** (`train_traj_reas.py`): VLM + flow-matching DiT 헤드로 연속 궤적 생성.
두 경로는 같은 train.jsonl을 `task` 필드로 공유한다. 이 모듈은 `--tasks`로 driving/spatial만 골라
쓰고 trajectory 레코드는 건너뛴다(궤적은 텍스트 SFT 대상이 아니므로 별도 모듈에서 학습).

베이스 모델 id는 `nureasoning.DEFAULT_MODEL`(단일 출처)에서 가져온다 — `--model`로 후보 교체 가능.

스펙 §9: 베이스는 평가와 동일 계열(공정 비교), `peft`+`bitsandbytes` 4-bit QLoRA, 8뷰
surround 카메라, 작은 규모(수백 샘플·1~3 epoch). 학습 후 **평가와 동일 하니스**(`run_zeroshot.py
--adapter`)로 val 재평가 → zero-shot 대비 비교.

데이터 형식은 `sft_data.chat_format.to_qwen_chat`(학습=평가 동일 프롬프트 + canonical JSON
타깃)을 그대로 재사용한다. Collator가 user(이미지+프롬프트) 토큰을 -100으로 가리고 assistant
타깃 토큰만 loss에 들어가게 만든다.

⚠️ 컴퓨트(DECISIONS): GPU0 결함 → `CUDA_VISIBLE_DEVICES=0,1,2,3`. VRAM 부족 시 batch↓/grad-accum↑/
LoRA rank↓로 대응(아래 인자).

멀티-GPU: HF `Trainer`는 `torchrun --nproc_per_node=N`으로 띄우면 환경변수(LOCAL_RANK/WORLD_SIZE)를
자동 인식해 **DDP**(GPU당 1프로세스, gradient all-reduce 동기화)로 학습한다. 우리가 손댈 건 4-bit 모델을
이 rank의 GPU(local_rank)에 올리는 `device_map`뿐(나머지 DDP·샤딩·rank0 저장은 Trainer가 처리).
DataParallel이 아니라 DDP인 이유: 4-bit QLoRA·device_map 고정과 호환되는 유일한 멀티-GPU 경로.

Run(단일 GPU):  CUDA_VISIBLE_DEVICES=0 python -m training.train_text_sft [--epochs 2] [--lora-rank 16] ...
Run(멀티 GPU): torchrun --nproc_per_node=4 -m training.train_text_sft --epochs 2 --grad-accum 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# DEFAULT_MODEL·SEED·load_processor(비전토큰 캡)는 학습·평가 공통이라 중립 모듈에서 가져온다(단일 출처).
from nureasoning import DEFAULT_MODEL, IMAGE_MAX_PIXELS, SEED, load_image, load_processor  # noqa: E402
from sft_data.chat_format import to_qwen_chat  # noqa: E402  (학습=평가 동일 포맷)
from training import ddp                         # noqa: E402  (torchrun DDP: local_rank/rank0 판정)
from training.run_logging import RunLogger       # noqa: E402  (실시간 step·epoch 종합 로깅)
from transformers import TrainerCallback         # noqa: E402  (RunLogCallback 베이스)

DEFAULT_TRAIN = REPO / "data" / "sft" / "train.jsonl"
DEFAULT_VAL = REPO / "data" / "sft" / "val.jsonl"
DEFAULT_OUT = REPO / "models" / "text_sft"

# LoRA를 붙일 모듈 = LLM(Qwen2)의 attention·MLP 투영. 비전 타워는 이름이 달라 자동 제외 → 동결.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


'''
train.jsonl
   │  SFTDataset.__getitem__      (디스크 → raw dict)
   ▼
raw record (dict)
   │  to_qwen_chat                (dict → Qwen 대화 포맷)   [chat_format.py]
   ▼
messages
   │  QwenVLCollator._one/__call__ (이미지 로드+토큰화+마스킹+패딩)
   ▼
batch tensors (input_ids, pixel_values, labels …)
   │  ← Trainer가 만든 DataLoader가 이 과정을 배치마다 호출
   ▼
model.forward → loss
'''


# SFTDataset: train.jsonl의 원본 SFT 레코드를 보관(실제 텐서화는 collator가 담당).
# tasks 필터로 text_sft 대상(driving/spatial)만 남기고 trajectory 레코드는 제외(별도 모듈 학습).
class SFTDataset:
    def __init__(self, path: Path, limit: int = 0, tasks: tuple = ("driving", "spatial")):
        records = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        if tasks:
            records = [r for r in records if r.get("task", "driving") in tasks]
        if limit:
            records = records[:limit]
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        return self.records[i]


# QwenVLCollator: SFT 레코드 배치 → 모델 입력 텐서. user(이미지+프롬프트) 구간은 -100으로 가리고
# assistant 타깃만 라벨로 남긴다(프롬프트만 두 번 토크나이즈해 길이를 재는 표준 기법).
class QwenVLCollator:
    def __init__(self, processor, max_len: int = 2048):
        self.processor = processor
        self.max_len = max_len
        self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def _one(self, rec):
        chat = to_qwen_chat(rec)
        msgs = chat["messages"]
        # user content interleaves N labelled view images; load them all in order.
        img_paths = [c["image"] for c in msgs[0]["content"] if c.get("type") == "image"]
        images = [load_image(p) for p in img_paths]            # 정사각형 정책 공통 적용(IMAGE_SQUARE)
        full_text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)

        full = self.processor(text=[full_text], images=images, return_tensors="pt")
        prompt = self.processor(text=[prompt_text], images=images, return_tensors="pt")

        ids = full["input_ids"][0]
        labels = ids.clone()
        plen = prompt["input_ids"].shape[1]      # user+이미지+"assistant\n" 길이 → 여기까지 마스크
        labels[:plen] = -100
        ids, labels = ids[: self.max_len], labels[: self.max_len]
        attn = full["attention_mask"][0][: self.max_len]
        return ids, labels, attn, full["pixel_values"], full["image_grid_thw"]

    def __call__(self, batch):
        import torch

        ids_l, lab_l, attn_l, pix_l, grid_l = zip(*(self._one(r) for r in batch))
        maxn = max(x.size(0) for x in ids_l)

        def pad(seq, value):
            return torch.stack([
                torch.cat([s, torch.full((maxn - s.size(0),), value, dtype=s.dtype)]) for s in seq
            ])

        return {
            "input_ids": pad(ids_l, self.pad_id),
            "attention_mask": pad(attn_l, 0),
            "labels": pad(lab_l, -100),
            "pixel_values": torch.cat(pix_l, dim=0),       # Qwen-VL: 패치들을 dim0로 concat
            "image_grid_thw": torch.cat(grid_l, dim=0),    # 이미지별 grid(t,h,w)
        }


def build_model(model_id: str, rank: int, alpha: int, dropout: float, device_index: int = 0):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    # DDP: 각 rank의 4-bit 모델은 자기 GPU(local_rank)에 올린다(Trainer가 그 위에서 DDP 래핑).
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": device_index},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


# RunLogCallback: HF Trainer 콜백 → RunLogger로 우리 로깅 규칙 적용.
#   on_log    = Trainer가 logging_steps마다 train loss를 줄 때 → 실시간 한 줄 덮어쓰기(터미널만).
#                step은 epoch 단위로 환산("epoch 1 | 12/389") → 전역 step에서 현재 epoch 몫을 뺀다.
#   on_evaluate = eval 끝나면 eval_loss를 잡아 같은 epoch 종합에 "EVAL"로 실어줌.
#   on_epoch_end = epoch 평균 train loss + (있으면) eval_loss를 영구 줄+로그파일로.
# rank0만 동작(RunLogger 내부 is_main 가드). epoch 평균은 on_log의 loss들을 누적해 계산.
class RunLogCallback(TrainerCallback):
    def __init__(self, logger: RunLogger):
        self.logger = logger
        self._ep_sum = 0.0
        self._ep_n = 0
        self._last_eval = None
        self._spe = 0                             # steps_per_epoch(첫 step에서 산정)

    def _steps_per_epoch(self, state) -> int:
        # Trainer는 epoch당 step을 직접 안 주므로 max_steps/num_train_epochs로 추정.
        if self._spe:
            return self._spe
        if state.max_steps and state.num_train_epochs:
            self._spe = max(1, round(state.max_steps / state.num_train_epochs))
        return self._spe

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:        # eval/기타 로그는 무시(train loss만)
            return
        loss = float(logs["loss"])
        self._ep_sum += loss
        self._ep_n += 1
        spe = self._steps_per_epoch(state)
        gstep = int(state.global_step)
        ep = gstep // spe + 1 if spe else 1                      # 현재 epoch(1-base)
        step_in_ep = gstep - (ep - 1) * spe if spe else gstep    # epoch 안에서의 step
        self.logger.step(ep, step_in_ep, spe, loss=loss)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self._last_eval = {"val_loss": float(metrics["eval_loss"])}

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(round(state.epoch)) if state.epoch else 0
        avg = self._ep_sum / self._ep_n if self._ep_n else float("nan")
        self.logger.epoch(ep, avg, eval=self._last_eval, extra=f"total {int(state.global_step)}")
        self._ep_sum = 0.0
        self._ep_n = 0
        self._last_eval = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--val", default=str(DEFAULT_VAL), help="epoch eval용 val.jsonl(같은 task 필터)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--eval-every", type=int, default=1, help="N epoch마다 val eval(평균 val loss). 0=생략")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=3072)  # 8뷰(~1824) + spatial 타깃(15객체 ~500) + 여유
    ap.add_argument("--limit", type=int, default=0, help="subset train for a quick run (0=all)")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke test); -1=use epochs")
    ap.add_argument("--tasks", default="driving,spatial",
                    help="text_sft 태스크 필터(쉼표구분). trajectory는 train_traj_reas.py로 학습")
    args = ap.parse_args()

    import torch
    from transformers import Trainer, TrainingArguments

    torch.manual_seed(SEED)
    processor = load_processor(args.model)     # 평가와 동일한 비전토큰 캡
    tasks = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    train_ds = SFTDataset(Path(args.train), args.limit, tasks)
    collator = QwenVLCollator(processor, args.max_len)
    # epoch eval용 val(같은 task 필터). eval-every=0이면 미사용.
    do_eval = args.eval_every > 0
    val_ds = SFTDataset(Path(args.val), 0, tasks) if do_eval else None

    # 실행별 산출물/로깅 = results/text_sft/<timestamp>/. rank0만 기록(나머지 no-op).
    img_tok = IMAGE_MAX_PIXELS // (28 * 28)               # 비전토큰 상한(이미지당). 28×28px=토큰1.
    logger = RunLogger("text_sft", is_main=ddp.is_main())
    logger.info(f"text_sft | model={args.model} tasks={tasks} world_size={ddp.world_size()}")
    logger.info(f"config | epochs={args.epochs} batch_size={args.batch_size} grad_accum={args.grad_accum} "
                f"lr={args.lr} lora_rank={args.lora_rank} image={img_tok}tok(<= {IMAGE_MAX_PIXELS}px)")
    logger.info(f"train={len(train_ds)} val(eval)={len(val_ds) if val_ds else 0} "
                f"eval_every={args.eval_every}\n")

    # 4-bit 모델을 이 rank의 GPU(local_rank)에 배치 → Trainer가 그 위에서 DDP 구성.
    model = build_model(args.model, args.lora_rank, args.lora_alpha, args.lora_dropout,
                        device_index=ddp.local_rank())

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        # epoch eval: do_eval이면 매 epoch val loss 계산(eval_loss → 콜백이 epoch 종합에 실음).
        eval_strategy="epoch" if do_eval else "no",
        per_device_eval_batch_size=args.batch_size,
        disable_tqdm=True,                    # tqdm 진행바가 우리 \r 실시간 줄을 덮지 않게 끔
        report_to="none",
        remove_unused_columns=False,          # collator가 비표준 키(pixel_values 등)를 넘기므로 필수
        label_names=["labels"],
        seed=SEED,
        # DDP: 비전 타워가 동결돼 unused param이 생기므로 True 필요(Trainer는 torchrun 시 자동 DDP).
        ddp_find_unused_parameters=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=collator)
    trainer.add_callback(RunLogCallback(logger))
    trainer.train()

    # 저장: 모델(LoRA 어댑터)은 args.out(=models/text_sft, eval 기본 경로), 로그/run_info는 run_dir.
    out = Path(args.out)
    trainer.save_model(str(out))              # Trainer.save_model은 rank0에서만 실제 기록(내부 가드)
    if ddp.is_main():
        def _rel(p):                          # REPO 하위면 상대경로, 아니면(예: /tmp) 절대경로
            try:
                return str(Path(p).relative_to(REPO))
            except ValueError:
                return str(p)
        processor.save_pretrained(str(out))
        (out / "train_config.json").write_text(json.dumps(vars(args), indent=2))
        (logger.run_dir / "run_info.json").write_text(json.dumps(
            {"tag": "text_sft", "args": vars(args), "model_out": _rel(out)},
            ensure_ascii=False, indent=2))
        logger.info(f"saved LoRA adapter -> {out}")
        logger.info(f"run logs -> {_rel(logger.run_dir)}")
    logger.close()


if __name__ == "__main__":
    main()
