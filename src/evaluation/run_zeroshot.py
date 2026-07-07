"""zero-shot driving-decision eval (baseline for the fine-tuned model).

For each manifest frame: feed the ego vehicle's 8 surround-view cameras + mission to a
pretrained VLM, parse a JSON ``{longitudinal, lateral, reasoning}`` answer, normalise
prediction *and* GT to the canonical taxonomy with the same mapper, and score
longitudinal/lateral accuracy and macro-F1. Greedy decoding + fixed seed for reproducibility.

This module is also the shared harness re-used by the fine-tuned model: pass
a different ``--model`` / adapter and the same manifest to get a comparable number.

Run:  python -m evaluation.run_zeroshot \
        [--model Qwen/Qwen3-VL-2B-Instruct] [--manifest ...] [--limit N] [--tag zeroshot]
Use  CUDA_VISIBLE_DEVICES=0,1,2,3  (GPU0 is faulty per DECISIONS.md).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# DEFAULT_MODEL·SEED·load_processor(비전토큰 캡)는 학습·평가 공통이라 중립 모듈에서 가져온다(단일 출처).
from nureasoning import DEFAULT_MODEL, SEED, load_image, load_processor, resolve_path  # noqa: E402
from sft_data.chat_format import view_caption  # noqa: E402  (학습=평가 동일 뷰 캡션)
from evaluation import taxonomy as T  # noqa: E402

DEFAULT_MANIFEST = REPO / "data" / "eval" / "zeroshot_manifest.jsonl"  # 평가셋
PROMPT_FILE = REPO / "prompts" / "driving_multiview_v1.txt"  # 버전 관리되는 멀티뷰 프롬프트 템플릿
RESULTS = REPO / "results"

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)    # 모델 출력에서 첫 {...} JSON 블록을 뽑는 정규식


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
# load_model: 모델 id로 프로세서+VLM을 로드(bf16, GPU). transformers 5.12.1에서
# AutoModelForImageTextToText로 Qwen-VL(3-VL/2.5-VL) 로딩 확인됨. adapter가 주어지면 LoRA 어댑터를
# 베이스 위에 얹고 merge → 파인튜닝 모델을 zero-shot과 '동일 하니스'로 평가. (processor, model)
def load_model(model_id: str, adapter: str = ""):
    import torch
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(SEED)
    processor = load_processor(model_id)       # 비전토큰 캡 적용(학습과 동일)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda:0"
    )
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()      # 추론 속도 위해 LoRA를 베이스에 병합
    return model.eval(), processor


# generate: 라벨된 8뷰 이미지 + 프롬프트로 1회 채팅 생성. 각 이미지 앞에 view_caption(학습과
# 동일 문구)을 인터리브 → 채팅 템플릿 → greedy 디코딩(do_sample=False, 재현성) → 입력 길이만큼
# 잘라 어시스턴트 답변만 디코딩해 반환. views = [(view_name, PIL image)] 순서 보존.
def generate(model, processor, views: list, prompt: str, max_new_tokens: int = 256) -> str:
    """Multi-view chat generation; returns the decoded assistant text."""
    content: list = []
    for view, image in views:
        content.append({"type": "text", "text": view_caption(view)})
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    images = [image for _, image in views]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    import torch

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]  # 입력 프롬프트 토큰 제거
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
# parse_answer: 모델 출력에서 JSON을 파싱해 (longitudinal/lateral/reasoning, parse_ok) 반환.
# JSON 디코딩 실패 시 parse_ok=False, 그래도 원문을 양쪽 액션 키에 넣어 매퍼가 라벨을 건질 수 있게 하되(신호 보존), 실패 자체는 따로 집계함(파싱 실패율).
def parse_answer(text: str) -> tuple[dict, bool]:
    """Extract ``{longitudinal, lateral, reasoning}`` from model output.

    Returns ``(fields, parse_ok)``. ``parse_ok`` is False when no JSON object could
    be decoded; the raw text is still returned under both action keys so the taxonomy
    mapper can salvage a label, but the failure is counted.
    """
    m = _JSON_RE.search(text or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {
                    "longitudinal": str(obj.get("longitudinal", "")),
                    "lateral": str(obj.get("lateral", "")),
                    "reasoning": str(obj.get("reasoning", "")),
                }, True
        except json.JSONDecodeError:
            pass
    return {"longitudinal": text, "lateral": text, "reasoning": ""}, False


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
# _axis_metrics: 한 축(종/횡)의 GT·예측 리스트로 accuracy, macro-F1, 라벨목록,
# 혼동행렬, 클래스별 support(GT 개수)를 계산해 dict로 반환.
def _axis_metrics(gt: list[str], pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    labels = sorted(set(gt) | set(pred))        # GT∪예측에 등장한 모든 라벨(행/열 축)
    return {
        "accuracy": round(float(accuracy_score(gt, pred)), 4),
        "macro_f1": round(float(f1_score(gt, pred, labels=labels, average="macro", zero_division=0)), 4),
        "labels": labels,
        "confusion": confusion_matrix(gt, pred, labels=labels).tolist(),
        "support": {lab: gt.count(lab) for lab in labels},
    }


# evaluate: 전체 결과 행(rows)에서 종/횡 GT·예측을 모아 전체 지표 dict를 구성.
# 샘플수·파싱실패율·예측 매핑실패수와 종/횡 축별 지표(_axis_metrics)를 담음.
def evaluate(rows: list[dict]) -> dict:
    gt_long = [r["gt_long_canon"] for r in rows]   # 종방향 GT(표준 라벨)
    gt_lat = [r["gt_lat_canon"] for r in rows]     # 횡방향 GT
    pr_long = [r["pred_long_canon"] for r in rows]  # 종방향 예측(표준 라벨)
    pr_lat = [r["pred_lat_canon"] for r in rows]    # 횡방향 예측
    n = len(rows)
    return {
        "n_samples": n,
        "parse_fail_rate": round(sum(not r["parse_ok"] for r in rows) / n, 4) if n else None,
        "pred_unmapped_long": sum(p == T.UNMAPPED for p in pr_long),
        "pred_unmapped_lat": sum(p == T.UNMAPPED for p in pr_lat),
        "longitudinal": _axis_metrics(gt_long, pr_long),
        "lateral": _axis_metrics(gt_lat, pr_lat),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
# [한국어] main: 전체 평가 파이프라인. manifest 로드 → 모델 로드 → 프레임마다
#         (front 이미지 로드 → 프롬프트 생성 → generate → parse → 표준 매핑 → 정오 기록) →
#         지표 집계 → results/{tag}_metrics.json·{tag}_predictions.jsonl 저장 + 실패사례 출력.
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default="", help="LoRA 어댑터 경로(파인튜닝). 비우면 zero-shot 베이스.")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N samples (0=all)")
    ap.add_argument("--tag", default="zeroshot", help="output filename tag")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    manifest = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        manifest = manifest[: args.limit]
    if not manifest:
        raise SystemExit(f"empty manifest: {args.manifest}")

    prompt_tpl = PROMPT_FILE.read_text()
    print(f"loading model {args.model}" + (f" + adapter {args.adapter}" if args.adapter else "") + " ...")
    model, processor = load_model(args.model, args.adapter)
    print(f"evaluating {len(manifest)} samples\n")

    rows = []                                   # rows: 프레임별 결과(원본+예측+정오) 누적
    for i, rec in enumerate(manifest, 1):       # rec: manifest 한 줄(평가 샘플 1개)
        clip_dir = resolve_path(rec["clip_dir"], REPO)   # 절대경로면 그대로, 레거시 상대면 REPO 기준
        # 신규 manifest: images=[{view,path}]. 레거시(front_image) 폴백 지원.
        view_recs = rec.get("images") or [{"view": "front", "path": rec["front_image"]}]
        views = [(im["view"], load_image(clip_dir / im["path"])) for im in view_recs]  # 정사각형 정책 공통 적용
        # .replace (not .format): the template contains literal JSON braces
        prompt = prompt_tpl.replace("{mission}", rec.get("mission") or "drive safely")
        raw = generate(model, processor, views, prompt, args.max_new_tokens)  # 모델 원문 출력
        fields, parse_ok = parse_answer(raw)    # JSON 파싱 결과 + 성공여부
        pred_long = T.map_longitudinal(fields["longitudinal"])  # 예측 종방향 → 표준 라벨
        pred_lat = T.map_lateral(fields["lateral"])             # 예측 횡방향 → 표준 라벨
        row = {
            **rec,
            "raw_output": raw,
            "parse_ok": parse_ok,
            "pred_longitudinal": fields["longitudinal"],
            "pred_lateral": fields["lateral"],
            "pred_reasoning": fields["reasoning"],
            "pred_long_canon": pred_long,
            "pred_lat_canon": pred_lat,
            "long_correct": pred_long == rec["gt_long_canon"],
            "lat_correct": pred_lat == rec["gt_lat_canon"],
        }
        rows.append(row)
        print(f"[{i}/{len(manifest)}] {rec['id']}  "
              f"GT(l={rec['gt_long_canon']},lat={rec['gt_lat_canon']})  "
              f"PRED(l={pred_long},lat={pred_lat})  parse_ok={parse_ok}")

    metrics = evaluate(rows)
    metrics["model"] = args.model
    metrics["adapter"] = args.adapter or None
    metrics["prompt"] = PROMPT_FILE.name

    RESULTS.mkdir(parents=True, exist_ok=True)
    pred_path = RESULTS / f"{args.tag}_predictions.jsonl"
    with open(pred_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics_path = RESULTS / f"{args.tag}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    # a few failure cases for the report
    failures = [
        {"id": r["id"], "gt": [r["gt_long_canon"], r["gt_lat_canon"]],
         "pred": [r["pred_long_canon"], r["pred_lat_canon"]], "raw": r["raw_output"][:200]}
        for r in rows if not (r["long_correct"] and r["lat_correct"])
    ][:5]

    print("\n=== metrics ===")
    print(json.dumps({k: metrics[k] for k in
                      ("n_samples", "parse_fail_rate", "pred_unmapped_long",
                       "pred_unmapped_lat", "longitudinal", "lateral")},
                     ensure_ascii=False, indent=2))
    print(f"\nwrote {metrics_path.relative_to(REPO)} and {pred_path.relative_to(REPO)}")
    if failures:
        print(f"\n{len(failures)} sample failure case(s):")
        for f in failures:
            print(f"  {f['id']}: GT={f['gt']} PRED={f['pred']}")


if __name__ == "__main__":
    main()
