"""zero-shot vs fine-tuned 비교표 생성 (스펙 §9 완료기준).

두 metrics JSON(예: results/zeroshot_metrics.json, results/finetuned_metrics.json)을 읽어
longitudinal/lateral의 accuracy·macro-F1과 파싱/매핑 실패를 나란히 표로 만들고 Δ를 계산한다.
개선이 작거나 음수여도 그대로 보고(스펙: 정직하게).

Run:  python -m evaluation.compare [--baseline zeroshot] [--model finetuned]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


def _row(name, a, b):
    d = b - a
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "=")
    return f"| {name} | {a:.4f} | {b:.4f} | {arrow} {d:+.4f} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="zeroshot", help="기준(보통 zero-shot) tag")
    ap.add_argument("--model", default="finetuned", help="비교 대상(보통 파인튜닝) tag")
    args = ap.parse_args()

    base = json.loads((RESULTS / f"{args.baseline}_metrics.json").read_text())
    new = json.loads((RESULTS / f"{args.model}_metrics.json").read_text())

    lines = [
        f"# {args.baseline} vs {args.model}",
        "",
        f"- baseline: `{base.get('model')}` (adapter={base.get('adapter')})",
        f"- model:    `{new.get('model')}` (adapter={new.get('adapter')})",
        f"- n_samples: {base.get('n_samples')} (동일 val manifest)",
        "",
        f"| metric | {args.baseline} | {args.model} | Δ |",
        "|---|---|---|---|",
        _row("longitudinal acc", base["longitudinal"]["accuracy"], new["longitudinal"]["accuracy"]),
        _row("longitudinal macro-F1", base["longitudinal"]["macro_f1"], new["longitudinal"]["macro_f1"]),
        _row("lateral acc", base["lateral"]["accuracy"], new["lateral"]["accuracy"]),
        _row("lateral macro-F1", base["lateral"]["macro_f1"], new["lateral"]["macro_f1"]),
        _row("parse-fail rate", base["parse_fail_rate"], new["parse_fail_rate"]),
        "",
        f"- pred never-predicted classes (collapse) 회복 여부는 `{args.model}_findings.md` 참조.",
    ]
    out = RESULTS / f"compare_{args.baseline}_vs_{args.model}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
