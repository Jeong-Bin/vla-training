"""Render a metrics JSON + predictions JSONL into portfolio artifacts.

Produces (per tag): a side-by-side confusion-matrix PNG and a short markdown findings
summary (headline metrics, prediction-distribution / class-collapse check, and a few
concrete failure cases). Shared by zero-shot and fine-tuned eval so the
two are reported identically.

Run:  python -m evaluation.report --tag zeroshot
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
DEFAULT_MANIFEST = REPO / "data" / "eval" / "zeroshot_manifest.jsonl"

# GT Risk level 심각도 순위 (Unsafe가 가장 위험). 매칭 안 됨/없음은 0.
RISK_RANK = {"Unsafe": 3, "Suboptimal": 2, "Safe": 1, None: 0}


# matched_risk: 모델이 예측한 (종,횡) 행동이 이 프레임의 GT counterfactual 액션 중
# 하나와 일치하면, 그 액션의 GT Risk level과 Reason을 반환(여럿이면 가장 심각한 것).
# 일치 없으면 (None, None). → "모델이 GT가 위험하다고 표시한 행동을 골랐는가"의 데이터 근거.
# ⚠️ 주의(정직성): canonical 택소노미는 'gently accelerate'(Safe)와 'quickly accelerate'(Unsafe)를
#    둘 다 accelerate로 합치므로, 같은 canonical에 여러 등급이 걸리면 worst-case(가장 심각)를 택함.
#    따라서 이 지표는 보수적(위험을 과대평가할 수 있음) — 정렬/주의환기용이지 정식 채점이 아님.
def matched_risk(row: dict) -> tuple:
    pl, pa = row.get("pred_long_canon"), row.get("pred_lat_canon")
    best = None
    for a in row.get("gt_counterfactual") or []:
        if a.get("long") == pl and a.get("lat") == pa and a.get("risk"):
            if best is None or RISK_RANK.get(a["risk"], 0) > RISK_RANK.get(best["risk"], 0):
                best = a
    return (best["risk"], best.get("reason", "")) if best else (None, None)


# plot_confusion: metrics dict의 혼동행렬을 종/횡 2개 패널 PNG로 렌더.
# 각 칸에 개수를 표기하고, 제목에 accuracy·macro-F1을 함께 표시.
def plot_confusion(metrics: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))   # 좌: 종방향, 우: 횡방향
    for ax, axis in zip(axes, ("longitudinal", "lateral")):
        m = metrics[axis]
        cm = np.array(m["confusion"])           # cm: 혼동행렬(행=GT, 열=예측)
        labels = m["labels"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("ground truth")
        ax.set_title(f"{axis}\nacc={m['accuracy']:.3f}  macroF1={m['macro_f1']:.3f}")
        thresh = cm.max() / 2 if cm.max() else 0
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")
    fig.suptitle(f"{metrics.get('model', '?')}  ·  n={metrics['n_samples']}", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# write_findings: 결과를 사람이 읽는 마크다운 요약으로 저장.
# 헤드라인 지표 + 예측 분포(클래스 붕괴 점검) + 안전 우선 정렬된 실패사례 몇 개.
def write_findings(metrics: dict, rows: list[dict], out_md: Path) -> None:
    long_pred = Counter(r["pred_long_canon"] for r in rows)  # 종방향 예측 분포
    lat_pred = Counter(r["pred_lat_canon"] for r in rows)    # 횡방향 예측 분포
    long_gt = Counter(r["gt_long_canon"] for r in rows)      # 종방향 GT 분포
    lat_gt = Counter(r["gt_lat_canon"] for r in rows)        # 횡방향 GT 분포
    
    '''
    종방향(longitudinal) = "얼마나 빠르게" — 속도 제어 축
        차량의 진행 방향(앞뒤)으로의 가감속. 액셀/브레이크로 제어.
        클래스: accelerate(가속) / maintain(정속) / slow_down(감속) / stop(정지)
        "앞에 보행자/적색신호 → 멈춰야 하나?"를 답하는 축.
    
    횡방향(lateral) = "어디로 향할지" — 조향/차로 축
        좌우 방향으로의 움직임. 스티어링으로 제어.
        클래스: keep_lane / nudge_left·right(차로 내 미세이동) / change_left·right(차로변경) / turn_left·right(회전)
        "차로 유지? 옆 차로로? 좌회전?"을 답하는 축.
    '''

    # classes that exist in GT but the model never predicts (collapse)
    # GT엔 있는데 모델이 한 번도 예측 안 한 클래스 = '클래스 붕괴' 신호
    long_never = sorted(set(long_gt) - set(long_pred))
    lat_never = sorted(set(lat_gt) - set(lat_pred))

    # 모델 예측이 GT counterfactual에서 어떤 위험등급으로 분류되는지 집계 (데이터 근거 지표).
    # 특히 Unsafe = "모델이 GT가 위험하다고 명시한 행동을 골랐다".
    risk_choice = Counter(matched_risk(r)[0] for r in rows)
    n_unsafe = risk_choice.get("Unsafe", 0)

    # surface the most safety-relevant misses first: GT wants to stop/slow but the
    # model chose a more aggressive longitudinal action.
    # aggressive: 종방향 '공격성' 점수(클수록 적극적). GT는 멈추려는데 모델이 더공격적으로 답한 케이스(점수차 큰 순)를 위로 정렬
    # → 안전상 치명적 실패를 먼저 노출. 직접 창작한 sample failure cases의 정렬 기준일 뿐 지표에 영향은 없음
    aggressive = {"accelerate": 2, "maintain": 1, "slow_down": 0, "stop": -1}
    wrong = sorted(
        (r for r in rows if not (r["long_correct"] and r["lat_correct"])),
        key=lambda r: aggressive.get(r["pred_long_canon"], 0) - aggressive.get(r["gt_long_canon"], 0),
        reverse=True,
    )
    lines = [
        f"# {out_md.stem.replace('_findings', '')} — eval findings",
        "",
        f"- **model**: `{metrics.get('model')}`  ·  **prompt**: `{metrics.get('prompt')}`  ·  **n**: {metrics['n_samples']}",
        f"- **parse-fail rate**: {metrics['parse_fail_rate']}  ·  unmapped pred long/lat: "
        f"{metrics['pred_unmapped_long']}/{metrics['pred_unmapped_lat']}",
        f"- **longitudinal**: acc={metrics['longitudinal']['accuracy']}  macro-F1={metrics['longitudinal']['macro_f1']}",
        f"- **lateral**: acc={metrics['lateral']['accuracy']}  macro-F1={metrics['lateral']['macro_f1']}",
        f"- **GT-graded action choice** (pred matched to GT counterfactual, worst-case grade): {dict(risk_choice.most_common())}"
        + (f"  ⚠️ {n_unsafe}/{len(rows)} predictions = GT-flagged **Unsafe**" if n_unsafe else ""),
        "",
        "## prediction distribution (class collapse check)",
        f"- GT long: {dict(long_gt.most_common())}",
        f"- pred long: {dict(long_pred.most_common())}"
        + (f"  ⚠️ never predicts: {long_never}" if long_never else ""),
        f"- GT lat: {dict(lat_gt.most_common())}",
        f"- pred lat: {dict(lat_pred.most_common())}"
        + (f"  ⚠️ never predicts: {lat_never}" if lat_never else ""),
        "",
        "## sample failure cases",
    ]
    for r in wrong[:6]:
        lines += [
            f"- `{r['id']}` mission={r['mission']}",
            f"  - GT: long={r['gt_longitudinal']!r} lat={r['gt_lateral']!r} "
            f"→ canon ({r['gt_long_canon']}, {r['gt_lat_canon']})",
            f"  - PRED: ({r['pred_long_canon']}, {r['pred_lat_canon']})  — {r['pred_reasoning'][:120]}",
        ]

    # 두 번째 정렬: GT Risk level 기반. 모델이 택한 행동이 GT counterfactual에서 가장 위험하게
    # 분류된(주로 Unsafe) 케이스를 위로. aggressive(임의 휴리스틱)와 달리 데이터셋 라벨이 근거.
    risk_ranked = sorted(
        (r for r in rows if matched_risk(r)[0] is not None),
        key=lambda r: RISK_RANK.get(matched_risk(r)[0], 0),
        reverse=True,
    )
    lines += ["", "## sample failure cases — ranked by GT Risk level",
              "_(모델이 고른 행동이 GT counterfactual에서 받은 위험등급 순. Unsafe = GT가 위험하다고 명시한 행동을 모델이 선택.)_",
              "_(주의: canonical 매핑상 같은 행동에 여러 등급이 걸리면 worst-case 채택 → 보수적 지표.)_"]
    for r in risk_ranked[:6]:
        risk, reason = matched_risk(r)
        correct = "✓ correct-but-graded" if (r["long_correct"] and r["lat_correct"]) else "✗ wrong"
        lines += [
            f"- `{r['id']}` mission={r['mission']}  **[{risk}]** ({correct})",
            f"  - PRED: ({r['pred_long_canon']}, {r['pred_lat_canon']})  vs  GT ({r['gt_long_canon']}, {r['gt_lat_canon']})",
            f"  - GT가 이 행동을 {risk}로 본 이유: {reason[:160]}",
        ]

    out_md.write_text("\n".join(lines) + "\n")


# main: --tag로 지정된 results/{tag}_metrics.json·{tag}_predictions.jsonl을 읽어
# 혼동행렬 PNG와 findings.md를 생성. (zero-shot/파인튜닝 모두 같은 방식으로 리포트)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zeroshot")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="GT counterfactual을 가져올 평가셋(예측 파일에 없을 때 id로 조인)")
    args = ap.parse_args()

    metrics = json.loads((RESULTS / f"{args.tag}_metrics.json").read_text())
    rows = [json.loads(l) for l in (RESULTS / f"{args.tag}_predictions.jsonl").read_text().splitlines() if l.strip()]

    # 예측 파일에 gt_counterfactual이 없으면(구버전 실행) manifest에서 id로 조인해 채움.
    mpath = Path(args.manifest)
    if mpath.exists():
        cf_by_id = {m["id"]: m.get("gt_counterfactual", [])
                    for m in (json.loads(l) for l in mpath.read_text().splitlines() if l.strip())}
        for r in rows:
            r.setdefault("gt_counterfactual", cf_by_id.get(r["id"], []))

    png = RESULTS / f"{args.tag}_confusion.png"
    md = RESULTS / f"{args.tag}_findings.md"
    plot_confusion(metrics, png)
    write_findings(metrics, rows, md)
    print(f"wrote {png.relative_to(REPO)} and {md.relative_to(REPO)}")


if __name__ == "__main__":
    main()
