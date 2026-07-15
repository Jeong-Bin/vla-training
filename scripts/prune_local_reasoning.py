#!/usr/bin/env python3
"""로컬 보유 clip(part_1)에서 reasoning 프레임(∪ history) 외 카메라 이미지를 삭제해 용량을 줄인다.

기본은 **dry-run**: 무엇을/얼마나 삭제할지 요약만 출력하고 실제로는 안 지운다.
실제 삭제는 `--apply`를 명시할 때만. cameras/*.jpg만 대상(주석·metadata·map은 보존).

사용:
  # 1) 먼저 dry-run으로 확인 (아무것도 안 지움)
  python scripts/prune_local_reasoning.py --clips-root /home/etri/DATASET/nureasoning/clips
  # 2) 확인 후 실제 삭제
  python scripts/prune_local_reasoning.py --clips-root /home/etri/DATASET/nureasoning/clips --apply
  # 소량만 먼저 테스트: --limit 5 --apply
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reasoning_keepset import HISTORY_DEFAULT, prune_clip_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", required=True, help="clip 디렉터리들이 들어있는 루트")
    ap.add_argument("--history", type=int, default=HISTORY_DEFAULT,
                    help=f"temporal history 오프셋(기본 {HISTORY_DEFAULT}=1초 전). reasoning∪(reasoning−오프셋) 보존.")
    ap.add_argument("--apply", action="store_true", help="실제 삭제 실행(미지정=dry-run, 아무것도 안 지움)")
    ap.add_argument("--limit", type=int, default=0, help="앞 N개 clip만 처리(0=전체). 소량 테스트용.")
    args = ap.parse_args()

    clips = sorted(d for d in os.listdir(args.clips_root)
                   if os.path.isdir(os.path.join(args.clips_root, d)))
    if args.limit:
        clips = clips[:args.limit]
    mode = "APPLY(실제 삭제)" if args.apply else "DRY-RUN(미삭제)"
    print(f"[{mode}] clips-root={args.clips_root} | clip {len(clips)}개 | history={args.history}")
    if not args.apply:
        print("  ⚠️ dry-run: 실제 삭제 안 함. 확인 후 --apply 로 재실행하세요.\n")

    tot_all = tot_keep = tot_del = 0
    tot_del_bytes = 0
    skipped = []
    t0 = time.time()
    for i, name in enumerate(clips, 1):
        r = prune_clip_dir(os.path.join(args.clips_root, name), args.history, args.apply)
        st = r["status"]
        if st != "ok":
            skipped.append((name, st))
            continue
        tot_all += r["n_all"]; tot_keep += r["n_keep"]; tot_del += r["n_del"]
        tot_del_bytes += r["del_bytes"]
        if i <= 3 or i % 100 == 0:                        # 처음 몇 개 + 100개마다 진행 출력
            print(f"  [{i}/{len(clips)}] {name[:36]} | jpg {r['n_all']}→keep {r['n_keep']}/del {r['n_del']} "
                  f"| {r['del_bytes']/1e6:.0f}MB | frames={r['keep_frames']}")

    dt = time.time() - t0
    print(f"\n=== 요약 [{mode}] ({dt:.0f}s) ===")
    print(f"  처리 clip: {len(clips) - len(skipped)} (건너뜀 {len(skipped)})")
    print(f"  카메라 jpg: 전체 {tot_all} → 유지 {tot_keep} / 삭제 {tot_del}")
    print(f"  {'삭제됨' if args.apply else '삭제 예정'} 용량: {tot_del_bytes/1e9:.1f} GB "
          f"(유지 후 ≈ {(tot_all and tot_keep/tot_all or 0)*100:.0f}% 프레임)")
    if skipped:
        from collections import Counter
        print(f"  건너뛴 clip 사유: {dict(Counter(s for _n, s in skipped))}")
        # keep_empty_abort는 반드시 살펴봐야 함(파싱 이상)
        aborts = [n for n, s in skipped if s == "keep_empty_abort"]
        if aborts:
            print(f"  ⚠️ keep_empty_abort(수동 확인 필요): {aborts[:10]}{' …' if len(aborts) > 10 else ''}")
    if not args.apply:
        print("\n  → 위 내용 확인 후 실제 삭제하려면 동일 명령에 --apply 추가.")


if __name__ == "__main__":
    main()
