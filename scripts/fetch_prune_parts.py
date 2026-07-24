#!/usr/bin/env python3
"""HF의 part_2/part_3(미보유) clip을 받아 **reasoning 프레임만** 남기고 저장한다.

clip당: ZIP 다운로드(2~3GB) → **keep-set 이미지 + 주석만 선택 추출**(비-keep jpg는 디스크에 안 씀) →
ZIP 삭제. 즉 통째 압축해제(2GB) 없이 clip당 ~145MB만 디스크에 남긴다. 이미 있는 clip은 건너뜀(재개 가능).

⚠️ HF ZIP은 clip 통째로만 제공되므로 **다운로드 대역폭은 clip당 2~3GB 그대로**다(디스크만 절약).
필요: `pip install huggingface_hub`. 게이트면 `HF_TOKEN` 환경변수(또는 `huggingface-cli login`).

사용:
  # 1) dry-run: 받을 clip 목록·개수만 확인(다운로드 안 함)
  python scripts/fetch_prune_parts.py --clips-root /home/etri/DATASET/nureasoning/clips --parts part_2,part_3
  # 2) 실제 다운로드+추출
  python scripts/fetch_prune_parts.py --clips-root /home/etri/DATASET/nureasoning/clips --parts part_2,part_3 --apply
  # 소량 테스트: --limit 2 --apply
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reasoning_keepset import HISTORY_DEFAULT, keep_basenames  # noqa: E402

REPO_ID = "qixuewei/nuReasoning"


# ─── 실시간 진행 로그(download_clips.py의 _log/_progress 재활용) ────────────────────────────────
def _fmt_dur(sec: float) -> str:
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _log(path, msg: str) -> None:
    """터미널 + 로그파일에 타임스탬프 한 줄 기록 후 즉시 flush(실시간 추적)."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True)
    if path:
        with open(path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()


def _progress(path, part: str, p_done: int, p_total: int, g_done: int, g_total: int,
              t0: float, tail: str) -> None:
    """part별 진행률 + 전체 진행률 + elapsed/ETA를 한 줄로 기록(전체 기준 ETA)."""
    p_pct = 100.0 * p_done / p_total if p_total else 0.0
    g_pct = 100.0 * g_done / g_total if g_total else 0.0
    elapsed = time.time() - t0
    eta = (elapsed / g_done) * (g_total - g_done) if g_done else 0.0
    _log(path, f"{part} [{p_done}/{p_total}] {p_pct:5.1f}% | 전체 [{g_done}/{g_total}] {g_pct:5.1f}% "
               f"| {tail} | elapsed {_fmt_dur(elapsed)} eta {_fmt_dur(eta)}")


def selective_extract(zip_path: str, dest_clip_dir: str, history: int) -> dict:
    """ZIP에서 keep-set 카메라 이미지 + 모든 비-카메라 파일만 dest_clip_dir로 추출.
    반환: {n_all_jpg, n_keep_jpg, status}. metadata 파싱 실패/ reasoning 없음이면 추출 안 함."""
    import json
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        meta_members = [n for n in names if n.endswith("metadata.json")]
        if not meta_members:
            return {"status": "no_metadata"}
        meta_member = meta_members[0]
        prefix = os.path.dirname(meta_member)             # zip 내부 clip 접두("clipname" 또는 "")
        meta = json.loads(zf.read(meta_member).decode("utf-8"))
        keep_bn, _keep_fi, reas = keep_basenames(meta, history)
        if not reas:
            return {"status": "no_reasoning"}
        n_all_jpg = n_keep_jpg = 0
        for m in names:
            if m.endswith("/"):
                continue
            rel = os.path.relpath(m, prefix) if prefix else m
            is_cam_jpg = ("/cameras/" in ("/" + rel.replace("\\", "/"))) and rel.endswith(".jpg")
            if is_cam_jpg:
                n_all_jpg += 1
                if os.path.basename(m) not in keep_bn:    # 비-keep 카메라 이미지 → 추출 안 함
                    continue
                n_keep_jpg += 1
            out_path = os.path.join(dest_clip_dir, rel)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with zf.open(m) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        return {"status": "ok", "n_all_jpg": n_all_jpg, "n_keep_jpg": n_keep_jpg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips-root", required=True, help="추출 결과를 둘 루트(기존 clip과 동일 위치)")
    ap.add_argument("--parts", default="part_2,part_3", help="처리할 part 디렉터리(쉼표구분)")
    ap.add_argument("--history", type=int, default=HISTORY_DEFAULT)
    ap.add_argument("--apply", action="store_true", help="실제 다운로드+추출(미지정=dry-run: 목록만)")
    ap.add_argument("--limit", type=int, default=0, help="part별 앞 N clip만(0=전체). 테스트용.")
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="이 여유공간 미만이면 중단(ZIP 임시분+안전마진). 기본 30GB.")
    ap.add_argument("--tmp", default=None, help="ZIP 임시 저장 위치(기본 clips-root/_zip_tmp)")
    ap.add_argument("--retries", type=int, default=0,
                    help="clip당 다운로드 재시도 횟수. **0=무한 재시도(기본)** — 네트워크가 끊겨도 성공할 때까지 "
                         "계속 시도하므로 clip을 빠뜨리지 않는다. N>0이면 N회 실패 시 그 clip을 건너뛰고 진행.")
    ap.add_argument("--max-backoff", type=int, default=300,
                    help="재시도 대기 상한(초, 기본 300=5분). 지수 백오프(5→10→20…)가 이 값을 넘지 않는다.")
    ap.add_argument("--timeout", type=int, default=60,
                    help="HF 다운로드 네트워크 타임아웃(초, 기본 60). 서버가 연결을 끊고 응답이 없을 때 "
                         "무한 대기(hang)하는 것을 막는다.")
    args = ap.parse_args()

    # ⚠️ huggingface_hub는 **import 시점에** 이 환경변수를 읽어 타임아웃을 정한다 → import보다 먼저 설정.
    #    (실제 사고: 서버가 소켓을 CLOSE-WAIT로 끊었는데 타임아웃이 없어 14시간 futex 대기로 멈춤)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(args.timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(args.timeout))

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub 필요: pip install huggingface_hub")

    token = os.environ.get("HF_TOKEN")                    # 없으면 캐시 로그인 사용(None)
    api = HfApi()
    all_files = api.list_repo_files(REPO_ID, repo_type="dataset", token=token)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    tmp_dir = args.tmp or os.path.join(args.clips_root, "_zip_tmp")
    mode = "APPLY(다운로드+추출)" if args.apply else "DRY-RUN(목록만)"

    # part별 ZIP 목록을 미리 확정 → 전체 target 개수(정확한 %/ETA 기준). --limit는 part별 앞 N개.
    part_zips = {}
    for part in parts:
        zips = sorted(f for f in all_files
                      if f.startswith(f"data/train/{part}/") and f.endswith(".zip"))
        part_zips[part] = zips[:args.limit] if args.limit else zips
    g_total = sum(len(z) for z in part_zips.values())

    # 진행 로그 파일: 데이터셋 저장 루트(clips-root의 부모)/fetch_prune.log — download.log 관례 재활용.
    os.makedirs(args.clips_root, exist_ok=True)
    log_file = os.path.join(os.path.dirname(os.path.normpath(args.clips_root)), "fetch_prune.log")
    _log(log_file, f"START [{mode}] repo={REPO_ID} parts={parts} "
                   f"target={g_total} clips | parts={{{', '.join(f'{p}:{len(z)}' for p, z in part_zips.items())}}}")

    grand_done = grand_skip = grand_fail = 0
    g_seen = 0                                            # 전체에서 지금까지 처리(완료+스킵)한 clip 수
    t0 = time.time()
    for part in parts:
        zips = part_zips[part]
        _log(log_file, f"--- {part}: ZIP {len(zips)}개 시작 ---")
        for i, zf_path in enumerate(zips, 1):
            name = os.path.basename(zf_path)[:-4]         # clip 이름(.zip 제거)
            dest = os.path.join(args.clips_root, name)
            if os.path.isfile(os.path.join(dest, "metadata.json")):
                grand_skip += 1; g_seen += 1
                if i % 100 == 0:                          # 스킵은 100개마다만 진행 표시(로그 과다 방지)
                    _progress(log_file, part, i, len(zips), g_seen, g_total, t0, f"{name[:32]} 스킵(이미 있음)")
                continue                                  # 이미 추출됨 → 재개 스킵
            if not args.apply:
                if i <= 3:
                    _log(log_file, f"  [예정] {part}/{name[:40]}")
                g_seen += 1
                continue
            free_gb = shutil.disk_usage(args.clips_root).free / 1e9
            if free_gb < args.min_free_gb:
                _log(log_file, f"⚠️ 여유공간 {free_gb:.0f}GB < {args.min_free_gb}GB → 중단. 지금까지 {grand_done}개 완료.")
                _summary(log_file, grand_done, grand_skip, grand_fail, t0); return
            os.makedirs(tmp_dir, exist_ok=True)
            # 다운로드 재시도: 네트워크 끊김/타임아웃에도 clip을 빠뜨리지 않는다.
            #   --retries 0(기본) = **무한 재시도**(지수 백오프, 상한 --max-backoff) → 회선이 복구되면
            #   알아서 이어받는다. N>0이면 N회 실패 시 FAIL 기록 후 다음 clip으로 진행.
            local_zip = None
            last_err = None
            attempt = 0
            while True:
                attempt += 1
                try:
                    local_zip = hf_hub_download(REPO_ID, filename=zf_path, repo_type="dataset",
                                                local_dir=tmp_dir, token=token)
                    if attempt > 1:
                        _log(log_file, f"  ✅ {name[:32]} {attempt}번째 시도에서 성공")
                    break
                except KeyboardInterrupt:
                    raise                                 # Ctrl-C는 재시도 대상 아님(즉시 중단)
                except Exception as e:                    # noqa: BLE001 (네트워크/HTTP 등 모든 실패 재시도)
                    last_err = e
                    local_zip = None
                    if args.retries and attempt >= args.retries:
                        break                             # 유한 재시도 모드에서 소진 → 건너뛰기
                    wait = min(5 * (2 ** (attempt - 1)), args.max_backoff)   # 5,10,20,…(상한)
                    _log(log_file, f"  ⚠️ {name[:32]} 다운로드 실패(시도 {attempt}"
                                   f"{'/' + str(args.retries) if args.retries else ', 무한재시도'}) "
                                   f"{type(e).__name__}: {str(e)[:80]} → {wait}s 후 재시도")
                    time.sleep(wait)
            g_seen += 1
            if local_zip is None:                         # (유한 모드) 재시도 소진 → 이 clip 건너뛰고 계속
                grand_fail += 1
                _progress(log_file, part, i, len(zips), g_seen, g_total, t0,
                          f"{name[:32]} FAIL:{type(last_err).__name__}")
                continue
            try:
                tmp_clip = dest + ".partial"
                shutil.rmtree(tmp_clip, ignore_errors=True)
                r = selective_extract(local_zip, tmp_clip, args.history)
                if r["status"] != "ok":
                    shutil.rmtree(tmp_clip, ignore_errors=True)
                    _progress(log_file, part, i, len(zips), g_seen, g_total, t0,
                              f"{name[:32]} SKIP:{r['status']}")
                else:
                    os.replace(tmp_clip, dest)            # 원자적 완료(부분추출이 최종처럼 안 보이게)
                    grand_done += 1
                    _progress(log_file, part, i, len(zips), g_seen, g_total, t0,
                              f"{name[:32]} ok jpg{r['n_all_jpg']}→keep{r['n_keep_jpg']}")
            finally:
                if local_zip and os.path.isfile(local_zip):
                    os.remove(local_zip)                  # ZIP 즉시 삭제(디스크 회수)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    _summary(log_file, grand_done, grand_skip, grand_fail, t0)


def _summary(log_file, done, skip, fail, t0):
    msg = f"DONE 추출 완료 clip: {done} | 이미 있어 스킵: {skip} | 실패(건너뜀): {fail} | 총 {_fmt_dur(time.time() - t0)}"
    if fail:
        msg += "  ⚠️ 실패분은 스크립트를 다시 실행하면 재시도됩니다(완료분은 자동 스킵)."
    _log(log_file, msg)


if __name__ == "__main__":
    main()
