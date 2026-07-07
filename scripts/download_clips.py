"""Download nuReasoning clips.

Per DECISIONS.md (disk is tight): for each clip we download the ~2 GB zip, extract
only the low-level assets (metadata, map, ego_state, annotations, reasoning ≈ 17 MB)
plus the camera images **at the driving keyframes** for the requested views, then
delete the zip. Net footprint ≈ 20 MB/clip (front only) or ≈ 35 MB/clip (8 views).

Views (--views):
  front  → front camera only   (≈ 3 imgs/clip)  — legacy single-view pipeline.
  all    → all 8 surround views (≈ 24 imgs/clip, ≈ +15 MB/clip) — multi-view pipeline.

Frame policy (--policy):
  driving → action-labelled keyframes only (0.2Hz, ≈ 3 frames/clip).
  spatial → all 1Hz annotated frames (≈ 13 frames/clip; superset of driving). Reproduces
            the paper's 1Hz frame usage. With --views all → ≈ 13×8 ≈ 104 imgs/clip
            (≈ 130 MB/clip), so the 160-clip mini set lands at ≈ 25 GB.
  all     → EVERY 10Hz frame (≈ 200 frames/clip). With --views all → ≈ 200×8 ≈ 1600 imgs/clip
            (≈ 1.8 GB/clip = 원본 zip 전체). 논문 원본 그대로의 10Hz 멀티뷰를 디스크에 보존한다.
            ⚠️ 대용량: part_1(1000클립) 전체면 ≈ 1.8 TiB. zip은 클립마다 받아서 풀고 곧바로
            지우므로(스트리밍) 피크 디스크는 '해제분 + zip 1개' 수준이다.

Output (--out_path):
  미지정 → data/raw (레포 내부; mini 파이프라인 기본).
  지정   → <out_path>/clips/<clip_dirname>/ 에 추출. 큰 디스크에 10Hz 전체를 받을 때 사용.

Count (--n vs --limit):
  --n     → ensure N clips extracted, counting already-cached toward N(mini 증분 풀; 기본 30).
  --limit → 이번 실행에서 처리할 클립 수. 0=선택 part의 zip 전부, K>0=앞에서 K개(테스트/부분).
            지정하면(>=0) --n 대신 이 값이 개수를 지배한다.

Modes:
  default          → pull clips from --part (first-time pull; --n 또는 --limit로 개수 제어).
  --refetch-local  → for every clip already on disk (clips/ + _probe_extract/),
                     (re)fetch the requested views' keyframe images and consolidate
                     them under clips/. Used to upgrade an existing front-only mini
                     dataset to 8 views WITHOUT changing the clip set. Idempotent:
                     a clip whose requested-view frames are all present is skipped
                     (no download), so the run is resumable.

Extracted clips land in <out_path or data/raw>/clips/<clip_dirname>/ — the layout the loader expects.

Run:  python scripts/download_clips.py --refetch-local --views all
      python scripts/download_clips.py --n 30 --part part_1 [--views all]
      # part_1 전체를 10Hz 멀티뷰로 큰 디스크에(스트리밍, zip 즉시 삭제):
      python scripts/download_clips.py --part part_1 --views all --policy all \
             --out_path /home/etri/DATASET/nureasoning --limit 0
      # 먼저 5개만 테스트:
      python scripts/download_clips.py --part part_1 --views all --policy all \
             --out_path /home/etri/DATASET/nureasoning --limit 5
             
다운로드가 끝나면(part_1 1000클립):
# 1. val 100클립 고정 manifest 생성 (클립 단위, seed 1234 재현)
    python -m evaluation.eval_set /home/etri/DATASET/nureasoning/clips --n-clips-val 100 --seed 1234

# 2. SFT 빌드 (manifest 기준 train 900 / val 100 분리)
    python -m sft_data.build_sft --clips-root /home/etri/DATASET/nureasoning/clips
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

from nureasoning import CAMERA_VIEWS, parse_clip, select_keyframes  # noqa: E402

REPO_ID = "qixuewei/nuReasoning"
RAW = REPO / "data" / "raw"
CLIPS_DIR = RAW / "clips"
PROBE_DIR = RAW / "_probe_extract"
DL_DIR = RAW / "_dl"

ALWAYS = ("metadata.json", "map.pkl")
LOWLEVEL_PREFIXES = ("ego_state/", "annotations/", "reasoning/")


def _fmt_dur(sec: float) -> str:
    """초 → 'Hh MMm SSs'(진행 로그의 elapsed/eta 표기)."""
    sec = int(max(0, sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def log_path() -> Path:
    """진행 로그 파일 경로 = 데이터셋 저장 루트(CLIPS_DIR.parent)/download.log.
    --out_path로 CLIPS_DIR이 바뀌면 로그도 그 경로에 남는다(부모 폴더 보장)."""
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    return CLIPS_DIR.parent / "download.log"


def _log(path: Path, msg: str) -> None:
    """터미널 + 로그파일에 타임스탬프 한 줄을 기록하고 즉시 flush(실시간 진행 추적)."""
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True)
    with open(path, "a") as fh:
        fh.write(line + "\n")
        fh.flush()


def _progress(path: Path, done: int, target: int, t0: float, tail: str) -> None:
    """완료 개수/퍼센트/elapsed/ETA를 한 줄로 기록. done>=1일 때 남은 시간을 선형 추정."""
    pct = 100.0 * done / target if target else 0.0
    elapsed = time.time() - t0
    eta = (elapsed / done) * (target - done) if done else 0.0
    _log(path, f"[{done}/{target}] {pct:5.1f}% | {tail}  | elapsed {_fmt_dur(elapsed)} eta {_fmt_dur(eta)}")


def _inner(name: str, clip_dirname: str) -> str:
    """Path inside the clip dir (strip the leading ``<clip_dirname>/``)."""
    prefix = clip_dirname + "/"
    return name[len(prefix):] if name.startswith(prefix) else name


def _resolve_views(views: str) -> tuple[str, ...]:
    """``"front"`` -> just front; ``"all"`` -> the 8 logical surround views."""
    return ("front",) if views == "front" else CAMERA_VIEWS


def existing_clip_dir(clip_dirname: str) -> Path | None:
    """Where this clip's metadata currently lives on disk (clips/ or _probe_extract/), if anywhere."""
    for base in (CLIPS_DIR, PROBE_DIR):
        if (base / clip_dirname / "metadata.json").exists():
            return base / clip_dirname
    return None


def keyframe_view_relpaths(clip_root: Path, views: tuple[str, ...], policy: str) -> set[str]:
    """Relative paths (under the clip dir) of the requested views' images at the selected frames."""
    clip = parse_clip(clip_root)
    rels: set[str] = set()
    for f in select_keyframes(clip, policy=policy):
        for v in views:
            rp = f.camera_relpaths.get(v)
            if rp:
                rels.add(rp)
    return rels


def have_all_views(clip_dirname: str, views: tuple[str, ...], policy: str) -> bool:
    """True iff clips/<clip> already holds every requested view's image for the selected frames."""
    clip_root = CLIPS_DIR / clip_dirname
    if not (clip_root / "metadata.json").exists():
        return False
    rels = keyframe_view_relpaths(clip_root, views, policy)
    return bool(rels) and all((clip_root / rp).exists() for rp in rels)


def process_zip(zip_repo_path: str, views: tuple[str, ...], policy: str) -> tuple[str, str]:
    """Download one clip zip, extract low-level + the requested views' selected-frame images
    into clips/, then delete the zip. Consolidates probe clips into clips/."""
    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      allow_patterns=[zip_repo_path], local_dir=str(DL_DIR))
    zip_path = DL_DIR / zip_repo_path
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            clip_dirname = names[0].split("/")[0]

            # 1) low-level assets + metadata/map (idempotent overwrite; needed to read keyframes)
            base = [n for n in names if not n.endswith("/") and (
                _inner(n, clip_dirname) in ALWAYS
                or _inner(n, clip_dirname).startswith(LOWLEVEL_PREFIXES))]
            zf.extractall(CLIPS_DIR, members=base)

            # 2) camera images at the selected frames (policy), for the requested views
            clip_root = CLIPS_DIR / clip_dirname
            rels = {f"{clip_dirname}/{rp}" for rp in keyframe_view_relpaths(clip_root, views, policy)}
            imgs = [n for n in names if n in rels]
            zf.extractall(CLIPS_DIR, members=imgs)
            return clip_dirname, f"ok ({len(imgs)} imgs / {len(views)} views / {policy})"
    finally:
        zip_path.unlink(missing_ok=True)
        # drop the now-empty download tree for this clip to reclaim space
        part_dir = (DL_DIR / zip_repo_path).parent
        if part_dir.exists() and not any(part_dir.iterdir()):
            shutil.rmtree(part_dir, ignore_errors=True)


def all_repo_zips(api: HfApi) -> dict[str, str]:
    """Map clip dirname (zip stem) -> repo zip path, across every train part."""
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    return {Path(f).stem: f for f in files
            if f.startswith("data/train/") and f.endswith(".zip")}


def local_clip_dirnames() -> list[str]:
    """Clip dirnames currently on disk (clips/ ∪ _probe_extract/), as the refetch target set."""
    names: set[str] = set()
    for base in (CLIPS_DIR, PROBE_DIR):
        if base.exists():
            names |= {p.parent.name for p in base.glob("*/metadata.json")}
    return sorted(names)


def refetch_local(views: tuple[str, ...], policy: str) -> None:
    """(Re)fetch the requested views for every clip already on disk; consolidate into clips/."""
    api = HfApi()
    zip_by_stem = all_repo_zips(api)
    targets = local_clip_dirnames()
    lp = log_path()
    t0 = time.time()
    total = len(targets)
    _log(lp, f"START refetch-local: {total} local clips; views={len(views)}; policy={policy}; out={CLIPS_DIR}")

    n_ok = n_cached = n_skip = 0
    for i, dirname in enumerate(targets, 1):
        if have_all_views(dirname, views, policy):
            n_cached += 1
            _progress(lp, i, total, t0, f"{dirname}  cached (all frames present)")
            continue
        zip_path = zip_by_stem.get(dirname)
        if not zip_path:
            n_skip += 1
            _progress(lp, i, total, t0, f"{dirname}  SKIP: no matching zip in repo")
            continue
        try:
            _, status = process_zip(zip_path, views, policy)
        except Exception as e:  # noqa: BLE001 — keep going on a bad clip, report it
            n_skip += 1
            _progress(lp, i, total, t0, f"{dirname}  SKIP: {type(e).__name__}: {e}")
            continue
        n_ok += 1
        _progress(lp, i, total, t0, f"{dirname}  {status}")

    _log(lp, f"DONE refetch-local: {total}/{total} (100.0%) "
             f"| ok {n_ok} cached {n_cached} skip {n_skip} | total {_fmt_dur(time.time()-t0)}")


def pull_new(n: int, part: str, views: tuple[str, ...], policy: str, limit: int = -1) -> None:
    """First-time pull from ``part`` with the requested views.

    limit < 0  → 레거시: ``n`` 클립을 확보(이미 있는 것도 n에 셈).
    limit == 0 → 선택한 part의 zip 전부 처리.
    limit > 0  → 앞에서부터 ``limit`` 개만 처리(테스트/부분 다운로드).
    """
    api = HfApi()
    files = api.list_repo_files(REPO_ID, repo_type="dataset")
    zips = sorted(f for f in files
                  if f.startswith(f"data/train/{part}/") and f.endswith(".zip"))
    if limit >= 0:                                   # --limit 지정: 후보 리스트를 잘라 개수를 지배
        zips = zips if limit == 0 else zips[:limit]
        target = len(zips)
    else:                                            # 레거시 --n (cached 포함 ensure)
        target = n

    lp = log_path()
    t0 = time.time()
    _log(lp, f"START pull {part}: {len(zips)} candidate zips; target {target}; "
             f"views={len(views)}; policy={policy}; out={CLIPS_DIR}")

    done = n_ok = n_cached = n_skip = 0
    t0_dl = None                                     # 실제 다운로드 시작 시각(캐시 스킵 후) → ETA는 다운로드 속도로 산출
    for z in zips:
        if done >= target:
            break
        dirname = Path(z).stem
        if have_all_views(dirname, views, policy):   # 이미 추출된 클립 → 로그 없이 카운트만(이어받기 스킵)
            done += 1; n_cached += 1
            if n_cached % 100 == 0:                   # 재스캔 진행은 터미널에만 표시(로그파일은 실제 진행만)
                print(f"  ...scanned {done}/{target} extracted (skip)", flush=True)
            continue
        if t0_dl is None:                             # 첫 실제 다운로드 직전 = 이어받기 재개 지점
            t0_dl = time.time()
            if n_cached:                              # 스킵 요약 1줄만 로그에 남기고 여기서부터 진행 기록
                _log(lp, f"resume: skipped {n_cached} already-extracted clips → 이어받기 [{done + 1}/{target}]")
        try:
            name, status = process_zip(z, views, policy)
        except Exception as e:  # noqa: BLE001
            n_skip += 1
            _log(lp, f"SKIP {dirname}: {type(e).__name__}: {e}")
            continue
        done += 1; n_ok += 1
        # ETA는 '실제 다운로드' 속도(n_ok) 기준 — 캐시 스킵분은 순식간이라 포함하면 ETA가 왜곡됨.
        el = time.time() - t0_dl
        eta = (el / n_ok) * (target - done)
        _log(lp, f"[{done}/{target}] {100.0 * done / target:5.1f}% | {name}  {status}  "
                 f"| dl {_fmt_dur(el)} eta {_fmt_dur(eta)}")

    _log(lp, f"DONE pull {part}: {done}/{target} ({100.0*done/target if target else 0:.1f}%) "
             f"| ok {n_ok} cached {n_cached} skip {n_skip} | total {_fmt_dur(time.time()-t0)}")


def main() -> None:
    global CLIPS_DIR, DL_DIR, PROBE_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30,
                    help="ensure N clips extracted, counting already-cached toward N (mini 증분 풀)")
    ap.add_argument("--part", default="part_1", choices=["part_1", "part_2", "part_3"])
    ap.add_argument("--views", default="all", choices=["front", "all"],
                    help="which camera views to keep at the selected frames (default: all 8)")
    ap.add_argument("--policy", default="driving", choices=["driving", "spatial", "all"],
                    help="추출할 프레임: driving(0.2Hz) / spatial(1Hz) / all(=10Hz 전체 프레임). "
                         "all + --views all = 클립당 ~1600장(≈1.8GB, 원본 zip 전체) — 대용량 주의.")
    ap.add_argument("--out_path", default=None,
                    help="추출 루트(미지정=data/raw). 지정 시 <out_path>/clips/에 저장 — "
                         "다른/큰 디스크에 10Hz 전체를 받을 때 사용.")
    ap.add_argument("--limit", type=int, default=-1,
                    help="이번 실행에서 처리할 클립 수. 0=선택 part 전부, K>0=앞에서 K개. "
                         "지정(>=0)하면 --n 대신 개수를 지배(테스트/부분 다운로드).")
    ap.add_argument("--refetch-local", action="store_true",
                    help="(re)fetch requested views for every clip already on disk")
    args = ap.parse_args()

    if args.out_path:                                 # 다른 경로/디스크로 추출(대용량 10Hz)
        root = Path(args.out_path)
        CLIPS_DIR = root / "clips"
        DL_DIR = root / "_dl"
        PROBE_DIR = root / "_probe_extract"

    views = _resolve_views(args.views)
    if args.refetch_local:
        refetch_local(views, args.policy)
    else:
        pull_new(args.n, args.part, views, args.policy, args.limit)

    n_clips = len(list(CLIPS_DIR.glob("*/metadata.json"))) if CLIPS_DIR.exists() else 0
    print(f"\nextracted clips on disk ({CLIPS_DIR}): {n_clips}")
    DL_DIR.exists() and shutil.rmtree(DL_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
