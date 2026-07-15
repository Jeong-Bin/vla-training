#!/usr/bin/env python3
"""reasoning-only 서브셋 공용 로직: clip에서 **유지할 카메라 이미지**를 계산하고 나머지를 삭제한다.

keep-set(프레임) = reasoning 주석 있는 프레임 ∪ (그 프레임 − HISTORY).
  - reasoning 프레임: metadata.frames[*].reasoning 이 빈 문자열이 아닌 프레임(=spatial 1Hz, 30~150, 13개).
  - HISTORY(기본 10): --temporal 과거뷰(1초 전=프레임 t−10) 보존용. 존재하는 frame_index만 추가.
  → clip당 유지 프레임 14개(20,30,…,150) × 8뷰 = 112장. 나머지 카메라 jpg(≈1490장)만 삭제.

⚠️ **cameras/*.jpg만** 삭제 대상이다. metadata.json / reasoning / annotations / ego_state / map.pkl은
   절대 건드리지 않는다(용량 미미·학습에 필요). reasoning 프레임이 하나도 없거나 keep 계산이 비면
   그 clip은 **삭제를 건너뛴다**(파싱 이상으로 전체를 날리는 사고 방지).
"""
import glob
import json
import os

HISTORY_DEFAULT = 10       # temporal history offset (1초 전 = 프레임 t-10, 미니 서브셋 1Hz 이미지 기준 10배수)


def keep_basenames(meta: dict, history: int = HISTORY_DEFAULT):
    """metadata dict → (유지할 카메라 파일 basename 집합, keep 프레임 인덱스 집합, reasoning 프레임 인덱스 집합)."""
    frames = meta["frames"]
    fi_all = {f["frame_index"] for f in frames}
    reas = {f["frame_index"] for f in frames if f.get("reasoning")}
    keep_fi = reas | {fi - history for fi in reas if (fi - history) in fi_all}
    bns = set()
    for f in frames:
        if f["frame_index"] in keep_fi:
            for _view, path in f["sensors"]["cameras"].items():
                bns.add(os.path.basename(path))          # 파일명은 뷰+타임스탬프라 clip 내 유일
    return bns, keep_fi, reas


def prune_clip_dir(clip_dir: str, history: int = HISTORY_DEFAULT, apply: bool = False):
    """clip 디렉터리에서 keep-set 외 cameras/*.jpg를 삭제(apply=True)하거나 계산만(dry-run).

    반환 dict: {status, n_all, n_keep, n_del, del_bytes, keep_frames, reas_frames}.
      status: ok | no_metadata | no_reasoning | keep_empty_abort
    """
    meta_path = os.path.join(clip_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return {"status": "no_metadata"}
    with open(meta_path) as fh:
        meta = json.load(fh)
    keep_bn, keep_fi, reas = keep_basenames(meta, history)
    if not reas:                                          # reasoning 프레임 0 → 이상 clip, 손대지 않음
        return {"status": "no_reasoning", "n_all": 0, "n_keep": 0, "n_del": 0, "del_bytes": 0}
    cam_dir = os.path.join(clip_dir, "cameras")
    all_jpg = glob.glob(os.path.join(cam_dir, "**", "*.jpg"), recursive=True)
    keep_jpg = [j for j in all_jpg if os.path.basename(j) in keep_bn]
    del_jpg = [j for j in all_jpg if os.path.basename(j) not in keep_bn]
    # 안전장치: 유지 대상이 0인데 이미지가 존재하면(basename 매칭 실패=파싱 이상) 이 clip 삭제 중단.
    if all_jpg and not keep_jpg:
        return {"status": "keep_empty_abort", "n_all": len(all_jpg),
                "n_keep": 0, "n_del": 0, "del_bytes": 0}
    del_bytes = sum(os.path.getsize(j) for j in del_jpg)
    if apply:
        for j in del_jpg:
            os.remove(j)
    return {"status": "ok", "n_all": len(all_jpg), "n_keep": len(keep_jpg),
            "n_del": len(del_jpg), "del_bytes": del_bytes,
            "keep_frames": sorted(keep_fi), "reas_frames": sorted(reas)}
