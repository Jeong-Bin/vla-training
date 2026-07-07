"""SFT content(JSONL) → 모델 학습 포맷(Qwen-VL chat) 어댑터.

스펙 §7.4: 데이터 *내용*과 모델 *형식*을 분리한다. `build_sft.py`가 만든 내용 레코드를 받아
학습용 대화(messages)로 렌더한다.

설계상의 핵심 정합성:
  - user 프롬프트 = 평가와 **동일한 템플릿**(`prompts/driving_multiview_v1.txt`).
    학습 때 본 프롬프트와 평가 프롬프트가 같아야 파인튜닝 개선이 동일 하니스에서 측정된다.
  - assistant 타깃 = canonical JSON `{"longitudinal","lateral","reasoning"}`.
    longitudinal/lateral은 택소노미로 표준화(프롬프트가 요구하는 라벨공간과 일치),
    reasoning은 GT 원문을 그대로 둔다(장면 근거가 핵심 학습 신호 — zero-shot의 약점).

Run(미리보기):  python -m sft_data.chat_format data/sft/val.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from nureasoning import VIEW_LABELS, resolve_path  # noqa: E402  (뷰별 캡션 라벨 + 이미지 경로 복원)
from evaluation import taxonomy as T  # noqa: E402

PROMPT_FILE = REPO / "prompts" / "driving_multiview_v1.txt"        # driving(행동 결정) 태스크
_PROMPT_TPL = PROMPT_FILE.read_text()
SPATIAL_PROMPT_FILE = REPO / "prompts" / "spatial_describe_v1.txt"  # spatial(객체 인식) 태스크
_SPATIAL_PROMPT_TPL = SPATIAL_PROMPT_FILE.read_text()


# render_target: SFT output(raw GT)을 학습 타깃 문자열(canonical JSON 한 줄)로 변환.
def render_target(output: dict) -> str:
    return json.dumps({
        "longitudinal": T.map_longitudinal(output.get("longitudinal", "")),
        "lateral": T.map_lateral(output.get("lateral", "")),
        "reasoning": output.get("reasoning", ""),
    }, ensure_ascii=False)


# render_spatial_target: spatial 태스크 타깃 = 가까운 순 객체 리스트 JSON 한 줄.
# output["objects"]는 build_sft가 ego-frame 위치로 이미 계산해 둔 [{category,dist_m,fwd_m,left_m}].
def render_spatial_target(output: dict) -> str:
    return json.dumps({"objects": output.get("objects", [])}, ensure_ascii=False)


# view_caption: 이미지 앞에 붙는 뷰 라벨 텍스트(학습=평가 공용 — 양쪽이 같은 문구를 써야 정합).
def view_caption(view: str) -> str:
    return f"{VIEW_LABELS.get(view, view)} camera view:"


# to_qwen_user_content: 멀티뷰 user 턴 content 생성(학습=평가 공용). 각 뷰 이미지 앞에
# view_caption을 붙여 인터리브한 뒤 마지막에 task 프롬프트를 둔다 → 모델이 어느 이미지가
# 어느 카메라인지 식별 가능. images = [{"view","path"}](신규) 또는 [경로str](레거시 single-front).
def to_qwen_user_content(images: list, prompt: str, image_root: Path = REPO) -> list:
    content: list = []
    for im in images:
        if isinstance(im, dict):
            view, rel = im["view"], im["path"]
        else:                                    # 레거시: front 경로 문자열만
            view, rel = "front", im
        abs_path = str(resolve_path(rel, image_root).resolve())   # 절대경로면 그대로, 상대면 image_root 기준
        content.append({"type": "text", "text": view_caption(view)})
        content.append({"type": "image", "image": abs_path})
    content.append({"type": "text", "text": prompt})
    return content


# to_qwen_chat: 레코드 1개 → Qwen-VL 대화 dict. image_root는 images의 상대경로 기준(기본 레포 루트).
# 이미지는 절대경로 문자열로 넣어 학습 시 그대로 로드 가능하게 한다.
# task("driving"|"spatial")에 따라 프롬프트·타깃 렌더가 분기한다(멀티태스크 SFT, 논문 A 방식).
def to_qwen_chat(record: dict, image_root: Path = REPO) -> dict:
    task = record.get("task", "driving")
    if task == "spatial":
        prompt = _SPATIAL_PROMPT_TPL
        target = render_spatial_target(record["output"])
    else:
        mission = record.get("mission") or "drive safely"
        prompt = _PROMPT_TPL.replace("{mission}", mission)  # .replace: 템플릿에 JSON 중괄호가 있어 .format 불가
        target = render_target(record["output"])
    return {
        "id": record.get("id"),
        "messages": [
            {"role": "user", "content": to_qwen_user_content(record["images"], prompt, image_root)},
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="SFT content jsonl (e.g. data/sft/val.jsonl)")
    ap.add_argument("--n", type=int, default=1, help="how many rendered examples to print")
    args = ap.parse_args()

    lines = [l for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    for l in lines[: args.n]:
        chat = to_qwen_chat(json.loads(l))
        print(json.dumps(chat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
