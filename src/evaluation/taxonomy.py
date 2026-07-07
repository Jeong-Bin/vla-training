"""Canonical driving-action taxonomy + rule-based mapper (zero-shot / fine-tuned scoring).

GT decisions are free text ("Slow down gently", "Right lane change"), so exact-match
scoring is impossible. We normalise *both* the GT and the model prediction to a small
canonical taxonomy with the **same** rule-based mapper, then score on the canonical
labels. Using one mapper for both sides keeps zero-shot vs fine-tuned comparisons fair.

Taxonomy (designed from the observed GT vocabulary + spec §8):
  longitudinal : accelerate | maintain | slow_down | stop
  lateral      : keep_lane | nudge_left | nudge_right | change_left | change_right
                 | turn_left | turn_right

Unmappable strings map to ``UNMAPPED`` and are reported as a mapping-failure rate
rather than silently bucketed, so the harness stays honest.
"""
from __future__ import annotations

from typing import Optional

# UNMAPPED: 어느 표준 클래스에도 매핑되지 않은 경우의 표시값.
# 빈 칸으로 숨기지 않고 "매핑 실패율"로 따로 보고하기 위한 sentinel.
UNMAPPED = "unmapped"

# -- longitudinal (종방향: 속도 제어) -------------------------------------
# 종방향 표준 4-클래스: 가속 / 정속 유지 / 감속 / 정지.
# LONGITUDINAL 튜플 = 채점에 쓰는 정식 라벨 집합(+ 모델이 곧바로 이 값을 출력했을 때 통과시키는 화이트리스트 역할).
ACCELERATE = "accelerate"
MAINTAIN = "maintain"
SLOW_DOWN = "slow_down"
STOP = "stop"
LONGITUDINAL = (ACCELERATE, MAINTAIN, SLOW_DOWN, STOP)

# -- lateral (횡방향: 조향/차로) ------------------------------------------
# 횡방향 표준 7-클래스: 차로 유지 / 차로 내 좌·우 미세이동(nudge) / 좌·우 차로변경(change) / 좌·우 회전(turn). 방향이 의미를 가지므로 분리.
KEEP_LANE = "keep_lane"
NUDGE_LEFT = "nudge_left"
NUDGE_RIGHT = "nudge_right"
CHANGE_LEFT = "change_left"
CHANGE_RIGHT = "change_right"
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
LATERAL = (
    KEEP_LANE, NUDGE_LEFT, NUDGE_RIGHT,
    CHANGE_LEFT, CHANGE_RIGHT, TURN_LEFT, TURN_RIGHT,
)


# _direction: 문자열에서 좌/우 방향을 뽑아냄. 'left'와 'right' 중 더 먼저 나오는 단어를 채택, 둘 다 없으면 None. (횡방향 매핑의 보조 함수)
def _direction(t: str) -> Optional[str]:
    """Resolve a left/right direction from text (whichever keyword appears first)."""
    li, ri = t.find("left"), t.find("right")
    if li == -1 and ri == -1:
        return None
    if ri == -1 or (li != -1 and li < ri):
        return "left"
    return "right"


# map_longitudinal: GT/예측의 자유텍스트 종방향 액션 → 표준 4-클래스.
# 키워드 우선순위(먼저 맞는 것 채택): 정지 > 감속(스로틀 해제) > 가속 > 감속 > 정속.
# GT와 예측에 '같은' 매퍼를 써야 zero-shot↔파인튜닝 비교가 공정해짐.
def map_longitudinal(text: Optional[str]) -> str:
    """Map a free-text longitudinal action to the canonical taxonomy.

    Rule order (first match wins): stop > accelerate > slow_down > maintain.
    'brake'/'halt'/'standstill' -> stop; 'speed up'/'accelerate' -> accelerate;
    'slow'/'decelerate'/'reduce'/'ease' -> slow_down; 'maintain'/'keep'/'steady'/
    'constant'/'cruise'/'continue' -> maintain.
    """
    if not text:
        return UNMAPPED
    t = text.strip().lower()                    # t: 소문자·공백정리한 비교용 텍스트
    if t in LONGITUDINAL:                       # already canonical (e.g. model output)
        return t                                # 이미 표준 라벨이면 그대로 통과
    if any(k in t for k in ("stop", "halt", "standstill", "brake")):
        return STOP                             # 정지/제동 계열
    # throttle-release phrases mean slow_down; check before 'acceler' so
    # "ease off the accelerator" is not misread as accelerate.
    if any(k in t for k in ("ease off", "ease up", "let off", "lift off", "off the gas",
                            "off the throttle", "off the accelerator", "coast")):
        return SLOW_DOWN
    if any(k in t for k in ("acceler", "speed up", "speed-up", "increase speed", "faster")):
        return ACCELERATE                       # 가속 계열
    if any(k in t for k in ("slow", "decel", "reduce speed")):
        return SLOW_DOWN                         # 감속 계열
    if any(k in t for k in ("maintain", "keep", "constant", "steady", "cruise",
                            "continue", "same speed", "hold speed")):
        return MAINTAIN                          # 정속 유지 계열
    return UNMAPPED                              # 어디에도 안 맞으면 매핑 실패


# [한국어] map_lateral: 자유텍스트 횡방향 액션 → 표준 7-클래스.
#         우선순위: 차로유지 > 차로변경(좌/우) > 회전(좌/우) > 미세이동(좌/우) > 방향만 있으면 nudge.
#         방향은 _direction()으로 판정. ('차로 내 살짝 이동'은 nudge, '차로 변경'은 change, '회전'은 turn)
def map_lateral(text: Optional[str]) -> str:
    """Map a free-text lateral action to the canonical taxonomy.

    Rule order: keep_lane > lane-change(L/R) > turn(L/R) > nudge(L/R) > bare-direction.
    A 'slight move'/'shift'/'within the lane' resolves to a directional *nudge*; an
    explicit 'lane change'/'merge' to *change*; 'turn' to *turn*. Direction comes
    from the first of 'left'/'right' to appear.
    """
    if not text:
        return UNMAPPED
    t = text.strip().lower()
    if t in LATERAL:                            # already canonical
        return t
    if any(k in t for k in ("no lateral", "keep lane", "keep the lane", "stay in lane",
                            "stay in the lane", "maintain lane", "remain in lane",
                            "center", "centre", "straight ahead", "go straight",
                            "follow the lane", "keep straight")):
        return KEEP_LANE
    direction = _direction(t)
    if any(k in t for k in ("lane change", "change lane", "change to the", "merge")):
        if direction == "left":
            return CHANGE_LEFT
        if direction == "right":
            return CHANGE_RIGHT
    if "turn" in t:
        if direction == "left":
            return TURN_LEFT
        if direction == "right":
            return TURN_RIGHT
    if any(k in t for k in ("slight", "nudge", "move", "shift", "edge", "drift",
                            "bias", "within the lane", "in the lane", "lateral")):
        if direction == "left":
            return NUDGE_LEFT
        if direction == "right":
            return NUDGE_RIGHT
    if direction == "left":                     # bare directional fallback -> nudge
        return NUDGE_LEFT
    if direction == "right":
        return NUDGE_RIGHT
    return UNMAPPED


# [한국어] map_decision: (Longitudinal, Lateral) 한 쌍을 표준 라벨 dict로 묶어 반환하는 편의 함수.
def map_decision(longitudinal: Optional[str], lateral: Optional[str]) -> dict:
    """Map a ``{Longitudinal, Lateral}`` pair to canonical labels."""
    return {
        "longitudinal": map_longitudinal(longitudinal),
        "lateral": map_lateral(lateral),
    }


__all__ = [
    "UNMAPPED",
    "LONGITUDINAL", "LATERAL",
    "ACCELERATE", "MAINTAIN", "SLOW_DOWN", "STOP",
    "KEEP_LANE", "NUDGE_LEFT", "NUDGE_RIGHT",
    "CHANGE_LEFT", "CHANGE_RIGHT", "TURN_LEFT", "TURN_RIGHT",
    "map_longitudinal", "map_lateral", "map_decision",
]
