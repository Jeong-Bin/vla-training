"""학습 실행 로깅/산출물 보관 공용 헬퍼 (text_sft·traj_reas 공통).

요구 동작:
  - 산출물은 `results/<tag>/<YYYYMMDD_HHMMSS>/`에 학습 실행별로 보관(tag=traj_reas|text_sft).
  - **터미널**: step마다 한 줄을 덮어써(`\r`) 실시간 step/loss 변화를 보여주고, epoch가 끝나면
    그 epoch의 종합(평균 loss + eval)을 영구 줄로 남긴다.
  - **로그 파일**: 실시간 step은 적지 않고 epoch 종합만 기록(사람이 나중에 훑기 좋게).
  - DDP-safe: rank0에서만 디렉터리 생성·파일 기록·출력(다른 rank는 no-op).

RunLogger 하나로 위를 캡슐화한다 — 학습 루프는 `step()`/`epoch()`만 부르면 된다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"


# RunLogger: 한 번의 학습 실행에 대한 로깅/산출물 디렉터리 관리자.
#   tag       = "traj_reas" | "text_sft" (results/<tag>/<timestamp>/ 하위에 모든 산출물)
#   is_main   = DDP rank0 여부(False면 모든 출력/기록이 no-op → 중복·경쟁 방지)
class RunLogger:
    def __init__(self, tag: str, is_main: bool = True):
        self.is_main = is_main
        self._live_open = False              # 현재 줄에 덮어쓰기 중인 step 라인이 떠 있는가
        if not is_main:
            self.run_dir = None
            self._fh = None
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RESULTS / tag / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.run_dir / "train.log", "w", buffering=1)  # line-buffered
        self._file(f"=== run {tag}/{stamp} ===")

    # path: 이 실행의 산출물(모델/메트릭 등)을 둘 경로 헬퍼.
    def path(self, *parts) -> Path:
        return self.run_dir.joinpath(*parts)

    # _file: 로그 파일에만 한 줄 기록(터미널엔 안 씀). epoch 종합 전용.
    def _file(self, line: str):
        if self._fh:
            self._fh.write(line + "\n")

    # step: 실시간 진행 표시. 같은 줄을 계속 덮어쓴다(\r, 개행 없음). 로그 파일엔 기록 안 함.
    #   epoch 단위로 step을 표시한다: "epoch 1 | 12/389  loss ...".
    #   ep        = 현재 epoch 번호(1-base)
    #   step_in_ep= 이 epoch 안에서의 step(1-base)
    #   steps_per_epoch = 이 epoch의 총 step 수(없으면 분모 생략)
    def step(self, ep: int, step_in_ep: int, steps_per_epoch: int = 0, **metrics):
        if not self.is_main:
            return
        m = "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                      for k, v in metrics.items())
        denom = f"/{steps_per_epoch}" if steps_per_epoch else ""
        # \033[K = 커서부터 줄 끝까지 지움. 이전(더 긴) step 줄의 꼬리가 남아 "lm"이 "m"처럼
        # 보이던 문제 방지 — \r은 커서만 앞으로 보낼 뿐 줄을 지우지 않기 때문.
        sys.stdout.write(f"\r  epoch {ep} | {step_in_ep}{denom}  {m}\033[K")
        sys.stdout.flush()
        self._live_open = True

    # epoch: 한 epoch 종합을 출력. 떠 있던 step 줄을 개행으로 마감한 뒤, 영구 줄을 터미널+파일에 남긴다.
    #   train_loss = 그 epoch 평균 train loss
    #   eval       = (선택) {지표: 값} dict. eval을 건너뛴 epoch면 None.
    # 형식: "epoch 1 | train_loss 2.10 | EVAL val_loss 8.28 val_lm 2.81 | step 389/389"
    #       → eval 지표는 "EVAL" 태그 뒤에 모아 학습 손실과 시각적으로 구분한다.
    def epoch(self, epoch: int, train_loss: float, eval: dict | None = None, extra: str = ""):
        if not self.is_main:
            return
        if self._live_open:                  # 덮어쓰던 step 줄을 마감(다음 출력이 안 겹치게)
            sys.stdout.write("\n")
            self._live_open = False
        parts = [f"epoch {epoch}", f"train_loss {train_loss:.4f}"]
        if eval:                             # eval 지표는 "EVAL " 접두로 한 덩어리(학습/검증 구분 명확화)
            ev = " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in eval.items())
            parts.append(f"EVAL {ev}")
        if extra:
            parts.append(extra)
        line = " | ".join(parts)
        print(line)                          # 터미널 영구 줄
        self._file(line)                     # 로그 파일(종합만)

    # info: 일반 메시지(설정 요약 등) — 터미널+파일 양쪽에 영구 기록. step 줄이 떠 있으면 먼저 마감.
    def info(self, line: str):
        if not self.is_main:
            return
        if self._live_open:
            sys.stdout.write("\n")
            self._live_open = False
        print(line)
        self._file(line)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None
