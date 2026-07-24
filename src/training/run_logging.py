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
    # active: 가장 최근에 생성된 rank0 RunLogger(모듈 레벨로 추적). main()의 학습 루프 어디서든 미처리
    #   예외가 나면, 호출부(if __name__=="__main__")가 이 인스턴스를 찾아 error()로 train.log에 남긴다
    #   — main() 내부를 통째로 try/except로 재들여쓰기하지 않고도 "예외=로그 없이 사라짐"을 없앤다.
    active: "RunLogger | None" = None

    def __init__(self, tag: str, is_main: bool = True):
        self.is_main = is_main
        self._live_open = False              # 현재 줄에 덮어쓰기 중인 step 라인이 떠 있는가
        if not is_main:
            self.run_dir = None
            self._fh = None
            return
        self._start = datetime.now()          # 종료 시 총 소요시간 계산용(close()에서 사용)
        stamp = self._start.strftime("%Y%m%d_%H%M%S")
        self.run_dir = RESULTS / tag / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.run_dir / "train.log", "w", buffering=1)  # line-buffered
        self._file(f"=== run {tag}/{stamp} ===")
        RunLogger.active = self

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
    #   train_loss    = 그 epoch 평균 train loss(flow 기준, 하위호환 유지)
    #   train_metrics = (선택) {"flow_avg":..,"lm_avg":..,"gate_ce_avg":..} 등 train 쪽 세부 평균. selective_view
    #                   설정에 따라 lm/gate_ce가 없을 수 있어(--reasoning-types 無 / --gate-weight 0) 호출측이
    #                   존재하는 키만 채워 넘긴다 — 여기선 그대로 나열만 한다.
    #   eval          = (선택) {지표: 값} dict. eval을 건너뛴 epoch면 None.
    # 형식: "epoch 1 | train_loss 2.10 | TRAIN flow_avg 2.05 lm_avg 0.62\n"
    #       "                              | EVAL val_loss 8.28 | step 389/389"
    #       → train_metrics까지는 한 줄(| 구분), EVAL부터는 줄바꿈 후 "epoch N | train_loss X.XX | " 폭만큼
    #         들여써 이어붙인다(train/eval 두 블록을 시각적으로 분리하되 같은 epoch 줄임을 들여쓰기로 표시).
    #         eval이 없으면(그 epoch에 검증을 안 함) 줄바꿈 없이 1줄 그대로.
    def epoch(self, epoch: int, train_loss: float, train_metrics: dict | None = None,
              eval: dict | None = None, extra: str = ""):
        if not self.is_main:
            return
        if self._live_open:                  # 덮어쓰던 step 줄을 마감(다음 출력이 안 겹치게)
            sys.stdout.write("\n")
            self._live_open = False
        indent_ref = f"epoch {epoch} | train_loss {train_loss:.4f} "   # 들여쓰기 기준 폭(TRAIN 블록 제외)
        head_parts = [f"epoch {epoch}", f"train_loss {train_loss:.4f}"]
        if train_metrics:                    # train 세부 평균은 "TRAIN " 접두로 한 덩어리(첫 줄에 유지)
            tm = " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in train_metrics.items())
            head_parts.append(f"TRAIN {tm}")
        head = " | ".join(head_parts)
        tail_parts = []
        if eval:                             # eval 지표는 "EVAL " 접두로 한 덩어리(학습/검증 구분 명확화)
            ev = " ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in eval.items())
            tail_parts.append(f"EVAL {ev}")
        if extra:
            tail_parts.append(extra)
        if tail_parts:                       # EVAL/extra는 줄바꿈 후 "epoch N | train_loss X.XX " 폭만큼 들여씀
            indent = " " * len(indent_ref)
            line = head + "\n" + indent + "| " + " | ".join(tail_parts)
        else:
            line = head
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

    # error: 학습/검증 중 처리 못한 예외가 올라올 때 호출측(main의 바깥 try/except)이 부른다.
    #   스택트레이스 전체를 train.log 마지막에 이어붙인다 — 지금까지는 미처리 예외가 로그 없이 stderr로만
    #   사라져(로그가 "running final eval ..." 같은 데서 뚝 끊김) 원인 파악이 어려웠던 문제를 없앤다.
    #   close()보다 먼저 불러야 한다(파일이 아직 열려 있어야 기록됨) — main()이 finally에서 순서 보장.
    def error(self, exc: BaseException):
        if not self.is_main:
            return
        import traceback
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.info(f"=== 에러 발생: {datetime.now():%Y-%m-%d %H:%M:%S} ===\n{tb.rstrip()}")

    # close: 학습+검증+최종평가가 전부 끝난 뒤 호출측(main)이 부른다 — 그 시점을 "완전 종료 시각"으로 기록.
    def close(self):
        if self._fh:
            end = datetime.now()
            dur = end - self._start
            h, rem = divmod(int(dur.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            self.info(f"=== run 종료: {end:%Y-%m-%d %H:%M:%S} (총 소요 {h}h{m:02d}m{s:02d}s) ===")
            self._fh.close()
            self._fh = None
        if RunLogger.active is self:
            RunLogger.active = None
