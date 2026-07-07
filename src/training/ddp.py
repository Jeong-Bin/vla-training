"""torchrun(DDP) 멀티-GPU 학습 공용 헬퍼.

`torchrun --nproc_per_node=N`이 프로세스를 GPU당 1개씩 띄우면, 각 프로세스(rank)는 환경변수
`RANK`/`LOCAL_RANK`/`WORLD_SIZE`로 자기 위치를 안다. 이 모듈은 그 초기화/정리, "내 rank가 맡을
샘플만 고르기"(샤딩), rank0에서만 출력/저장 같은 반복 패턴을 한 곳에 모은다.

DataParallel(단일 프로세스 모델 복제)이 아니라 **DistributedDataParallel** 방식이다 — 4-bit QLoRA·
`device_map` 고정 모델과 호환되는 유일한 멀티-GPU 경로이기 때문(DataParallel은 양쪽과 충돌).

torchrun 없이 그냥 `python ...`으로 실행하면 WORLD_SIZE가 없어 **단일 GPU**로 자연히 폴백한다.
"""
from __future__ import annotations

import os


# is_dist: torchrun으로 실행됐는가(WORLD_SIZE>1). 아니면 단일 프로세스(단일 GPU)로 동작.
def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


# local_rank: 이 노드 안에서의 GPU 인덱스 = 이 프로세스가 점유할 cuda 디바이스.
def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return rank() == 0


# setup: 분산 프로세스 그룹 초기화 + 이 프로세스의 cuda 디바이스 고정. (단일 실행이면 디바이스만 고정.)
# 반환: 이 프로세스가 쓸 디바이스 문자열("cuda:{local_rank}").
def setup():
    import torch

    lr = local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(lr)
    if is_dist():
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return f"cuda:{lr}"


def cleanup():
    if is_dist():
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


# shard: 인덱스 리스트를 rank별로 분할(rank i는 i, i+W, i+2W, … 를 맡음). 단일 실행이면 전체 반환.
def shard(indices: list) -> list:
    if not is_dist():
        return list(indices)
    return list(indices)[rank():: world_size()]


# barrier: 모든 rank를 동기화(예: rank0 저장 완료를 다른 rank가 기다릴 때).
def barrier():
    if is_dist():
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()


# all_reduce_mean: 스칼라(파이썬 float)를 전 rank 평균으로 집계(로깅용). 단일 실행이면 그대로.
def all_reduce_mean(value: float) -> float:
    if not is_dist():
        return value
    import torch
    import torch.distributed as dist

    t = torch.tensor([value], dtype=torch.float64, device=f"cuda:{local_rank()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item() / world_size())


# all_reduce_sum: 여러 스칼라를 한 번에 전 rank 합산(분산 eval에서 부분합/부분개수 집계용).
# rank별로 다른 개수를 처리해도 (합, 개수)를 따로 sum하면 전 rank 가중평균을 정확히 복원할 수 있다.
def all_reduce_sum(values: list) -> list:
    if not is_dist():
        return list(values)
    import torch
    import torch.distributed as dist

    t = torch.tensor(values, dtype=torch.float64, device=f"cuda:{local_rank()}")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


# all_reduce_min: 정수 스칼라를 전 rank 최소로 집계. rank별 샘플 수가 달라도(샤딩 나머지)
# 학습 step 수를 전 rank 공통 최소값으로 맞춰 epoch 경계의 collective 불일치(데드락)를 막는 데 쓴다.
def all_reduce_min(value: int) -> int:
    if not is_dist():
        return value
    import torch
    import torch.distributed as dist

    t = torch.tensor([value], dtype=torch.int64, device=f"cuda:{local_rank()}")
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


# sync_grads: 파라미터들의 .grad를 전 rank 평균으로 맞춘다(수동 DDP). 모듈을 DDP로 못 감싸는
# 커스텀 forward(여러 손실이 흩어진 학습 루프)에서 backward 직후 호출 → rank 간 grad 동기화.
#   - grad가 None인 파라미터(이 step에 미사용, 예: reasoning 없어 LM head 안 씀)는 0으로 채워
#     all_reduce에 참여시킨다 → 모든 rank가 **동일한 텐서 집합**으로 collective를 호출(횟수·shape 일치)
#     하므로, step별 그래프가 달라도 hang나지 않는다(표준 DDP reducer의 그래프 일치 요구 회피).
def sync_grads(params):
    if not is_dist():
        return
    import torch
    import torch.distributed as dist

    ws = world_size()
    for p in params:
        if not p.requires_grad:
            continue
        if p.grad is None:                                 # 이 rank에서 이번 step 미사용 → 0 grad로 참여
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= ws
