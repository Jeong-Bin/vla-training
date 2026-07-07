"""flow-matching DiT 궤적 planning 헤드 (trajectory objective).

논문 nuVLA가 VLM 백본 위에 올린 flow-matching DiT 궤적 헤드를 **미니 규모로 재현**한다.
VLM(Qwen3-VL)이 8뷰 surround를 인코딩해 만든 condition 벡터를 받아, ego-frame 미래 waypoints
(N×2 [fwd,left])를 **rectified flow**(직선 보간 flow-matching)로 생성한다.

설계(작은 데이터·단일 24GB GPU 대응):
  - 입력 x: (B,N,2) noised waypoints / t: (B,) flow time∈[0,1] / cond: (B,Dc) VLM feature
  - 출력 v: (B,N,2) velocity field. AdaLN-Zero 조건화(처음엔 identity → 안정적 학습).
  - rectified flow: x0~N(0,I), xt=(1−t)x0+t·x1, target v=x1−x0, loss=MSE(v̂,v).
  - 샘플링: x0~N(0,I)에서 Euler ODE 적분(steps번) → 예측 waypoints(정규화 공간).
waypoints는 정규화 공간에서 다루며(평균≈0,표준편차≈1) 정규화/역정규화는 TrajectoryNormalizer가 담당.
헤드 자체는 정규화를 모른다(순수 (xt,t,cond)→v) — 관심사 분리.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# timestep_embedding: flow time t∈[0,1]을 sinusoidal 임베딩으로(diffusion 표준). t를 1000배 스케일해 해상도 확보.
def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None] * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:                                            # 홀수 dim이면 0 패딩
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# modulate: AdaLN scale/shift 적용. (1+scale)로 곱하므로 scale=0이면 항등(AdaLN-Zero 초기값).
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# DiTBlock: AdaLN-Zero 조건화 transformer 블록. condition c로 self-attn/MLP의 shift·scale·gate를 생성.
# gate를 0으로 초기화(AdaLN-Zero)해 학습 초반엔 잔차가 0 → identity로 시작(안정적).
#   cross_attn=True(논문 nuVLA): self-attn 뒤에 **cross-attention** 서브레이어 추가 — waypoint 토큰(query)이
#   VLM feature 시퀀스(mem_kv, key/value)를 참조한다. AdaLN-Zero 게이트(gate_c, 0-init)로 identity 시작.
class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0, cross_attn: bool = False):
        super().__init__()
        self.cross_attn = cross_attn
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        if cross_attn:                                     # waypoint(query) → VLM 시퀀스(key/value) cross-attn
            self.cross_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
            self.cross = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.cross_gate = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model))  # 0-init gate(AdaLN-Zero)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))   # → 6개 변조 파라미터

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                mem_kv: "torch.Tensor | None" = None, mem_kpm: "torch.Tensor | None" = None) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, dim=-1)
        h = modulate(self.norm1(x), shift_a, scale_a)
        attn, _ = self.attn(h, h, h, need_weights=False)  # waypoint 간 양방향 self-attention
        x = x + gate_a.unsqueeze(1) * attn
        if self.cross_attn and mem_kv is not None:        # cross-attention: query=waypoint, kv=VLM 시퀀스
            q = self.cross_norm(x)
            ca, _ = self.cross(q, mem_kv, mem_kv, key_padding_mask=mem_kpm, need_weights=False)
            x = x + self.cross_gate(c).unsqueeze(1) * ca   # gate 0-init → 초반 identity
        h = modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


# TrajectoryDiT: VLM condition을 받아 ego-frame waypoints를 생성하는 flow-matching DiT.
class TrajectoryDiT(nn.Module):
    def __init__(self, cond_dim: int, n_points: int = 50, point_dim: int = 2,
                 d_model: int = 256, n_layers: int = 4, n_heads: int = 4, ego_dim: int = 0,
                 ego_as_state_token: bool = False,
                 beta_alpha: float = 2.0, beta_beta: float = 2.0,
                 cross_attn: bool = False):
        super().__init__()
        self.n_points, self.point_dim, self.d_model = n_points, point_dim, d_model
        self.ego_dim = ego_dim
        # VLM feature 소비 방식(논문 nuVLA #9):
        #   cross_attn=True: DiT가 VLM hidden **시퀀스 전체**(mem, image+prompt 토큰)에 cross-attention.
        #     scene 정보는 cross-attn으로 들어오고 AdaLN 조건 c는 **timestep t만**(PixArt/논문식). pooled cond
        #     미사용 → cond_proj 생성 안 함(DDP unused-param 방지). mem은 mem_proj+mem_norm으로 d_model KV 변환.
        #   False(구형): VLM hidden을 mean-pool한 pooled cond을 cond_proj로 AdaLN c에 더한다(mem 미사용).
        self.cross_attn = cross_attn
        # flow-matching 시간 t 샘플링 분포(논문: "sample t from a Beta distribution"). Beta(α,β)로 중간 t를
        #   봉긋하게 뽑아(기본 Beta(2,2)=대칭 mode 0.5) 어려운 중간 구간에 학습 신호를 집중 → K=5 소스텝 샘플링
        #   품질↑. α=β=1.0이면 Beta(1,1)=Uniform(구형 torch.rand와 동일)이라 하위호환. 논문이 α,β 수치는 미명시.
        self.beta_alpha, self.beta_beta = float(beta_alpha), float(beta_beta)
        # ego 입력(dynamics[+history]) 처리 방식:
        #   ego_as_state_token=True(논문 nuVLA): ego를 **state token** 1개로 임베딩해 waypoint 토큰들 앞에 붙여
        #     self-attention에 참여시킨다(DiT 시퀀스 = [state] + [T waypoint]). ego dynamics + 이동 history를
        #     flatten한 벡터가 ego_dim. 출력은 waypoint 토큰만 사용.
        #   False(구형·하위호환): ego를 ego_proj로 AdaLN 조건 c에 더한다(과거 방식).
        self.ego_as_state_token = ego_as_state_token and ego_dim > 0
        self.x_embed = nn.Linear(point_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_points, d_model))       # waypoint 순서 임베딩
        self.t_embed = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        if cross_attn:                                     # VLM 시퀀스 → d_model KV(블록 공용, forward에서 1회 투영)
            self.mem_proj = nn.Linear(cond_dim, d_model)
            self.mem_norm = nn.LayerNorm(d_model, eps=1e-6)
        else:                                              # 구형: pooled cond → AdaLN 조건
            self.cond_proj = nn.Sequential(nn.Linear(cond_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        # ego 정규화 통계 버퍼(ego_dim>0, 두 방식 공용) — 학습기가 set_ego_stats로 per-dim mean/std 적합.
        if ego_dim > 0:
            self.register_buffer("ego_mean", torch.zeros(ego_dim))
            self.register_buffer("ego_std", torch.ones(ego_dim))
            if self.ego_as_state_token:                    # 논문식: ego → state token(임베딩) + 토큰 타입 임베딩
                self.state_embed = nn.Sequential(nn.Linear(ego_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
                self.state_type = nn.Parameter(torch.zeros(1, 1, d_model))
            else:                                          # 구형: ego → AdaLN 조건 경로
                self.ego_proj = nn.Sequential(nn.Linear(ego_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, cross_attn=cross_attn) for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.final_linear = nn.Linear(d_model, point_dim)
        # cond 정규화 통계(per-dim mean/std) 버퍼. VLM hidden mean-pool은 massive-activation 공통성분이
        # 지배해 장면 간 cond이 거의 동일(cosine~0.99) → DiT가 cond을 무시하고 평균궤적으로 붕괴한다.
        # 학습기가 학습셋 subsample로 per-dim mean/std를 적합해 set_cond_stats로 채우면, forward에서
        # cond을 (cond-mean)/std로 정규화해 장면별 차이가 드러난다(진단: cosine 0.99→0.29, mean collapse 해소).
        # 버퍼는 state_dict에 저장 → 평가 시 자동 복원(eval 스크립트 변경 불필요). 기본값(0/1)이면 항등.
        self.register_buffer("cond_mean", torch.zeros(cond_dim))
        self.register_buffer("cond_std", torch.ones(cond_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        if getattr(self, "ego_as_state_token", False):
            nn.init.normal_(self.state_type, std=0.02)     # state token 타입 임베딩(waypoint와 구분)
        for b in self.blocks:                              # AdaLN-Zero: 변조 출력 0-init → gate=0(identity 시작)
            nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
            if getattr(b, "cross_attn", False):            # cross-attn 게이트도 0-init → 초반 cross 잔차 0
                nn.init.zeros_(b.cross_gate[-1].weight); nn.init.zeros_(b.cross_gate[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight); nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.final_linear.weight); nn.init.zeros_(self.final_linear.bias)

    # set_cond_stats: 학습기가 적합한 cond 정규화 통계(per-dim mean/std)를 버퍼에 채운다.
    #   std는 0 division 방지로 1e-6 clamp. 호출 후 forward가 cond을 자동 정규화한다.
    @torch.no_grad()
    def set_cond_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.cond_mean.copy_(mean.to(self.cond_mean.device, self.cond_mean.dtype))
        self.cond_std.copy_(std.clamp_min(1e-6).to(self.cond_std.device, self.cond_std.dtype))

    # set_ego_stats: ego 운동상태 정규화 통계(per-dim mean/std)를 버퍼에 채운다(ego_dim>0일 때).
    @torch.no_grad()
    def set_ego_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.ego_mean.copy_(mean.to(self.ego_mean.device, self.ego_mean.dtype))
        self.ego_std.copy_(std.clamp_min(1e-6).to(self.ego_std.device, self.ego_std.dtype))

    # forward: (noised waypoints, flow time, cond/mem, [ego 운동상태]) → velocity field. 전부 fp32(작은 헤드라 안정 우선).
    #   cross_attn=True: mem(VLM 시퀀스)+mem_mask로 cross-attention, AdaLN c=timestep t만(cond 미사용).
    #   False(구형): cond(pooled)을 정규화해 AdaLN c에 더함(mem 미사용).
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None,
                ego: Optional[torch.Tensor] = None,
                mem: Optional[torch.Tensor] = None, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.float()
        h = self.x_embed(x) + self.pos_embed               # (B, T, d) waypoint 토큰
        c = self.t_embed(timestep_embedding(t, self.d_model))   # AdaLN 조건: 항상 timestep 포함
        mem_kv = mem_kpm = None
        if self.cross_attn:                                # VLM 시퀀스 → KV(블록 공용 1회 투영). c는 t만(논문식)
            if mem is not None:
                mem_kv = self.mem_norm(self.mem_proj(mem.float()))          # (B, L, d)
                mem_kpm = (~mem_mask.bool()) if mem_mask is not None else None  # True=무시(패딩/응답 토큰)
        else:                                              # 구형: pooled cond을 정규화해 AdaLN에 더함
            cond = (cond.float() - self.cond_mean) / self.cond_std          # massive-activation 공통성분 제거
            c = c + self.cond_proj(cond)
        has_state = self.ego_as_state_token and ego is not None
        if has_state:                                      # 논문식: ego(dynamics+history)를 state token으로 prepend
            ego_n = (ego.float() - self.ego_mean) / self.ego_std
            state_tok = self.state_embed(ego_n).unsqueeze(1) + self.state_type   # (B, 1, d)
            h = torch.cat([state_tok, h], dim=1)           # (B, 1+T, d) — waypoint가 self-attn으로 state를 참조
        elif self.ego_dim > 0 and ego is not None:         # 구형: ego → AdaLN 조건 c
            ego_n = (ego.float() - self.ego_mean) / self.ego_std
            c = c + self.ego_proj(ego_n)
        for blk in self.blocks:
            h = blk(h, c, mem_kv, mem_kpm)
        if has_state:                                      # state token 제거 → waypoint 토큰만 velocity로 디코드
            h = h[:, 1:]
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        return self.final_linear(h)                        # (B,T,point_dim) velocity

    # flow_loss: rectified-flow 학습 손실. x1=정규화된 GT waypoints, cond=pooled(구형)/mem=VLM 시퀀스(cross_attn).
    def flow_loss(self, x1: torch.Tensor, cond: Optional[torch.Tensor] = None,
                  ego: Optional[torch.Tensor] = None,
                  mem: Optional[torch.Tensor] = None, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x1 = x1.float()
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        # 논문: t ~ Beta(α,β) (중간 t 집중). Beta(1,1)이면 Uniform과 동일(하위호환).
        if self.beta_alpha == 1.0 and self.beta_beta == 1.0:
            t = torch.rand(B, device=x1.device)
        else:
            a = torch.full((B,), self.beta_alpha, device=x1.device)
            b = torch.full((B,), self.beta_beta, device=x1.device)
            t = torch.distributions.Beta(a, b).sample()   # (B,) ∈(0,1)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v_target = x1 - x0                                 # rectified flow의 상수 velocity
        v_pred = self.forward(xt, t, cond, ego, mem=mem, mem_mask=mem_mask)
        return ((v_pred - v_target) ** 2).mean()

    # sample: Euler ODE 적분으로 궤적 생성(정규화 공간). cond:(B,Dc) → (B,N,point_dim). 역정규화는 호출측.
    #   deterministic=True면 x0=노이즈 대신 **분포 평균(0)**에서 적분 → 결정론적·재현가능하며,
    #   덜 학습된 모델에서도 매끄럽고 ego 근처에서 시작하는 대표 궤적을 준다(랜덤 draw는 꼬이고 시작점이 튐).
    #   False(기본)는 x0~N(0,I) 랜덤(논문식 stochastic 샘플).
    #   steps=5(논문 K=5): flow-matching은 경로가 거의 직선이 되도록 학습되어(특히 t~Beta로 중간 구간
    #   집중 학습된 경우) 적은 스텝으로도 정확 — diffusion과 달리 수백 스텝이 불필요.
    @torch.no_grad()
    def sample(self, cond: Optional[torch.Tensor] = None, steps: int = 5, deterministic: bool = False,
               ego: Optional[torch.Tensor] = None,
               mem: Optional[torch.Tensor] = None, mem_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        ref = mem if self.cross_attn else cond             # cross_attn이면 mem에서 B/device 유도(cond=None 허용)
        B, dev = ref.shape[0], ref.device
        x = (torch.zeros if deterministic else torch.randn)(
            B, self.n_points, self.point_dim, device=dev)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), i * dt, device=dev)
            x = x + self.forward(x, t, cond, ego, mem=mem, mem_mask=mem_mask) * dt
        return x


# TrajectoryNormalizer: waypoints를 평균0·표준편차1 공간으로 정규화(flow-matching 안정화). 채널별(fwd,left) 통계.
class TrajectoryNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean.float()
        self.std = std.float()

    @classmethod
    def fit(cls, waypoints) -> "TrajectoryNormalizer":
        """waypoints: iterable of (N,2). 전체를 모아 채널별 mean/std 계산."""
        arr = torch.tensor([pt for w in waypoints for pt in w], dtype=torch.float32)  # (M*N, 2)
        return cls(arr.mean(0), arr.std(0).clamp_min(1e-6))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, d: dict) -> "TrajectoryNormalizer":
        return cls(torch.tensor(d["mean"]), torch.tensor(d["std"]))


__all__ = ["TrajectoryDiT", "TrajectoryNormalizer", "timestep_embedding"]
