# training.py — Loss 함수와 학습 루프

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import rearrange

from .config import Config
from .model import ProteinMPNN


def sequence_nll_loss(
    logits       : Float[Tensor, "res aa"],
    targets      : Int[Tensor, "res"],
    mask         : Bool[Tensor, "res"] | None = None,
    label_smooth : float = 0.1,
) -> Float[Tensor, ""]:
    # ──────────────────────────────────────────────────────────────────
    #  공간 변환:  L × Seq → R  (scalar loss)
    #
    #  수식:
    #    log_probs[r, a] = log_softmax(logits[r, :])_a
    #    nll[r]          = -log_probs[r, seq[r]]       ← 정답 AA의 log-prob
    #    smooth[r]       = mean_a(log_probs[r, :])      ← 모든 AA 평균 log-prob
    #    loss[r]         = (1-ε)·nll[r] - ε·smooth[r]
    #    L               = sum(loss[mask]) / normalizer
    #
    #  Normalizer = 2000:
    #    원저자 설정. 잔기 수로 나누는 대신 고정값 사용.
    #    다양한 길이의 단백질에서 loss scale을 일정하게 유지.
    # ──────────────────────────────────────────────────────────────────

    # L → (res, 20)
    log_probs = F.log_softmax(logits, dim=-1)

    # NLL: 정답 AA의 log-prob 추출
    # targets: (res,) → (res, 1) → gather → (res, 1) → (res,)
    tgt_idx = rearrange(targets, 'res -> res 1')
    nll     = -log_probs.gather(dim=1, index=tgt_idx)
    nll     = rearrange(nll, 'res 1 -> res')          # (res,)

    # Label smoothing 보정항: 균일 분포에 대한 log-prob 평균
    smooth  = reduce_mean_last(log_probs)              # (res,)
    loss    = (1.0 - label_smooth) * nll - label_smooth * smooth

    # mask가 있으면 CDR 등 특정 위치만 loss에 반영
    if mask is not None:
        loss = loss * mask.float()

    return loss.sum() / 2000.0


def reduce_mean_last(x: Float[Tensor, "res aa"]) -> Float[Tensor, "res"]:
    # (res, aa) → (res,): aa 축 평균
    # einops reduce 사용
    from einops import reduce as einops_reduce
    return einops_reduce(x, 'res aa -> res', 'mean')


def sequence_recovery(
    logits  : Float[Tensor, "res aa"],
    targets : Int[Tensor, "res"],
    mask    : Bool[Tensor, "res"] | None = None,
) -> float:
    # ──────────────────────────────────────────────────────────────────
    #  Sequence Recovery Rate:
    #    (1/N) Σ_r 1[argmax logits[r] == targets[r]]
    #
    #  의미: 구조를 조건으로 예측한 아미노산이 실제 서열과 일치하는 비율.
    #  Native sequence recovery ~40%: 단백질 design의 기준선.
    #  ProteinMPNN: ~50-60% recovery (structure-based)
    # ──────────────────────────────────────────────────────────────────

    pred    = logits.argmax(dim=-1)         # (res,): 가장 높은 logit의 AA
    correct = (pred == targets)             # (res,): Bool

    if mask is not None:
        m = mask.float()
        return (correct.float() * m).sum().item() / (m.sum().item() + 1e-8)
    return correct.float().mean().item()


def training_step(
    model     : ProteinMPNN,
    optimizer : torch.optim.Optimizer,
    batch     : dict,
) -> dict[str, float]:
    # ──────────────────────────────────────────────────────────────────
    #  학습 1스텝:
    #    1. 좌표에 noise 추가 (augmentation)
    #    2. Teacher-forcing forward: structure + seq → logits
    #    3. NLL loss (+ label smoothing) 계산
    #    4. Backward + gradient clipping + optimizer step
    #
    #  Coordinate Noise:
    #    ε ~ N(0, σ²·I_{res×3})  (σ = noise_std = 0.02Å)
    #    s_noisy = s + ε (4개 원자 전부 동일한 ε 사용)
    #    → backbone geometry 유지하며 위치를 약간 흔듦
    #    → 구조 예측 오차(AlphaFold prediction error)에 대한 robustness 확보
    # ──────────────────────────────────────────────────────────────────

    optimizer.zero_grad()

    n_c       = batch["n_coords"]
    ca_c      = batch["ca_coords"]
    c_c       = batch["c_coords"]
    o_c       = batch["o_coords"]
    seq       = batch["sequence"]
    mask      = batch.get("mask", None)
    noise_std = batch.get("noise_std", 0.02)

    # Coordinate augmentation: ε ~ N(0, σ²·I)
    if noise_std > 0:
        # 4개 원자에 동일한 noise: backbone rigid-body shift 모사
        ε   = torch.randn_like(ca_c) * noise_std
        n_c, ca_c, c_c, o_c = n_c + ε, ca_c + ε, c_c + ε, o_c + ε

    # Teacher-forcing forward: (structure, seq) → logits ∈ R^{res × 20}
    logits = model(n_c, ca_c, c_c, o_c, seq)

    # Loss: NLL + label smoothing
    loss = sequence_nll_loss(logits, seq, mask, model.cfg.label_smooth)

    # Backward
    loss.backward()

    # Gradient clipping: ||∇||_2 > 1.0이면 rescale
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return {
        "loss"     : loss.item(),
        "recovery" : sequence_recovery(logits, seq, mask),
    }
