# decoder.py — Autoregressive Graph Neural Network Decoder
#
# ══════════════════════════════════════════════════════════════════════
#  설계 의도: "Encoder가 구조를 읽었다면, Decoder는 서열을 쓴다.
#              단, 이미 결정된 이웃의 서열 정보를 조건으로 삼아."
# ══════════════════════════════════════════════════════════════════════
#
#  Encoder와 Decoder의 차이:
#    Encoder: 구조(edge features) → 구조 표현(node/edge hidden states)
#             모든 이웃의 정보를 동시에 사용 (fully visible)
#    Decoder: 구조 표현 + 이미 결정된 이웃 서열 → 다음 잔기 예측
#             인과적 마스킹(causal masking): 아직 결정되지 않은 이웃은 0
#
# ══════════════════════════════════════════════════════════════════════
#  수학적 구조 (Autoregressive Conditional Distribution)
# ══════════════════════════════════════════════════════════════════════
#
#  목표:
#    P(seq | structure) = Π_{t=1}^{res} P(aa_{π(t)} | structure, aa_{π(<t)})
#    π: decode order (permutation)
#    π(<t): t번째 이전에 이미 결정된 잔기들
#
#  Decoder Layer의 메시지 전달:
#
#    현재 위치 r을 생성하는 시점에서:
#    - 이미 결정된 이웃 j: seq_emb[j]를 조건으로 사용  (ar_mask[r,j] = 1)
#    - 아직 결정되지 않은 이웃 j: 조건으로 사용 불가     (ar_mask[r,j] = 0)
#
#    수식:
#      seq_j[r,j] = seq_emb[edge_idx[r,j]] × ar_mask[r,j]
#                 ∈ R^{hidden}
#      (ar_mask=0이면 0벡터 → "이 이웃은 아직 모름")
#
#      e_dec[r,j] = edge_h[r,j] ‖ seq_j[r,j]
#                 ∈ R^{hidden + hidden = 2·hidden}
#      (구조 정보 + 서열 정보를 concat → edge에 서열 조건 통합)
#
#      msg[r,j] = MLP_msg(h_i[r] ‖ h_j[r,j] ‖ e_dec[r,j])
#      agg[r]   = Σ_j msg[r,j]
#      h'[r]    = LN(h[r] + Drop(agg[r]))
#      h''[r]   = LN(h'[r] + Drop(FF(h'[r])))
#
#  ar_mask의 기하학:
#    ar_mask: Bool^{res × k} (실수로 표현, 0 or 1)
#    ar_mask × seq_emb: R^{res×k×hidden}에서 "정보 게이트"
#    ar_mask[r,j] = 1: 이웃 j의 서열 정보가 message에 포함됨
#    ar_mask[r,j] = 0: 이웃 j의 서열 정보가 차단됨 (0벡터)
#    → 시간적 인과관계(causal)를 공간적 마스킹으로 표현
#
#  왜 edge에 seq를 concat하는가 (node에 add하지 않고):
#    엣지 (r, j): "잔기 r이 잔기 j의 서열을 조건으로 삼는" 관계
#    이 관계는 방향성이 있음 (j가 결정됐는지는 r→j 방향에 의존)
#    node에 add하면 모든 이웃에 균등하게 적용 → 방향성 손실
#    edge에 concat → 각 이웃 방향의 서열 조건을 독립적으로 표현
#
#  학습 시 (teacher-forcing) vs 추론 시 (autoregressive):
#    학습: ar_mask = all-ones → 모든 이웃 서열이 조건으로 제공
#          P(seq[r] | structure, seq_all_neighbors) 학습
#          → 병렬 계산 가능, 빠른 학습
#    추론: ar_mask = 결정된 이웃만 1
#          실제 생성 조건에 맞게 autoregressive 샘플링
#          → 현실적이지만 단계적 계산 필요
#
# ══════════════════════════════════════════════════════════════════════

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange, reduce

from .config import Config


class DecoderLayer(nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        h = cfg.hidden_dim
        # Decoder edge dim: struct edge + seq embedding = h + h = 2h
        dec_edge_dim = h + h

        def _mlp(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, h), nn.GELU(),
                nn.Linear(h, h),      nn.GELU(),
                nn.Linear(h, h),
            )

        # msg: (h_i ‖ h_j ‖ e_dec) → message
        # e_dec = struct_edge(h) + seq_embedding(h) = 2h
        self.msg_mlp = _mlp(h + h + dec_edge_dim)
        self.ff_node = nn.Sequential(
            nn.Linear(h, h * 4), nn.GELU(),
            nn.Linear(h * 4, h),
        )
        self.norm1 = nn.LayerNorm(h)
        self.norm2 = nn.LayerNorm(h)
        self.drop  = nn.Dropout(cfg.dropout)
        self.h     = h

    def forward(
        self,
        node_h   : Float[Tensor, "res hidden"],
        edge_h   : Float[Tensor, "res k hidden"],
        edge_idx : Int[Tensor, "res k"],
        seq_emb  : Float[Tensor, "res hidden"],
        ar_mask  : Float[Tensor, "res k"],
    ) -> Float[Tensor, "res hidden"]:
        # ──────────────────────────────────────────────────────────────
        #  수식:
        #    seq_j[r,j] = seq_emb[edge_idx[r,j]] * ar_mask[r,j]
        #    e_dec[r,j] = edge_h[r,j] ‖ seq_j[r,j]   ∈ R^{2h}
        #    h_j[r,j]   = node_h[edge_idx[r,j]]
        #    msg[r,j]   = MLP(h_i[r] ‖ h_j[r,j] ‖ e_dec[r,j])
        #    agg[r]     = Σ_j msg[r,j]
        #    h'         = LN(node_h + Drop(agg))
        #    h''        = LN(h' + Drop(FF(h')))
        #
        #  einops:
        #    seq_j lookup: flat indexing + rearrange
        #    ar_mask 적용: rearrange로 hidden 차원 broadcast
        #    aggregation: reduce 'res k hidden -> res hidden' sum
        # ──────────────────────────────────────────────────────────────

        res, k = edge_idx.shape

        # seq_j: 이웃 잔기의 서열 임베딩 (결정되지 않은 이웃은 0)
        flat_idx = rearrange(edge_idx, 'res k -> (res k)')
        seq_j    = seq_emb[flat_idx]
        seq_j    = rearrange(seq_j, '(res k) hidden -> res k hidden', res=res, k=k)

        # ar_mask 적용: (res, k) → (res, k, 1) → broadcast → (res, k, hidden)
        # "결정된 이웃만 서열 정보를 전달"
        ar_gate  = rearrange(ar_mask, 'res k -> res k 1')
        seq_j    = seq_j * ar_gate   # (res, k, hidden): 0 or seq_emb

        # Decoder edge: 구조 정보(edge_h) + 서열 조건(seq_j) 통합
        e_dec = torch.cat([edge_h, seq_j], dim=-1)  # (res, k, 2h)

        # Node hidden 이웃 lookup
        h_j = seq_emb[flat_idx]   # 이 변수명은 node_h를 써야 함
        h_j = node_h[flat_idx]
        h_j = rearrange(h_j, '(res k) hidden -> res k hidden', res=res, k=k)
        h_i = rearrange(node_h, 'res hidden -> res 1 hidden').expand(-1, k, -1)

        # Message: (res, k, h+h+2h=4h) → (res, k, hidden)
        msg = self.msg_mlp(torch.cat([h_i, h_j, e_dec], dim=-1))

        # Aggregation: (res, k, hidden) → (res, hidden) [sum over k]
        agg = reduce(msg, 'res k hidden -> res hidden', 'sum')

        # Node Update (residual × 2)
        node_h = self.norm1(node_h + self.drop(agg))
        node_h = self.norm2(node_h + self.drop(self.ff_node(node_h)))

        return node_h
