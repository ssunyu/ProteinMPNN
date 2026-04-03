# decoder.py — Autoregressive Graph Neural Network Decoder
# ======================================================
# [목적]
#   : encoding 된 잔기 상태정보 -> sequence writing 
#
# [과정과 이유]
#   1. Message Passing 
#       : [node 상태,  edge 상태, 서열 embedding] -> 서열 logit
#   2. auto regressive causal masking 
#       : embedded 구조 표현 + 결정된 이웃서열 -> 다음 잔기 예측 
#       : 결정 되지 않은 서열 -> mask token embedding 
#   3. training -> parallel random ordering 
#       : 입체구조는 sequential 하지 않으므로 ordering bias 제거 목적
#       : random ordering에 대한 ar_masking을 parallel processing 
#   4. inference -> sequential 
#       : auto_regressive inference 
#
# [Tensor flow]
#   1. Input 
#       1) encoding_output: node_self, edge_features (N x 128, N x k x 128)
#       2) sequence embedding: sequence_groudtruth (N x 128)
#   2. Message Passing 
#       1) concatenation 
#           : [node_self || node_neighbor || edge_features || sequence_neighbor] (N x k x 128*4)
#       2) masking 
#           : concatenation * random_order_mask ( (N x k x 128*4 ) * (N x k x 1)) -> broadcasting으로 한번에 처리
#       3) projection  
#           : N x k x 512 -> N x k x 128 
#       4) aggregation 
#           : N x k x 128 -> N x 128
#   3. Output 
#       : N x 20 Logits
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
