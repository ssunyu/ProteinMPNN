# encoder.py — Graph Neural Network Encoder
# ======================================================
# [목적]
#   : 잔기간 물리적 거리 관계 grpah -> 잔기간 local 상태 update 
#
# [과정과 이유]
#   1. Message Passing (EncoderLayer > msg_mlp)
#       : 아미노산 sequence embedding을 위한 node, edge state 정보 확장 
#       : node 상태, edge 상태, 물리적 edge feature 정보 concatenation 
#       : node, edge 상태 및 edge feature update 
#   2. 3-layer encoding 
#       : 3-hop message passing으로 local to global context learning 
#   3. 정보 aggregation 
#       : 아미노산 Sequence Mapping을 위한 정보 aggregation 
#
# [Tensor flow]
#   1. Input  
#       : RBF_edge_feature (Nxkx400)
#   2. Message Passing
#       1) embedding
#           : node_feature, edge_feature embedding (N x 128, N x k x 128)
#       2) concatenation 
#           : [node_self, node_neighbor, edge_feature](N x k x 128*3)
#       3) projection
#           : N x k x 384 -> N x k x 128
#       4) aggregation
#           :  node_self (N x 128)
#   3. Output 
#       : node_self, edge_features (N x 128, N x k x 128)

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange, reduce

from .config import Config


class EncoderLayer(nn.Module):
    # 1개의 message passing 레이어.
    # n_enc_layers번 쌓아서 정보 전파 범위를 확장.

    def __init__(self, cfg: Config, edge_in_dim: int) -> None:
        super().__init__()
        h = cfg.hidden_dim
        # edge_in_dim: 첫 레이어는 raw edge (400), 이후 레이어는 hidden (128)

        def _mlp(in_dim: int) -> nn.Sequential:
            # 3층 MLP: in_dim → h → h → h
            # GELU: ReLU보다 부드러운 비선형성 (음수 입력에서도 gradient 존재)
            return nn.Sequential(
                nn.Linear(in_dim, h), nn.GELU(),
                nn.Linear(h, h),      nn.GELU(),
                nn.Linear(h, h),
            )

        # Message MLP: (h_i ‖ h_j ‖ e_{ij}) → message
        # 입력 dim = h (node_i) + h (node_j) + edge_in_dim (edge)
        self.msg_mlp  = _mlp(h + h + edge_in_dim)

        # Feed-Forward: 각 노드 독립적으로 표현 다듬기
        # 4h 중간층: Transformer FF와 동일한 확장 비율
        self.ff_node  = nn.Sequential(
            nn.Linear(h, h * 4), nn.GELU(),
            nn.Linear(h * 4, h),
        )

        # Edge Update MLP: 업데이트된 노드로 엣지 표현 갱신
        self.edge_mlp = _mlp(h + h + edge_in_dim)

        # Layer Normalization: 학습 안정성
        self.norm1    = nn.LayerNorm(h)
        self.norm2    = nn.LayerNorm(h)
        self.norm_e   = nn.LayerNorm(h)
        self.drop     = nn.Dropout(cfg.dropout)

        # 첫 레이어: edge_in_dim(400) → hidden(128) 투영 필요
        # 이후 레이어: edge_in_dim = h → Identity
        self.edge_in_proj = (
            nn.Linear(edge_in_dim, h) if edge_in_dim != h else nn.Identity()
        )
        self.h = h

    def forward(
        self,
        node_h   : Float[Tensor, "res hidden"],
        edge_h   : Float[Tensor, "res k edge_in"],
        edge_idx : Int[Tensor, "res k"],
    ) -> tuple[Float[Tensor, "res hidden"], Float[Tensor, "res k hidden"]]:
        # ──────────────────────────────────────────────────────────────
        #  전체 수식:
        #    h_j[r,j] = node_h[edge_idx[r,j]]    ← neighbor lookup
        #    msg[r,j] = MLP_msg(h_i[r] ‖ h_j[r,j] ‖ e[r,j])
        #    agg[r]   = Σ_j msg[r,j]              ← sum pooling
        #    node_h'  = LN(node_h + Drop(agg))    ← residual 1
        #    node_h'' = LN(node_h' + Drop(FF(node_h'))) ← residual 2
        #    e'[r,j]  = LN(e_proj[r,j] + Drop(MLP_edge(h''_i ‖ h''_j ‖ e)))
        # ──────────────────────────────────────────────────────────────

        res, k, _ = edge_h.shape

        # Neighbor lookup: h_j[r,j] = node_h[edge_idx[r,j]]
        # flat_idx: (res*k,) → index node_h → (res*k, hidden) → reshape
        flat_idx = rearrange(edge_idx, 'res k -> (res k)')
        h_j = node_h[flat_idx]
        h_j = rearrange(h_j, '(res k) hidden -> res k hidden', res=res, k=k)

        # h_i를 k이웃 차원으로 broadcast: (res, hidden) → (res, k, hidden)
        h_i = rearrange(node_h, 'res hidden -> res 1 hidden').expand(-1, k, -1)

        # Message: (res, k, h+h+edge_in) → (res, k, hidden)
        msg = self.msg_mlp(torch.cat([h_i, h_j, edge_h], dim=-1))

        # Aggregation: (res, k, hidden) → (res, hidden)  [sum over k]
        agg = reduce(msg, 'res k hidden -> res hidden', 'sum')

        # Node Update (residual × 2)
        node_h = self.norm1(node_h + self.drop(agg))
        node_h = self.norm2(node_h + self.drop(self.ff_node(node_h)))

        # Edge Update: 업데이트된 node_h로 edge 표현 갱신
        flat_idx_new = rearrange(edge_idx, 'res k -> (res k)')
        h_j_new = node_h[flat_idx_new]
        h_j_new = rearrange(h_j_new, '(res k) hidden -> res k hidden', res=res, k=k)
        h_i_new = rearrange(node_h, 'res hidden -> res 1 hidden').expand(-1, k, -1)

        edge_proj  = self.edge_in_proj(edge_h)          # edge_in → hidden
        delta_edge = self.edge_mlp(torch.cat([h_i_new, h_j_new, edge_h], dim=-1))
        edge_h_new = self.norm_e(edge_proj + self.drop(delta_edge))

        return node_h, edge_h_new
