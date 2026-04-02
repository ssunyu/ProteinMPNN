# encoder.py — Graph Neural Network Encoder
#
# ══════════════════════════════════════════════════════════════════════
#  설계 의도: "각 잔기가 이웃으로부터 메시지를 받아 자신의 표현을 갱신한다."
# ══════════════════════════════════════════════════════════════════════
#
#  Encoder가 하는 일:
#    입력: 구조 정보 (edge features: 원자간 거리)
#    출력: 각 잔기와 엣지의 hidden state (구조를 추상화한 표현)
#
#  왜 Graph Neural Network인가:
#    단백질은 "잔기들의 3D 배열"이다. 이를 grid가 아닌 graph로 처리하면:
#    - 잔기 수 N이 가변적이어도 동일한 모델 적용 가능
#    - k-NN 구조로 O(N·k) 계산 (fully-connected O(N²) 대비 효율적)
#    - 구조의 순열 불변성(permutation invariance) 자연스럽게 달성
#
# ══════════════════════════════════════════════════════════════════════
#  수학적 구조 (Message Passing Framework)
# ══════════════════════════════════════════════════════════════════════
#
#  [공간 H_node] Node hidden space  H = R^{res × hidden}
#    각 잔기(노드)의 표현. 초기값: 0 벡터 (구조만으로 학습 시작).
#
#  [공간 H_edge] Edge hidden space  E = R^{res × k × hidden}
#    각 엣지(잔기 쌍)의 표현. 초기값: raw edge features (거리 정보).
#
#  Message Passing 수식 (1 layer):
#
#    1. Message 생성:
#       m[r, j] = MLP_msg(h_i[r] ‖ h_j[r,j] ‖ e[r,j])
#       여기서 h_j[r,j] = h[edge_idx[r,j]]  (이웃 잔기의 hidden state)
#       ‖: concatenation
#
#    2. Aggregation (이웃으로부터 집계):
#       agg[r] = Σ_j m[r, j]    (sum pooling)
#
#    3. Node Update:
#       h'[r] = LayerNorm(h[r] + Dropout(agg[r]))             [잔차 연결 1]
#       h''[r] = LayerNorm(h'[r] + Dropout(FF(h'[r])))        [잔차 연결 2]
#
#    4. Edge Update:
#       e'[r,j] = LayerNorm(e_proj[r,j] + Dropout(MLP_edge(h''[r] ‖ h''[j] ‖ e[r,j])))
#
#  기하학:
#    각 레이어 = H_node × H_edge → H_node × H_edge 의 비선형 변환.
#    n_enc_layers번 반복 = 정보가 최대 n_enc_layers-hop 이웃까지 전파.
#
#  잔차 연결(Residual Connection)의 설계 의도:
#    h'' = h + ΔF(h):  "현재 표현은 보존하고, 학습은 잔차(Δ)를 목표"
#    깊은 네트워크에서 gradient가 h를 통해 직접 흘러 vanishing 방지.
#    초기: ΔF(h) ≈ 0 → h'' ≈ h (identity initialization 효과).
#
#  왜 Sum Pooling인가 (vs mean, max):
#    Sum: 이웃 수에 비례한 정보 축적 → 국소 밀도(density) 표현 가능
#    Mean: 이웃 수 정규화 → 이웃 수 정보 손실
#    Max: 가장 강한 신호만 → 약한 상호작용 손실
#    단백질에서 접촉 수는 구조 정보 → Sum이 적합.
#
# ══════════════════════════════════════════════════════════════════════

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
        #
        #  einops 사용:
        #    neighbor lookup: flat indexing + rearrange로 shape 복원
        #    aggregation: reduce 'res k hidden -> res hidden' sum
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
