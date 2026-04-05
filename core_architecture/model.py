# model.py — ProteinMPNN: Inverse Folding Model
# ======================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange

from .config import Config, EncoderOutput, DesignOutput
from .config import NUM_AA, MASK_TOKEN, NUM_TOKENS, NUM_ATOM_PAIRS
from .preprocessing import build_knn_graph, compute_edge_features
from .encoder import EncoderLayer
from .decoder import DecoderLayer


class ProteinMPNN(nn.Module):

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        h     = cfg.hidden_dim
        e_raw = NUM_ATOM_PAIRS * cfg.num_rbf   # 25 × 16 = 400

        # Sequence embedding: {0,...,20} → R^{hidden}
        # 20 AA + 1 MASK = 21 tokens
        self.seq_emb = nn.Embedding(NUM_TOKENS, h)

        # Encoder: edge_in_dim은 첫 레이어만 e_raw(400), 이후 h(128)
        enc_layers = []
        in_dim = e_raw
        for _ in range(cfg.n_enc_layers):
            enc_layers.append(EncoderLayer(cfg, edge_in_dim=in_dim))
            in_dim = h                 # 첫 레이어 이후 edge dim = hidden
        self.encoder = nn.ModuleList(enc_layers)

        # Decoder: 모든 레이어 동일 (decoder edge = h + h = 2h)
        self.decoder = nn.ModuleList([
            DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)
        ])

        # Output projection: hidden → 20 AA logits
        self.output_proj = nn.Linear(h, NUM_AA)

        # Xavier Uniform 초기화:
        #   선형 레이어의 초기 분산을 입출력 크기에 맞게 설정.
        #   수식: W ~ Uniform(-√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out)))
        #   목표: 초기화 직후 각 레이어의 출력 분산이 입력 분산과 동일 → stable forward pass
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Encode ────────────────────────────────────────────────────────
    def encode(
        self,
        n_coords  : Float[Tensor, "res xyz"],
        ca_coords : Float[Tensor, "res xyz"],
        c_coords  : Float[Tensor, "res xyz"],
        o_coords  : Float[Tensor, "res xyz"],
    ) -> EncoderOutput:
        # ──────────────────────────────────────────────────────────────
        #  공간 변환:  P → G → Z
        #
        #  P = R^{res × 5atoms × 3}  (물리 공간, backbone 좌표)
        #    ↓ build_knn_graph + compute_edge_features
        #  G = (edge_idx, edge_h_raw)
        #    edge_idx ∈ Z^{res×k}:  topology
        #    edge_h_raw ∈ R^{res×k×400}: RBF feature
        #    ↓ EncoderLayer × n_enc_layers (message passing)
        #  Z = (node_h, edge_h)
        #    node_h ∈ R^{res×hidden}: 각 잔기의 구조 표현
        #    edge_h ∈ R^{res×k×hidden}: 각 엣지의 구조 표현
        # ──────────────────────────────────────────────────────────────

        res    = ca_coords.shape[0]
        device = ca_coords.device

        # P → G: k-NN graph + edge features
        _, edge_idx = build_knn_graph(ca_coords, self.cfg.k_neighbors)
        edge_h_raw  = compute_edge_features(
            n_coords, ca_coords, c_coords, o_coords, edge_idx, self.cfg
        )  # (res, k, 400)

        # G → Z: message passing
        # node_h 초기값 = 0: 구조 정보는 edge에만 있음, node는 message를 통해 채워짐
        node_h = torch.zeros(res, self.cfg.hidden_dim, device=device)
        edge_h = edge_h_raw   # 첫 레이어는 raw (400), 이후 hidden (128)

        for layer in self.encoder:
            node_h, edge_h = layer(node_h, edge_h, edge_idx)

        return EncoderOutput(node_h=node_h, edge_h=edge_h, edge_idx=edge_idx)

    # ── Decode (Autoregressive Inference) ─────────────────────────────
    def decode(
        self,
        enc_out      : EncoderOutput,
        partial_seq  : Int[Tensor, "res"] | None = None,
        decode_order : Int[Tensor, "res"] | None = None,
        temperature  : float = 0.1,
    ) -> DesignOutput:

        res    = enc_out.node_h.shape[0]
        device = enc_out.node_h.device
        k      = enc_out.edge_idx.shape[1]

        # Decode order: 기본은 random permutation (위치 편향 제거)
        if decode_order is None:
            decode_order = torch.randperm(res, device=device)

        # 현재 서열 상태: MASK로 초기화 후 partial_seq 적용
        seq = torch.full((res,), MASK_TOKEN, dtype=torch.long, device=device)
        if partial_seq is not None:
            fixed       = partial_seq != MASK_TOKEN
            seq[fixed]  = partial_seq[fixed]

        logits_buf   = torch.zeros(res, NUM_AA, device=device)
        log_prob_sum = torch.tensor(0.0, device=device)
        n_designed   = 0
        decoded_set  : set[int] = set()

        # framework 위치를 decoded_set에 미리 등록
        if partial_seq is not None:
            fw_indices = torch.where(partial_seq != MASK_TOKEN)[0].tolist()
            decoded_set.update(fw_indices)

        for step in range(res):
            pos_i = int(decode_order[step].item())

            # 이미 결정된 위치(framework)는 skip
            if partial_seq is not None and partial_seq[pos_i] != MASK_TOKEN:
                continue

            # ar_mask: 현재 잔기 pos_i의 이웃 중 decoded_set에 있는 것만 1
            # (res, k) 전체 중 pos_i 행만 업데이트 (나머지는 0)
            ar_mask = torch.zeros(res, k, device=device)
            for j_local, j_global in enumerate(enc_out.edge_idx[pos_i].tolist()):
                if j_global in decoded_set:
                    ar_mask[pos_i, j_local] = 1.0

            # seq_emb: 현재 결정 상태의 서열을 임베딩
            seq_emb = self.seq_emb(seq)        # (res, hidden)

            # Decoder forward: 구조 표현 + 서열 조건 → 갱신된 node_h
            node_h = enc_out.node_h.clone()
            for layer in self.decoder:
                node_h = layer(
                    node_h, enc_out.edge_h, enc_out.edge_idx,
                    seq_emb, ar_mask,
                )

            # pos_i 위치의 logit → sampling
            logit_i = self.output_proj(node_h[pos_i])   # (20,)
            logit_i = logit_i.nan_to_num(0.0)           # NaN 방어 (미학습 모델)

            # Temperature scaling with max subtraction for numerical stability
            scaled  = (logit_i - logit_i.max()) / max(temperature, 1e-6)
            prob_i  = F.softmax(scaled, dim=-1)          # (20,)
            aa_i    = torch.multinomial(prob_i, 1).squeeze(-1)

            logits_buf[pos_i]  = logit_i
            seq[pos_i]         = aa_i
            log_prob_sum      += torch.log(prob_i[aa_i] + 1e-8)
            n_designed        += 1
            decoded_set.add(pos_i)

        mean_lp = log_prob_sum / max(n_designed, 1)
        return DesignOutput(logits=logits_buf, sequence=seq, log_prob=mean_lp)

    # ── Forward (Teacher-Forcing Training) ────────────────────────────
    def forward(
        self,
        n_coords  : Float[Tensor, "res xyz"],
        ca_coords : Float[Tensor, "res xyz"],
        c_coords  : Float[Tensor, "res xyz"],
        o_coords  : Float[Tensor, "res xyz"],
        sequence  : Int[Tensor, "res"],
    ) -> Float[Tensor, "res aa"]:

        enc_out = self.encode(n_coords, ca_coords, c_coords, o_coords)
        seq_emb = self.seq_emb(sequence)                # (res, hidden)
        res, k  = enc_out.node_h.shape[0], enc_out.edge_idx.shape[1]

        # Teacher-forcing: ar_mask = all-ones (모든 이웃 서열 visible)
        ar_mask = torch.ones(res, k, device=ca_coords.device)

        node_h = enc_out.node_h
        for layer in self.decoder:
            node_h = layer(
                node_h, enc_out.edge_h,
                enc_out.edge_idx, seq_emb, ar_mask,
            )

        return self.output_proj(node_h)    # (res, 20)
