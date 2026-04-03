# config.py — 모델 설정, 전역 상수, 데이터 컨테이너
# ======================================================
# [목적]
#   : 하이퍼파라미터 & 공간 정의 

from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor
from jaxtyping import Float, Int


# ── 전역 상수 ──────────────────────────────────────────────────────────
NUM_AA        = 20    # 표준 아미노산 종류 수
MASK_TOKEN    = 20    # decoder의 "아직 결정 안 됨" 상태 토큰 인덱스
NUM_TOKENS    = 21    # embedding table 크기: 20 AA + 1 MASK
NUM_ATOMS     = 5     # backbone 원자: N, Cα, C, O, Cβ
NUM_ATOM_PAIRS = NUM_ATOMS ** 2   # 원자 쌍의 수 = 25
AA_ALPHABET   = "ACDEFGHIKLMNPQRSTVWY"
assert len(AA_ALPHABET) == NUM_AA


# ── 모델 하이퍼파라미터 ────────────────────────────────────────────────
@dataclass
class Config:
    hidden_dim    : int   = 128    # 임베딩 공간의 차원  R^{128}
    k_neighbors   : int   = 48     # k-NN 그래프의 k  (이웃 수)
    n_enc_layers  : int   = 3      # encoder message passing 반복 횟수
    n_dec_layers  : int   = 3      # decoder message passing 반복 횟수
    dropout       : float = 0.1    # dropout 확률
    num_rbf       : int   = 16     # RBF basis 함수 수  (거리 인코딩 해상도)
    rbf_d_min     : float = 2.0    # RBF 최소 거리 [Å]
    rbf_d_max     : float = 22.0   # RBF 최대 거리 [Å]
    label_smooth  : float = 0.1    # label smoothing (NLL loss regularization)
    #
    # hidden_dim = 128:
    #   원저자 설정. R^{128}은 아미노산 선택 정보를 충분히 담으면서
    #   계산 비용이 합리적인 균형점.
    #
    # k_neighbors = 48:
    #   단백질 잔기 당 평균 접촉 수 (contact threshold 8Å 기준)는 ~15개.
    #   k=48은 이보다 넉넉하게 설정해 long-range 상호작용을 포착.
    #   all-to-all (O(N²))보다 훨씬 효율적 (O(N·k)).
    #
    # num_rbf = 16, d_min=2.0Å, d_max=22.0Å:
    #   RBF: 1D 거리를 16차원 basis 벡터로 확장.
    #   d_min=2.0Å: 공유결합 최소 거리 (N-H bond ~1.0Å, Cα-Cα ~3.8Å).
    #   d_max=22.0Å: contact map의 통상적 cutoff.
    #   16 basis: 각 basis가 (22-2)/16 ≈ 1.25Å 해상도로 거리 구간을 분할.


# ── 데이터 컨테이너 ────────────────────────────────────────

class EncoderOutput(NamedTuple):
    # Encoder가 생성하는 구조 표현 (structure representation)
    node_h   : Float[Tensor, "res hidden"]   # 각 잔기의 hidden state
    edge_h   : Float[Tensor, "res k hidden"] # 각 엣지의 hidden state
    edge_idx : Int[Tensor, "res k"]          # k-NN 인접 행렬 (topology)


class DesignOutput(NamedTuple):
    # Decoder (AR design)의 출력
    logits   : Float[Tensor, "res aa"]  # 각 위치의 미정규화 AA 점수
    sequence : Int[Tensor, "res"]       # 샘플링된 AA sequence
    log_prob : Float[Tensor, ""]        # 생성 sequence의 mean log-prob (scalar)
