# config.py — 모델 설정, 전역 상수, 데이터 컨테이너
#
# ══════════════════════════════════════════════════════════════════════
#  이 파일의 역할
# ══════════════════════════════════════════════════════════════════════
#
#  모든 하이퍼파라미터와 공간 정의를 한 곳에 모은다.
#  "무엇을 만드는가"와 "어떤 공간에서 만드는가"를 명시적으로 선언.
#
#  ProteinMPNN이 다루는 공간들:
#
#  [물리 공간] R^3
#    단백질 원자 좌표. Cartesian XYZ. 단위: Å (옹스트롬).
#    backbone 5개 원자: N, Cα, C, O, Cβ(virtual).
#
#  [아미노산 공간] {0, ..., 19}  ← NUM_AA = 20
#    20개의 표준 아미노산. 이산(discrete) 공간.
#    AA_ALPHABET: 알파벳 순서로 인덱스 부여.
#    MASK_TOKEN = 20: "아직 결정되지 않은 잔기"를 표현하는 특수 토큰.
#
#  [임베딩 공간] R^{hidden_dim}
#    신경망이 정보를 처리하는 연속 공간.
#    물리 공간 R^3 → 그래프 feature → R^{hidden_dim}의 변환 체계.
#
#  [그래프 공간] (V, E)
#    V = {0, ..., n_res-1}: 잔기(residue) 노드 집합
#    E ⊆ V × V:             k-NN 기반 인접 엣지 집합 (k = k_neighbors)
#    각 노드: R^{hidden_dim} 벡터
#    각 엣지: R^{edge_dim} 벡터  (edge_dim = NUM_ATOM_PAIRS * num_rbf = 400)

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


# ── 데이터 컨테이너 (NamedTuple) ────────────────────────────────────────
#
# NamedTuple을 사용하는 이유:
#   dict: key가 문자열 → IDE에서 타입 추론 불가
#   dataclass: mutable → 실수로 덮어쓸 위험
#   NamedTuple: immutable + 타입 주석 + 위치/이름 모두 접근 가능
#   → 함수 반환값의 의미가 코드 레벨에서 명확히 표현됨

class EncoderOutput(NamedTuple):
    # Encoder가 생성하는 구조 표현 (structure representation)
    # 이 세 텐서가 decoder로 전달되는 "컴파일된 구조 정보"
    node_h   : Float[Tensor, "res hidden"]   # 각 잔기의 hidden state
    edge_h   : Float[Tensor, "res k hidden"] # 각 엣지의 hidden state
    edge_idx : Int[Tensor, "res k"]          # k-NN 인접 행렬 (topology)


class DesignOutput(NamedTuple):
    # Decoder (AR design)의 출력
    logits   : Float[Tensor, "res aa"]  # 각 위치의 미정규화 AA 점수
    sequence : Int[Tensor, "res"]       # 샘플링된 AA sequence
    log_prob : Float[Tensor, ""]        # 생성 sequence의 mean log-prob (scalar)
