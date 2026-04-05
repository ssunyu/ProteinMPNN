# preprocessing.py — 물리 공간 → 그래프 feature 공간 변환
# ======================================================
# [목적]
#   : 단백질 물리 공간 -> graph 공간, 공간 변환 
#
# [과정과 이유]
#   1. 단백질 backbone 3차원 물리공간 -> graph 공간 (compute_edge_features)
#       : 잔기간 물리적 상호작용(latent Z)에 대한 표현공간으로 graph 가정
#   2. k개의 interaction 노드 선정 (build_knn_graph)
#       : local interaction이 잔기간 구조를 결정한다는 가정으로 top_k 노드 선정 
#   3. RBF embedding (compute_rbf)
#       : 거리에 따른 qualitative 상호작용의 변화 가정
#
# [Tensor flow]
#   1. Raw 
#       : res x atoms x 3 (res x 5 x 3)
#   2. Grpah 
#       : res x k x edges (res x k x 25)
#   3. RBF 
#       : res x k x edges * RBF (res x k x 400)
# 

from __future__ import annotations
import math

import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import rearrange, repeat

from .config import Config, NUM_ATOMS, NUM_ATOM_PAIRS


# ── Virtual Cβ ─────────────────────────────────────────────────────────
def compute_virtual_cb(
    n_coords  : Float[Tensor, "res xyz"],
    ca_coords : Float[Tensor, "res xyz"],
    c_coords  : Float[Tensor, "res xyz"],
) -> Float[Tensor, "res xyz"]:
    # ──────────────────────────────────────────────────────────────────
    #  수식 (tetrahedral 기하학):
    #    b = Cα - N          (N→Cα 방향 벡터)
    #    c = C  - Cα         (Cα→C 방향 벡터)
    #    a = b × c           (backbone 평면의 법선 벡터)
    #    Cβ = -0.58273431·a + 0.56802827·b - 0.54067466·c + Cα
    #
    #  Tips: 
    #    1. 왜 virtual Cβ가 필요한가:
    #       Glycine(Gly)은 Cβ가 없다. 하지만 Cβ의 위치는 잔기의 방향성
    #       (side chain이 어느 방향을 향하는가)을 인코딩하는 핵심 정보.
    #       Virtual Cβ = backbone 기하학(N, Cα, C)으로부터 Cβ 위치를 계산.
    #       → Gly도 포함해 모든 잔기에서 동일한 5-atom 표현 사용 가능.
    #    2. 계수 근거: tetrahedral sp³ 탄소의 이상적 결합각(109.5°)과
    #       N-Cα-C-Cβ 이면각을 수치적으로 최적화해 사전 계산된 값.
    # ──────────────────────────────────────────────────────────────────

    b = ca_coords - n_coords            # N→Cα: (res, 3)
    c = c_coords  - ca_coords           # Cα→C: (res, 3)
    a = torch.cross(b, c, dim=-1)       # backbone 법선: (res, 3)

    return (
        -0.58273431 * a
        + 0.56802827 * b
        - 0.54067466 * c
        + ca_coords
    )


# ── RBF Encoding ────────────────────────────────────────────────────────
def compute_rbf( # 거리라는 물리적 1d축을 따라 모든 데이터에 반복 적용-> broadcast function으로 design 
    dist   : Float[Tensor, "..."], # why Tensor? - parallel processing을 위해서
    d_min  : float,
    d_max  : float,
    n_rbf  : int,
) -> Float[Tensor, "... n_rbf"]:
    # ──────────────────────────────────────────────────────────────────
    #  공간 변환:  R^{...} → R^{... × n_rbf}
    #
    #  수식:
    #    μ_i = d_min + i·(d_max - d_min)/(n_rbf - 1),   i=0,...,n_rbf-1
    #    γ   = (n_rbf - 1) / (d_max - d_min)
    #    φ_i(d) = exp(-γ·(d - μ_i)²)
    #
    #  기하학:
    #    거리 d ∈ R = 한 개의 스칼라
    #    RBF(d) ∈ R^{n_rbf} = "d가 어느 거리 구간에 있는가"를 soft하게 표현
    #    각 basis φ_i: μ_i 근방에서 최대(=1), 멀어질수록 0에 수렴
    #    → n_rbf개의 Gaussian이 [d_min, d_max]를 분할한 "거리 스펙트럼"
    #
    #  Tips: 
    #    broadcasting: 확장하고 싶은 차원을 1로 비워놓기
    # ──────────────────────────────────────────────────────────────────

    # centers: μ_i  ∈ R^{n_rbf}
    centers = torch.linspace(d_min, d_max, n_rbf, device=dist.device)
    γ = (n_rbf - 1) / (d_max - d_min + 1e-8)

    # dist: (...) → (..., 1) for broadcasting with centers: (n_rbf,)
    d_expanded = dist.unsqueeze(-1)          # (..., 1)
    c_expanded = centers.view(*([1] * dist.dim()), n_rbf)   # (1,...,1,n_rbf)

    return torch.exp(-γ * (d_expanded - c_expanded) ** 2)   # (..., n_rbf)


# ── k-NN Graph ──────────────────────────────────────────────────────────
def build_knn_graph(
    ca_coords : Float[Tensor, "res xyz"],
    k         : int,
) -> tuple[Float[Tensor, "res k"], Int[Tensor, "res k"]]:
    # ──────────────────────────────────────────────────────────────────
    #  공간 변환:  P_Cα ⊂ R^{res×3} → G = (V, E)
    #
    #  수식:
    #    D[r, s] = ||Cα_r - Cα_s||_2²   (squared distance matrix)
    #    edge_idx[r, :] = argsort(D[r, :])[1:k+1]  (자기 자신 제외)
    #    knn_dist[r, j] = sqrt(D[r, edge_idx[r,j]])
    # ──────────────────────────────────────────────────────────────────

    res = ca_coords.shape[0]
    k   = min(k, res - 1)

    # Pairwise squared distance: (res, res)
    ca_i = rearrange(ca_coords, 'r xyz -> r 1 xyz')   # (res, 1, xyz)
    ca_j = rearrange(ca_coords, 's xyz -> 1 s xyz')   # (1, res, xyz)
    dist2 = ((ca_i - ca_j) ** 2).sum(dim=-1)          # (res, res)
    dist2.fill_diagonal_(float('inf'))                # 자기 자신 제외

    # top-k smallest
    _, edge_idx = dist2.topk(k, dim=-1, largest=False)  # (res, k)

    # gather로 선택된 k개의 거리 추출
    knn_dist = dist2.gather(1, edge_idx).clamp(min=0).sqrt()  # (res, k) [Å]

    return knn_dist, edge_idx


# ── Edge Features ───────────────────────────────────────────────────────
def compute_edge_features(
    n_coords  : Float[Tensor, "res xyz"],
    ca_coords : Float[Tensor, "res xyz"],
    c_coords  : Float[Tensor, "res xyz"],
    o_coords  : Float[Tensor, "res xyz"],
    idx_knn   : Int[Tensor, "res k"],
    cfg       : Config,
) -> Float[Tensor, "res k edge_dim"]:
    # ──────────────────────────────────────────────────────────────────
    #  공간 변환:  P = R^{res × atoms × 3} → E = R^{res × k × edge_dim}
    #
    #  수식:
    #    atoms_i[r, a, :] = 잔기 r의 원자 a 좌표
    #    atoms_j[r, j, b, :] = 잔기 edge_idx[r,j]의 원자 b 좌표
    #    d[r, j, a, b] = ||atoms_i[r,a] - atoms_j[r,j,b]||_2
    #    rbf[r, j, a, b, :] = RBF(d[r,j,a,b])  ∈ R^{num_rbf}
    #    E[r, j, :] = flatten(rbf[r, j, :, :, :])  ∈ R^{25 × 16 = 400}
    #
    #  Tips:
    #    1. 왜 5×5 모든 원자 쌍을 사용하는가:
    #        Cα-Cα 거리만 쓰면 잔기의 방향성(orientation) 정보 손실.
    #        → 25 쌍의 거리 분포를 통해 backbone geometry의 완전 표현.
    #    2. broadcasting :
    #        edge를 정의하는 기능적 단위 = atoms
    #        res x k x atoms x 3 -> res x k x atoms x atoms
    #        
    # ──────────────────────────────────────────────────────────────────

    res, k   = idx_knn.shape
    n_rbf    = cfg.num_rbf

    # Virtual Cβ 계산
    cb_coords = compute_virtual_cb(n_coords, ca_coords, c_coords)

    # atoms_i: (res, atoms=5, xyz=3)
    atoms = torch.stack([n_coords, ca_coords, c_coords, o_coords, cb_coords], dim=1)

    # atoms_j: k이웃의 원자 좌표 → (res, k, atoms=5, xyz=3)
    # idx_knn: (res, k) → flatten → (res*k,) → index atoms → (res*k, atoms, xyz) → reshape
    flat_idx = rearrange(idx_knn, 'res k -> (res k)')
    atoms_j  = atoms[flat_idx]                                       # (res*k, atoms, xyz)
    atoms_j  = rearrange(atoms_j, '(res k) atoms xyz -> res k atoms xyz', res=res, k=k)

    # atoms_i를 이웃 차원으로 broadcast: (res, 1, atoms, 1, xyz)
    # atoms_j를 원자 쌍 차원으로 broadcast: (res, k, 1, atoms, xyz)
    # diff[r, j, a_i, a_j, :] = atoms_i[r, a_i] - atoms_j[r, j, a_j]
    ai = rearrange(atoms,   'res ai xyz -> res 1 ai 1 xyz')
    aj = rearrange(atoms_j, 'res k aj xyz -> res k 1 aj xyz')

    diff = ai - aj                               # (res, k, ai=5, aj=5, xyz=3)
    dist = diff.norm(dim=-1)                     # (res, k, ai=5, aj=5)

    # RBF encoding: (res, k, ai, aj) → (res, k, ai, aj, num_rbf)
    rbf = compute_rbf(dist, cfg.rbf_d_min, cfg.rbf_d_max, n_rbf)

    # Flatten atom-pair × rbf 축: edge_dim = ai*aj*num_rbf = 25*16 = 400
    return rearrange(rbf, 'res k ai aj rbf -> res k (ai aj rbf)')
