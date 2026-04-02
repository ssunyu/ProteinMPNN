# preprocessing.py — 물리 공간 → 그래프 feature 공간 변환
#
# ══════════════════════════════════════════════════════════════════════
#  설계 의도: "3D 유클리드 공간의 원자 좌표를
#              신경망이 학습 가능한 거리 feature 공간으로 변환한다."
# ══════════════════════════════════════════════════════════════════════
#
#  전체 공간 변환 체계:
#
#  [공간 P] Physical space  P = R^{res × atoms × 3}
#    단백질 backbone의 원자 좌표.
#    atoms = 5: N(0), Cα(1), C(2), O(3), Cβ(4, virtual)
#    3: Cartesian XYZ. 단위: Å.
#
#  [공간 G] Graph space  G = (V, E)
#    V = {0,...,res-1}: residue 노드
#    E ⊆ V×V: k-NN 인접 관계  (edge_idx ∈ Z^{res×k})
#    knn_dist ∈ R^{res×k}: 각 엣지의 Cα-Cα 거리
#
#    수식: edge_idx[r, j] = j번째로 가까운 이웃 잔기의 인덱스
#          knn_dist[r, j] = ||Cα_r - Cα_{edge_idx[r,j]}||_2
#
#  [공간 RBF] Radial Basis Function space  RBF = R^{num_rbf}
#    1D 거리 d → num_rbf차원 벡터로 확장.
#    수식: φ_i(d) = exp(-γ(d - μ_i)²)
#          μ_i = d_min + i·(d_max - d_min)/(num_rbf-1)  (center i)
#          γ = (num_rbf-1) / (d_max - d_min)            (width)
#
#    기하학: 거리 d = R^1의 한 점 → RBF = R^{num_rbf}의 한 점
#             num_rbf개의 Gaussian이 [d_min, d_max]를 분할해 "어느 거리 구간인가"를 soft하게 표현.
#             num_rbf=16: 각 basis가 ~1.25Å 구간을 담당.
#
#  [공간 E] Edge feature space  E = R^{res × k × edge_dim}
#    edge_dim = NUM_ATOM_PAIRS × num_rbf = 25 × 16 = 400
#    수식: E[r, j, :] = flatten([φ(d(atom_a^r, atom_b^{j}))] for (a,b) in 5×5)
#    기하학: 각 엣지 = 잔기 쌍 (r, j)의 원자간 거리 분포를 R^{400}으로 표현.
#             5×5=25 원자 쌍 × 16 RBF = 400차원의 "거리 스펙트럼".
#
#  einops 사용 이유:
#    원자 쌍 연산에 unsqueeze+expand+norm 조합 대신
#    rearrange로 broadcasting 의도를 명시.
#    'res atoms xyz -> res 1 atoms xyz'가
#    "잔기 차원은 유지, 이웃 차원으로 broadcast"임을 직접 선언.

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
    #  왜 virtual Cβ가 필요한가:
    #    Glycine(Gly)은 Cβ가 없다. 하지만 Cβ의 위치는 잔기의 방향성
    #    (side chain이 어느 방향을 향하는가)을 인코딩하는 핵심 정보.
    #    Virtual Cβ = backbone 기하학(N, Cα, C)으로부터 Cβ 위치를 계산.
    #    → Gly도 포함해 모든 잔기에서 동일한 5-atom 표현 사용 가능.
    #
    #  수식 (tetrahedral 기하학):
    #    b = Cα - N          (N→Cα 방향 벡터)
    #    c = C  - Cα         (Cα→C 방향 벡터)
    #    a = b × c           (backbone 평면의 법선 벡터)
    #    Cβ = -0.58273431·a + 0.56802827·b - 0.54067466·c + Cα
    #
    #  계수 근거: tetrahedral sp³ 탄소의 이상적 결합각(109.5°)과
    #    N-Cα-C-Cβ 이면각을 수치적으로 최적화해 사전 계산된 값.
    #    실시간 삼각함수 계산 없이 선형 결합으로 Cβ 위치 복원.
    #
    #  einops: 이 함수는 residue 단위 연산이라 rearrange 불필요.
    #    torch.cross는 마지막 축(xyz)에 적용되므로 shape 유지.
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
def compute_rbf(
    dist   : Float[Tensor, "..."],
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
    #  왜 hard binary threshold (d < cutoff → 1 else 0)가 아닌가:
    #    hard threshold: cutoff 근방에서 불연속 → gradient 소실
    #    RBF: 거리의 연속 함수 → 미분 가능 → 학습 가능
    #    또한 n_rbf개의 basis가 서로 다른 거리 스케일의 정보를 동시에 표현
    #
    #  einops rearrange 사용:
    #    dist: (...,) → dist.unsqueeze(-1): (..., 1)
    #    centers: (n_rbf,) → rearrange로 broadcast 방향 명시
    #    결과: (..., n_rbf)
    # ──────────────────────────────────────────────────────────────────

    # centers: μ_i  ∈ R^{n_rbf}
    centers = torch.linspace(d_min, d_max, n_rbf, device=dist.device)
    γ = (n_rbf - 1) / (d_max - d_min + 1e-8)

    # dist: (...) → (..., 1) for broadcasting with centers: (n_rbf,)
    # rearrange는 임의 batch shape(...)를 지원하지 않으므로 unsqueeze 사용
    # → ARENA: "broadcast 의도가 명확할 때는 unsqueeze도 가독성 있음"
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
    #
    #  기하학:
    #    ca_coords: res개의 점이 R^3에 놓인 point cloud
    #    D: res×res 거리 행렬 (완전 그래프의 엣지 가중치)
    #    k-NN: 각 잔기에서 가장 가까운 k개 이웃만 남김
    #    → O(res²) 완전 그래프 → O(res·k) 희소 그래프
    #
    #  einops rearrange로 broadcasting:
    #    ca_coords: (res, xyz)
    #    diff[r, s, :] = ca_coords[r] - ca_coords[s]
    #    = rearrange(ca, 'r xyz -> r 1 xyz') - rearrange(ca, 's xyz -> 1 s xyz')
    #    → (res, res, xyz)
    # ──────────────────────────────────────────────────────────────────

    res = ca_coords.shape[0]
    k   = min(k, res - 1)

    # Pairwise squared distance: (res, res)
    # rearrange로 broadcasting 의도 명시
    ca_i = rearrange(ca_coords, 'r xyz -> r 1 xyz')   # (res, 1, xyz)
    ca_j = rearrange(ca_coords, 's xyz -> 1 s xyz')   # (1, res, xyz)
    dist2 = ((ca_i - ca_j) ** 2).sum(dim=-1)          # (res, res)
    dist2.fill_diagonal_(float('inf'))                  # 자기 자신 제외

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
    #  텐서 shape 흐름:
    #    atoms_i: (res, atoms=5, xyz=3)
    #    atoms_j: (res, k, atoms=5, xyz=3)  ← atoms_i를 idx_knn으로 indexing
    #    diff:    (res, k, atoms_i=5, atoms_j=5, xyz=3)
    #    dist:    (res, k, atoms_i=5, atoms_j=5)
    #    rbf:     (res, k, atoms_i=5, atoms_j=5, num_rbf=16)
    #    output:  (res, k, 5*5*16=400)  ← rearrange로 flatten
    #
    #  einops rearrange 사용처:
    #    atoms_i → broadcast용 shape 변환
    #    diff 계산의 원자 쌍 축 분리
    #    최종 flatten: 'res k ai aj rbf -> res k (ai aj rbf)'
    #
    #  왜 5×5 모든 원자 쌍을 사용하는가:
    #    Cα-Cα 거리만 쓰면 잔기의 방향성(orientation) 정보 손실.
    #    N, C, O, Cβ를 포함한 5×5=25 쌍의 거리 = backbone geometry의 완전 표현.
    #    같은 Cα-Cα 거리에서도 backbone의 방향성(β-strand vs α-helix)이 다름
    #    → 25 쌍의 거리 분포가 이를 구분.
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
