##### STEP 0 : PREPROCESSING ##### 

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Bool, Float, Int

@dataclass
class Config:
    hidden_dim      : int = 128
    k_neighbors     : int = 48
    n_enc_layers    : int = 3
    n_dec_layers    : int = 3
    dropout         : float = 0.1
    num_rbf         : int = 16
    rbf_d_min       : float = 2.0
    rbf_d_max       : float = 22.0
    label_smooth    : float = 0.1 

NUM_AA              = 20 
MASK_TOKEN          = 20 
NUM_TOKENS          = 21
NUM_ATOMS           = 5
NUM_ATOM_PAIRS      = NUM_ATOMS ** 2
AA_ALPHABET         = "ACDEFGHIKLMNPQRSTVWY"
assert len(AA_ALPHABET) == NUM_AA

class EncoderOutput(NamedTuple): 
    node_h      : Float[Tensor, "n_res hidden"]
    edge_h      : Float[Tensor, "n_res k hidden"]
    edge_idx    : Int[Tensor, "n_res k"] 
    
class DesignOutput(NamedTuple):
    logits      : Float[Tensor, "n_res 20"]
    sequence    : Int[Tensor, "n_res"]
    log_prob    : Float[Tensor, ""]

def compute_virtual_cb(
    n_coords    : Float[Tensor, "n_res 3"],
    ca_coords   : Float[Tensor, "n_res 3"],
    c_coords    : Float[Tensor, "n_res 3"],
) -> Float[Tensor, "n_res 3"]:

    b = ca_coords - n_coords
    c = c_coords - ca_coords
    a = torch.cross(b,c, dim=-1) # tensor 형태로 cross-product 한번에 

    return (
        -0.58273431 * a
            + 0.56802827 * b
            - 0.54067466 * c
            + ca_coords
    ) # tetrahedral 각도 기준으로 미리 계산해놓은 거리 


def compute_rbf( # 거리라는 물리적 1d축을 따라 모든 데이터에 반복 적용-> broadcast function으로 design 
    dist    : Float[Tensor, "..."], # why Tensor?? - parallel processing을 위해서
    d_min   : float,
    d_max   : float,
    n_rbf   : int, 
) -> Float[Tensor, "... n_rbf"]:
    
    centers = torch.linspace(d_min, d_max, n_rbf, device=dist.device)  # 새로운 숫자 텐서 창조, 모델이 관리하지 않는 temporary 텐서일 때 device 명시 필요 
    gamma   = (n_rbf - 1) / (d_max - d_min + 1e-8)

    return torch.exp(-gamma * (dist.unsqueeze(-1) - centers ** 2)) # 텐서 broadcasting 잘 활용하기 (brocasting할 차원을 unsqueeze로 만듦)

def build_knn_graph(
    ca_coords   : Float[Tensor, "n_res 3"],
    k           : int,
) -> tuple[Float[Tensor, "n_res k"], Int[Tensor, "n_res k"]]:
    
    n_res   = ca_coords.shape[0]
    k       = min(k, n_res - 1)
    # broadcasting -> 복제 방향(행, 열 시각화 )
    diff = ca_coords.unsqueeze(0) - ca_coords.unsqueeze(1)
    dist2 = (diff ** 2).sum(-1)
    dist2.fill_diagonal_(float("inf"))
    
    _, edge_idx = dist2.topk(k, dim=-1, largest=False) 
    dist = dist2.gather(1, edge_idx).clamp(min=0).sqrt() #부동소수점 오차 
    return dist, edge_idx

def compute_edge_features(
    n_coords    : Float[Tensor, "n_res 3"],
    ca_coords   : Float[Tensor, "n_res 3"],
    c_coords    : Float[Tensor, "n_res 3"],
    o_coords    : Float[Tensor, "n_res 3"],
    idx_knn     : Int[Tensor, "n_res k"], 
    cfg         : Config, # dataclass 
) -> Float[Tensor, "n_res k edge_dim"]:
    
    n_res, k    = idx_knn.shape
    n_rbf       = cfg.num_rbf

    cb_coords   = compute_virtual_cb(n_coords, ca_coords, c_coords)
    
    atoms_i = torch.stack(
        [n_coords, ca_coords, c_coords, o_coords, cb_coords], dim=1
    )
    # pytorch indexing 참고 
    
    atoms_j = atoms_i[idx_knn.reshape(-1)].reshape(n_res, k, NUM_ATOMS, 3) 
    atoms_i = atoms_i.unsqueeze(1).expand(-1, k, -1, -1)

    diff = atoms_i.unsqueeze(3) - atoms_j.unsqueeze(2)
    dist = diff.norm(dim=-1)

    rbf = compute_rbf(dist, cfg.rbf_d_min, cfg.rbf_d_max, n_rbf)
    return rbf.reshape(n_res, k, NUM_ATOM_PAIRS * n_rbf)
