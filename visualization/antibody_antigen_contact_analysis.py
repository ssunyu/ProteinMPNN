"""
항체–항원 복합체에서 residue 간 구조적 관계를 Cα–Cα 거리 행렬과 threshold 기반 contact map으로 계산하고 시각화.

목적:
    1) 각 chain (Ab-H, Ab-L, Ag) 내부의 구조적 패턴을 확인
    2) 항체와 항원 사이의 상호작용이 어떻게 나타나는지 관찰
    3) long-range interaction이 얼마나 sparse하고 국소적으로 나타나는지 파악

CORE:
    1) contact map은 학습된 representation이 아니라 단순 거리 threshold로부터 생성
    2) chain 간 interaction은 매우 제한적이고 국소적으로만 드러남
    3) 이는 단순한 구조 표현 방식이 long-range 혹은 고차 상호작용을 충분히 담지 못한다는 한계 검증 

DEPENDENCY: 코드를 돌리기 위해선 ProteinMPNN(https://github.com/dauparas/ProteinMPNN.git)의 Model weight이 필요합니다.
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

# ── paths ──────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "ProteinMPNN"))
sys.path.insert(0, BASE)

from protein_mpnn_utils import ProteinMPNN
from fmri_lens_proteinmpnn import (
    parse_pdb_backbone, featurize, load_model, encode_with_intermediates
)

PDB_PATH     = os.path.join(BASE, "inputs", "1mlc.pdb")
WEIGHTS_PATH = os.path.join(BASE, "ProteinMPNN", "vanilla_model_weights", "v_48_020.pt")
OUT_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "antibody_antigen_contact_analysis.png")

TARGET_CHAINS  = ["A", "B", "E"]
CHAIN_INT_IDS  = [0, 1, 2]   # parse_pdb_backbone encodes chains as integers
CHAIN_LABELS   = ["Ab-H (A)", "Ab-L (B)", "Ag (E)"]

# ── global style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "axes.edgecolor":    "#555555",
    "axes.labelcolor":   "#222222",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ── colourmap: dark-amber (close) → white (far) ─────────────────────────────
_dist_cmap = LinearSegmentedColormap.from_list(
    "dist", ["#7B2008", "#D4681E", "#F5C77E", "#FFFFFF"], N=256
)

# ── helper: chain boundary positions & labels ────────────────────────────────
def chain_layout(chain_ids):
    """Return (boundaries, midpoints, labels) for axis decoration.
    chain_ids are integers 0,1,2 corresponding to TARGET_CHAINS A,B,E."""
    ci = np.array(chain_ids)
    chains = []
    for int_id, label in zip(CHAIN_INT_IDS, CHAIN_LABELS):
        idx = np.where(ci == int_id)[0]
        if len(idx):
            chains.append((int_id, label, int(idx[0]), int(idx[-1])))
    boundaries = [c[3] + 0.5 for c in chains[:-1]]
    midpoints  = [int((c[2] + c[3]) / 2) for c in chains]
    labels     = [c[1] for c in chains]
    return boundaries, midpoints, labels

# ── interface rectangle helper ────────────────────────────────────────────────
def add_interface_rect(ax, chain_ids, row_int_id, col_int_id,
                        color="#00FF00", lw=2.5):
    """Draw a rectangle around the (row × col) chain block.
    row_int_id / col_int_id are integer chain IDs (0,1,2)."""
    ci = np.array(chain_ids)
    r_idx = np.where(ci == row_int_id)[0]
    c_idx = np.where(ci == col_int_id)[0]
    if not len(r_idx) or not len(c_idx):
        return
    r0, r1 = r_idx[0] - 0.5, r_idx[-1] + 0.5
    c0, c1 = c_idx[0] - 0.5, c_idx[-1] + 0.5
    rect = patches.Rectangle(
        (c0, r0), c1 - c0, r1 - r0,
        linewidth=lw, edgecolor=color, facecolor="none", zorder=10,
        clip_on=False
    )
    ax.add_patch(rect)

# ── axis decoration (no data labels inside) ──────────────────────────────────
def decorate_axes(ax, chain_ids, xlabel, ylabel):
    bounds, mids, labels = chain_layout(chain_ids)
    L = len(chain_ids)

    # chain boundary lines — thin, light gray
    for b in bounds:
        ax.axhline(b, color="#AAAAAA", lw=0.8, zorder=4)
        ax.axvline(b, color="#AAAAAA", lw=0.8, zorder=4)

    ax.set_xticks(mids)
    ax.set_xticklabels(labels, fontsize=8, rotation=0)
    ax.set_yticks(mids)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(L - 0.5, -0.5)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=12) 
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=12)
    ax.tick_params(length=0, pad=8)


# ════════════════════════════════════════════════════════════════════════════
def main():
    # ── load & encode ────────────────────────────────────────────────────────
    print("Loading 1MLC …")
    xyz, seq, chain_ids, res_idx = parse_pdb_backbone(PDB_PATH, TARGET_CHAINS)
    feats = featurize(xyz, chain_ids, res_idx)

    model = load_model(WEIGHTS_PATH)
    k = model.features.top_k          # 48
    print(f"k = {k},  L = {len(chain_ids)}")

    with torch.no_grad():
        h_E, h_V_list, E_idx, D_full = encode_with_intermediates(model, feats)

    L = D_full.shape[0]
    ci = np.array(chain_ids)

    # ── build binary k-NN matrix ──────────────────────────────────────────────
    E_np = E_idx.cpu().numpy()[0]          # [L, k]
    knn_mat = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for nb in E_np[i]:
            if 0 <= nb < L:
                knn_mat[i, nb] = 1.0
                knn_mat[nb, i] = 1.0

    # ── distance matrix capped at 60 Å for display ───────────────────────────
    D_disp = np.clip(D_full, 0, 60).astype(np.float32)

    
    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.6, 3.8), dpi=300)

    from matplotlib.gridspec import GridSpec
  
    gs = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[1, 0.055, 0.15, 1], 
        wspace=0.08,                      
        left=0.11, right=0.95,
        top=0.93,  bottom=0.14,
    )
    
    ax_a  = fig.add_subplot(gs[0, 0])
    cax_a = fig.add_subplot(gs[0, 1]) 
    ax_b  = fig.add_subplot(gs[0, 3]) 

    # ── panel (a): distance matrix ────────────────────────────────────────────
    im_a = ax_a.imshow(
        D_disp,
        cmap=_dist_cmap,
        vmin=0, vmax=60,
        aspect="auto", interpolation="nearest", origin="upper"
    )
    decorate_axes(ax_a,
                  chain_ids,
                  xlabel="Residue",
                  ylabel="Residue")

    # interface rectangles: Ab(0,1) ↔ Ag(2) blocks, integer chain IDs
    for r, c in [(0,2), (2,0), (1,2), (2,1)]:
        add_interface_rect(ax_a, chain_ids, r, c)

    cb_a = fig.colorbar(im_a, cax=cax_a)
    cb_a.set_label("Cα – Cα distance (Å)", fontsize=7, labelpad=4)
    cb_a.set_ticks([0, 20, 40, 60])
    cb_a.ax.tick_params(labelsize=6)

    # ── panel (b): k-NN adjacency matrix ─────────────────────────────────────
    im_b = ax_b.imshow(
        knn_mat,
        cmap="binary",          # white = absent, black = edge present
        vmin=0, vmax=1,
        aspect="auto", interpolation="nearest", origin="upper"
    )
    decorate_axes(ax_b,
                  chain_ids,
                  xlabel="Residue",
                  ylabel=None)
    ax_b.set_yticklabels([])   # shared y-axis with (a), suppress duplicate labels

    for r, c in [(0,2), (2,0), (1,2), (2,1)]:
        add_interface_rect(ax_b, chain_ids, r, c)

    # ── save ─────────────────────────────────────────────────────────────────
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()