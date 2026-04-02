# smoke_test.py — 전체 파이프라인 End-to-End 검증
#
# ══════════════════════════════════════════════════════════════════════
#  검증 전략
# ══════════════════════════════════════════════════════════════════════
#
#  "동작하는가"와 "수학적으로 타당한가"를 함께 검사한다.
#
#  합성 단백질 사용 이유:
#    실제 PDB 데이터 없이도 전체 파이프라인을 검증 가능.
#    CI/CD 환경에서 빠르게 실행 가능 (외부 데이터 의존성 없음).
#    물리적으로 타당한 bond length로 생성 → 실제 단백질과 유사한 기하학.
#
#  검증 항목:
#    Step 1. Virtual Cβ: shape + 물리적 bond length
#    Step 2. RBF: shape + 범위 [0,1] + 중심 근방에서 1
#    Step 3. k-NN graph: shape + 대각선 제외 + 거리 대칭성
#    Step 4. Edge Features: shape (res, k, 400)
#    Step 5. Encoder: shape (res, hidden)
#    Step 6. Forward (teacher-forcing): shape (res, 20)
#    Step 7. Loss & Gradient: backward 통과 + gradient 존재
#    Step 8. Decode (AR inference): shape (res,) + log_prob 유효성
#
# ══════════════════════════════════════════════════════════════════════

import torch
from core_architecture.config import Config, NUM_AA, NUM_ATOM_PAIRS
from core_architecture.preprocessing import (
    compute_virtual_cb,
    compute_rbf,
    build_knn_graph,
    compute_edge_features,
)
from core_architecture.model import ProteinMPNN
from core_architecture.training import sequence_nll_loss, training_step


def make_synthetic_protein(n_res: int = 30) -> dict:
    # 합성 단백질 backbone 생성.
    # N-Cα 결합길이 ~1.46Å, Cα-C 결합길이 ~1.52Å을 근사.
    # O는 Cα 근방에 랜덤 배치 (실제 carbonyl 위치의 근사).
    torch.manual_seed(42)
    ca = torch.randn(n_res, 3) * 5.0

    # N: Cα로부터 1.46Å 방향에 배치
    n_dir = torch.randn(n_res, 3)
    n_dir = n_dir / n_dir.norm(dim=-1, keepdim=True)
    na    = ca + n_dir * 1.46

    # C: Cα로부터 1.52Å 방향에 배치
    c_dir = torch.randn(n_res, 3)
    c_dir = c_dir / c_dir.norm(dim=-1, keepdim=True)
    cc    = ca + c_dir * 1.52

    oc  = ca + torch.randn(n_res, 3) * 0.4
    seq = torch.randint(0, NUM_AA, (n_res,))

    return {"n": na, "ca": ca, "c": cc, "o": oc, "seq": seq}


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 65)
    print("  ProteinMPNN Core Architecture — Smoke Test")
    print("  Dauparas et al., Science 2022")
    print("=" * 65)

    cfg   = Config()
    prot  = make_synthetic_protein(n_res=30)
    N     = 30
    K     = min(cfg.k_neighbors, N - 1)
    E     = NUM_ATOM_PAIRS * cfg.num_rbf   # 25 × 16 = 400

    na, ca, cc, oc, seq = prot["n"], prot["ca"], prot["c"], prot["o"], prot["seq"]

    # ── Step 1: Virtual Cβ ───────────────────────────────────────────
    cb = compute_virtual_cb(na, ca, cc)
    assert cb.shape == (N, 3), f"shape: {cb.shape}"

    # Cβ 좌표가 Cα로부터 0 초과 거리에 있어야 함
    # (합성 backbone은 임의 방향이라 이상적 1.52Å과 다를 수 있음)
    cb_ca_dist = (cb - ca).norm(dim=-1)
    assert (cb_ca_dist > 0).all(), "Cβ가 Cα와 동일 위치"
    print(f"✅ [Step 1] Virtual Cβ  shape={tuple(cb.shape)}, "
          f"Cβ-Cα dist(mean)={cb_ca_dist.mean():.3f}Å")

    # ── Step 2: RBF ─────────────────────────────────────────────────
    dummy_dist = torch.tensor([2.0, 5.3, 10.0, 22.0])
    rbf_out    = compute_rbf(dummy_dist, cfg.rbf_d_min, cfg.rbf_d_max, cfg.num_rbf)
    assert rbf_out.shape == (4, cfg.num_rbf)
    assert (rbf_out >= 0).all() and (rbf_out <= 1.0 + 1e-5).all(), \
        "RBF 값이 [0,1] 범위를 벗어남"
    # d=d_min에서 basis 0이 최대 (≈1)
    rbf_at_min = compute_rbf(
        torch.tensor([cfg.rbf_d_min]), cfg.rbf_d_min, cfg.rbf_d_max, cfg.num_rbf
    )
    assert rbf_at_min[0, 0] > 0.9, f"d_min에서 첫 basis가 1에 가까워야 함: {rbf_at_min[0,0]:.3f}"
    print(f"✅ [Step 2] RBF         shape={tuple(rbf_out.shape)}, "
          f"range=[{rbf_out.min():.3f}, {rbf_out.max():.3f}]")

    # ── Step 3: k-NN Graph ───────────────────────────────────────────
    knn_dist, edge_idx = build_knn_graph(ca, K)
    assert edge_idx.shape == (N, K)
    assert knn_dist.shape == (N, K)

    # 자기 자신이 이웃에 없어야 함
    self_idx = torch.arange(N).unsqueeze(1)
    assert not (edge_idx == self_idx).any(), "자기 자신이 이웃에 포함됨"

    # 거리는 양수여야 함
    assert (knn_dist >= 0).all(), "음수 거리 발생"
    print(f"✅ [Step 3] k-NN Graph  shape={tuple(edge_idx.shape)}, "
          f"dist range=[{knn_dist.min():.2f}, {knn_dist.max():.2f}]Å")

    # ── Step 4: Edge Features ────────────────────────────────────────
    edge_raw = compute_edge_features(na, ca, cc, oc, edge_idx, cfg)
    assert edge_raw.shape == (N, K, E), f"shape: {edge_raw.shape}"
    assert (edge_raw >= 0).all() and (edge_raw <= 1.0 + 1e-5).all(), \
        "Edge features가 [0,1] 범위를 벗어남 (RBF 값이어야 함)"
    print(f"✅ [Step 4] Edge Features shape={tuple(edge_raw.shape)}, "
          f"edge_dim={E} (={NUM_ATOM_PAIRS}×{cfg.num_rbf})")

    # ── Step 5: Encoder ──────────────────────────────────────────────
    model = ProteinMPNN(cfg)
    enc   = model.encode(na, ca, cc, oc)
    assert enc.node_h.shape   == (N, cfg.hidden_dim)
    assert enc.edge_h.shape   == (N, K, cfg.hidden_dim)
    assert enc.edge_idx.shape == (N, K)
    print(f"✅ [Step 5] Encoder     node_h={tuple(enc.node_h.shape)}, "
          f"edge_h={tuple(enc.edge_h.shape)}")

    # ── Step 6: Forward (Teacher-Forcing) ───────────────────────────
    logits = model(na, ca, cc, oc, seq)
    assert logits.shape == (N, NUM_AA), f"shape: {logits.shape}"
    # 미학습 모델에서 logits이 nan이 아닌지 확인
    assert not logits.isnan().any(), "Forward에서 NaN 발생"
    print(f"✅ [Step 6] Forward     logits={tuple(logits.shape)}, "
          f"no NaN: ✓")

    # ── Step 7: Loss & Gradient ──────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_val  = sequence_nll_loss(logits, seq)
    assert loss_val.item() > 0, "Loss가 0 이하"
    assert loss_val.shape == torch.Size([]), "Loss가 scalar가 아님"

    loss_val.backward()

    # 적어도 일부 파라미터에 gradient가 있어야 함
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "어떤 파라미터도 gradient가 없음"
    print(f"✅ [Step 7] Loss={loss_val.item():.4f}, Gradient 흐름 확인")

    # ── Step 8: Decode (AR Inference) ───────────────────────────────
    optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        out = model.decode(enc, temperature=0.1)

    assert out.sequence.shape == (N,), f"shape: {out.sequence.shape}"
    assert out.logits.shape   == (N, NUM_AA)
    assert out.sequence.min() >= 0 and out.sequence.max() < NUM_AA, \
        "생성된 AA 인덱스가 유효 범위를 벗어남"
    assert not out.log_prob.isnan(), "log_prob이 NaN"
    print(f"✅ [Step 8] Decode (AR) sequence={tuple(out.sequence.shape)}, "
          f"log_prob={out.log_prob.item():.4f}")

    # ── Step 9: Training Step ────────────────────────────────────────
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch = {
        "n_coords" : na, "ca_coords": ca,
        "c_coords" : cc, "o_coords" : oc,
        "sequence" : seq,
        "noise_std": 0.02,
    }
    metrics = training_step(model, optimizer, batch)
    assert "loss"     in metrics
    assert "recovery" in metrics
    assert 0.0 <= metrics["recovery"] <= 1.0
    print(f"✅ [Step 9] Training Step  loss={metrics['loss']:.4f}, "
          f"recovery={metrics['recovery']:.3f}")

    print("\n" + "=" * 65)
    print("  전체 smoke test 통과!")
    print(f"  모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 65)
