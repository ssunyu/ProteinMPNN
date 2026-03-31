### STEP 5: Smoke Test ### 
if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 65)
    print("  ProteinMPNN Core Architecture — Smoke Test")
    print("  논문: Dauparas et al., Science 2022")
    print("=" * 65)

    cfg = Config()
    N   = 30                         # 작은 단백질로 빠른 테스트
    K   = min(cfg.k_neighbors, N-1)  # 안전 클리핑
    E   = NUM_ATOM_PAIRS * cfg.num_rbf   # 400

    # 합성 backbone: 실제 단백질의 N-Cα-C 기하학을 모사
    ca = torch.randn(N, 3) * 5.0                    # Cα 좌표 
    na = ca + torch.randn(N, 3)
    na = ca + (na - ca) / (na - ca).norm(dim=-1, keepdim=True) * 1.46
    cc = ca + torch.randn(N, 3)
    cc = ca + (cc - ca) / (cc - ca).norm(dim=-1, keepdim=True) * 1.52
    oc = ca + torch.randn(N, 3) * 0.4
    seq = torch.randint(0, NUM_AA, (N,))

    # ── Step 1: Virtual Cβ ─────────────────────────────────────
    cb = compute_virtual_cb(na, ca, cc)
    assert cb.shape == (N, 3)
    print(f"✅ [Step 1] Virtual Cβ shape: {tuple(cb.shape)}")

    # ── Step 2: RBF 인코딩 ────────────────────────────────────
    dummy_dist = torch.tensor([2.0, 5.3, 10.0, 22.0])
    rbf_out    = compute_rbf(dummy_dist, cfg.rbf_d_min, cfg.rbf_d_max, cfg.num_rbf)
    assert rbf_out.shape == (4, cfg.num_rbf)
    print(f"✅ [Step 2] RBF 인코딩 shape: {tuple(rbf_out.shape)}")

    # ── Step 3: k-NN 그래프 ───────────────────────────────────
    dist_knn, idx_knn = build_knn_graph(ca, K)
    assert idx_knn.shape == (N, K)
    print(f"✅ [Step 3] k-NN 그래프 idx shape: {tuple(idx_knn.shape)}")

    # ── Step 4: Edge Features ─────────────────────────────────
    edge_raw = compute_edge_features(na, ca, cc, oc, idx_knn, cfg)
    assert edge_raw.shape == (N, K, E)
    print(f"✅ [Step 4] Edge Features shape: {tuple(edge_raw.shape)}")

    # ── Step 5: Encoder ───────────────────────────────────────
    model = ProteinMPNN(cfg)
    enc   = model.encode(na, ca, cc, oc)
    assert enc.node_h.shape   == (N, cfg.hidden_dim)
    print(f"✅ [Step 5] Encoder node_h shape: {tuple(enc.node_h.shape)}")

    # ── Step 6: Forward (학습) ────────────────────────────────
    logits = model(na, ca, cc, oc, seq)
    assert logits.shape == (N, NUM_AA)
    print(f"✅ [Step 6] Forward logits shape: {tuple(logits.shape)}")

    # ── Step 7: Loss & Gradient ───────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_val  = sequence_nll_loss(logits, seq)
    loss_val.backward()
    print(f"✅ [Step 7] Loss & Gradient 흐름 확인")

    # ── Step 8: Decode (AR 추론) ──────────────────────────────
    optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        out = model.decode(enc, temperature=0.1)
    assert out.sequence.shape == (N,)
    print(f"✅ [Step 8] Decode (AR 추론) 완료")

    print("\n" + "=" * 65)
    print("  전체 smoke test 통과!")
    print("=" * 65)

