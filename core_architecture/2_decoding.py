####### STEP2 : DECODING ######## 

class DecoderLayer(nn.Module):
    def __init__(self, cfg : Config) -> None: 
        super().__init__()
        h = cfg.hidden_dim
        dec_edge_dim = h + h

        def _mlp(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, h), nn.GELU(),
                nn.Linear(h, h), nn.GELU(),
                nn.Linear(h, h),
            )

        self.msg_mlp = _mlp(h + h + dec_edge_dim)
        self.ff_node = nn.Sequential(
            nn.Linear(h, h * 4), nn.GELU(),
            nn.Linear(h * 4, h),
        )
        self.norm1  = nn.LayerNorm(h)
        self.norm2  = nn.LayerNorm(h)
        self.drop   = nn.Dropout(cfg.dropout)
        self.h      = h

    def forward(
        self,
        node_h      : Float[Tensor, "n_res hidden"],
        edge_h      : Float[Tensor, "n_res k hidden"],
        edge_idx    : Int[Tensor, "n_res k"],
        seq_emb     : Float[Tensor, "n_res hidden"],
        ar_mask     : Float[Tensor, "n_res k"], 
    ) -> Float[Tensor, "n_res hidden"]:
        n_res, k = edge_idx.shape

        seq_j = seq_emb[edge_idx.reshape(-1)].reshape(n_res, k, self.h)
        seq_j = seq_j * ar_mask.unsqueeze(-1)

        edge_dec = torch.cat([edge_h, seq_j], dim=-1)

        h_j = node_h[edge_idx.reshape(-1)].reshape(n_res, k, self.h)
        h_i = node_h.unsqueeze(1).expand(-1, k, -1)

        msg = self.msg_mlp(torch.cat([h_i, h_j, edge_dec], dim=-1))
        agg = msg.sum(dim=1)
        node_h = self.norm1(node_h + self.drop(agg))

        # Residual connection의 철학 : 표현 학습에 입력은 보존하고 나머지를 학습 
        node_h = self.norm2(node_h + self.drop(self.ff_node(node_h)))

        return node_h