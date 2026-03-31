##### STEP 1 : ENCODING ##### 

class EncoderLayer(nn.Module):
  def __init__(self, cfg: Config, edge_in_dim : int ) -> None: 
    super().__init__()
    h = cfg.hidden_dim

    def _mlp(in_dim: int) -> nn.Sequential:
      return nn.Sequential(
        nn.Linear(in_dim, h), nn.GELU(),
        nn.Linear(h, h), nn.GELU(),
        nn.Linear(h, h),
      )
    
    self.msg_mlp = _mlp(h + h + edge_in_dim)
    self.ff_node = nn.Sequential(
      nn.Linear(h, h * 4), nn.GELU(),
      nn.Linear(h * 4, h),
    )
    self.edge_mlp = _mlp(h + h + edge_in_dim)

    self.norm1  = nn.LayerNorm(h)
    self.norm2  = nn.LayerNorm(h)
    self.norm_e = nn.LayerNorm(h)
    self.drop   = nn.Dropout(cfg.dropout)
    self.h      = h 
    self.edge_out_proj = nn.Linear(edge_in_dim, h) if edge_in_dim != h else nn.Identity()

  def forward(
    self, 
    node_h  : Float[Tensor, "n_res hidden"],
    edge_h  : Float[Tensor, "n_res k hidden"],
    edge_idx : Int[Tensor, "n_res k"],
  ) -> tuple[Float[Tensor, "n_res hidden"], Float[Tensor, "n_res k hidden"]]:
    n_res, k, _ = edge_h.shape
    h_j = node_h[edge_idx.reshape(-1)].reshape(n_res, k, self.h)
    h_i = node_h.unsqueeze(1).expand(-1, k, -1)

    msg = self.msg_mlp(torch.cat([h_i, h_j, edge_h], dim=-1))
    agg = msg.sum(dim=1)
    node_h = self.norm1(node_h + self.drop(agg))
    node_h = self.norm2(node_h + self.drop(self.ff_node(node_h)))

    edge_h_proj = self.edge_out_proj(edge_h)
    h_j_new     = node_h[edge_idx.reshape(-1)].reshape(n_res, k, self.h)
    h_i_new     = node_h.unsqueeze(1).expand(-1, k, -1)
    d_edge      = self.edge_mlp(torch.cat([h_i_new, h_j_new, edge_h], dim=-1))
    edge_h_new  = self.norm_e(edge_h_proj + self.drop(d_edge))

    return node_h, edge_h_new 