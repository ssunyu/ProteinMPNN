### STEP3 : Protein MPNN ####

class ProteinMPNN(nn.Module):

  def __init__(self, cfg: Config) -> None: 
    super().__init__()
    self.cfg = cfg 
    h = cfg.hidden_dim
    e_raw = NUM_ATOM_PAIRS * cfg.num_rbf 

    self.edge_proj = nn.Linear(e_raw, h)
    
    # 21 tocken emedding 
    self.seq_emb = nn.Embedding(NUM_TOKENS, h)

    # Encoder 
    enc_layers = []
    in_dim = e_raw
    for _ in range(cfg.n_enc_layers):
      enc_layers.append(EncoderLayer(cfg, edge_in_dim=in_dim))
      in_dim = h
    self.encoder = nn.ModuleList(enc_layers)

    # Decoder 
    self.decoder = nn.ModuleList([
      DecoderLayer(cfg) for _ in range(cfg.n_dec_layers)
    ])

    # Output : hidden -> 20 AA logits
    self.output_proj = nn.Linear(h, NUM_AA)
    # linear layer parameter initialization using Xavier Uniform
    self._init_weights()

  def _init_weights(self) -> None:
    for p in self.parameters():
      if p.dim() > 1:
          nn.init.xavier_uniform_(p) 

          
  def encode(
    self,
    n_coords : Float[Tensor, "n_res 3"],
    ca_coords: Float[Tensor, "n_res 3"],
    c_coords : Float[Tensor, "n_res 3"],
    o_coords : Float[Tensor, "n_res 3"],
  ) -> EncoderOutput:
    n_res = ca_coords.shape[0]
    device = ca_coords.device
    h = self.cfg.hidden_dim

    # knn graph
    _, edge_idx = build_knn_graph(ca_coords, self.cfg.k_neighbors)
    # raw edges
    raw_edge    = compute_edge_features(
      n_coords, ca_coords, c_coords, o_coords, edge_idx, self.cfg
    )

    node_h = torch.zeros(n_res, h, device=device)
    edge_h = self.edge_proj(raw_edge)
    
    for layer in self.encoder: 
      node_h, edge_h = layer(node_h, edge_h, edge_idx) 

    return EncoderOutput(node_h=node_h, edge_h=edge_h, edge_idx=edge_idx)
  
  def decode(
    self, 
    enc_out   : EncoderOutput,
    partial_seq : Int[Tensor, "n_res"] | None = None,
    decode_order : Int[Tensor, "n_res"] | None = None, 
    temperature : float = 0.1,
  ) -> DesignOutput:
    n_res = enc_out.node_h.shape[0]
    device = enc_out.node_h.device
    
    # 왜 decoding order를 random permutaion 하는가? 
    if decode_order is None:
      decode_order = torch.randperm(n_res, device=device)

    seq = torch.full((n_res,), MASK_TOKEN, dtype=torch.long, device=device)
    if partial_seq is not None: 
      fixed = partial_seq != MASK_TOKEN
      seq[fixed] = partial_seq[fixed]

    logits_buf    = torch.zeros(n_res, NUM_AA, device=device)
    log_prob_sum   = torch.tensor(0.0, device=device)
    n_designed    = 0 

    for step in range(n_res):
      pos_i = decode_order[step].item()

      if partial_seq is not None and partial_seq[pos_i] != MASK_TOKEN:
        continue
      
      # decoded position set
      decoded_set = set(decode_order[:step].tolist())
      if partial_seq is not None: 
        decoded_set |= set(
          torch.where(partial_seq != MASK_TOKEN)[0].tolist()
        )

      k       = enc_out.edge_idx.shape[1]
      ar_mask = torch.zeros(n_res, k, device=device) 
      for ni, nj in enumerate(enc_out.edge_idx[pos_i].tolist()):
        if nj in decoded_set:
          ar_mask[pos_i, ni] = 1.0
      
      seq_emb = self.seq_emb(seq)
      node_h  = enc_out.node_h.clone()

      for layer in self.decoder:
        node_h  = layer(
          node_h, enc_out.edge_h, enc_out.edge_idx,
          seq_emb, ar_mask
        )
      
      logit_i           = self.output_proj(node_h[pos_i])
      logits_buf[pos_i] = logit_i
      prob_i            = F.softmax(logit_i / temperature, dim=-1)
      aa_i              = torch.multinomial(prob_i, 1).squeeze(-1)

      seq[pos_i] = aa_i
      log_prob_sum += torch.log(prob_i[aa_i] + 1e-8)
      n_designed += 1
      
    mean_lp = log_prob_sum / max(n_designed, 1)
    return DesignOutput(logits=logits_buf, sequence=seq, log_prob=mean_lp)

    
  def forward(
    self, 
    n_coords : Float[Tensor, "n_res 3"],
    ca_coords : Float[Tensor, "n_res 3"],
    c_coords : Float[Tensor, "n_res 3"],
    o_coords : Float[Tensor, "n_res 3"],
    sequence : Int[Tensor, "n_res"],
  ) -> Float[Tensor, "n_res 20"]:
    
    enc_out = self.encode(n_coords, ca_coords, c_coords, o_coords)
    seq_emb = self.seq_emb(sequence)
    n_res   = enc_out.node_h.shape[0]
    k       = enc_out.edge_idx.shape[1]

    # Teacher-forcing 
    ar_mask = torch.ones(n_res, k, device=ca_coords.device)

    node_h = enc_out.node_h
    for layer in self.decoder:
      node_h = layer(
        node_h, enc_out.edge_h,
        enc_out.edge_idx, seq_emb, ar_mask
      )

    return self.output_proj(node_h)