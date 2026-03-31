##### STEP4: loss & trainig (추론과 학습의 분리) #####
def sequence_nll_loss(
  logits    : Float[Tensor, "n_res 20"],
  targets   : Int[Tensor, "n_res"],
  mask      : Bool[Tensor, "n_res"] | None = None,
  label_smooth : float = 0.1,
) -> Float[Tensor, ""]:
  log_probs = F.log_softmax(logits, dim=-1)
  nll       = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
  smooth    = log_probs.mean(dim=-1)
  loss      = (1.0 - label_smooth) * nll - label_smooth * smooth
  if mask is not None: 
    loss = loss * mask.float()

  return loss.sum() / 2000.0

def sequence_recovery(
  logits : Float[Tensor, "n_res 20"],
  targets : Int[Tensor, "n_res"],
  mask : Bool[Tensor, "n_res"] | None = None,
) -> float: 
  pred = logits.argmax(dim=-1)
  correct = pred == targets.float()
  if mask is not None:
    return (correct * mask.float()).sum().item() / (mask.float().sum().item() + 1e-8)
  return correct.float().mean().item()
    

def training_step(
  model     : ProteinMPNN,
  optimizer : torch.optim.Optimizer,
  batch     : dict,
) -> dict[str, float]:

  optimizer.zero_grad()

  n_c     = batch["n_coords"]
  ca_c    = batch["ca_coords"]
  c_c     = batch["c_coords"]
  o_c     = batch["o_coords"]
  seq     = batch["sequence"]
  mask    = batch.get("mask", None)

  noise_std = batch.get("noise_std", 0.02)
  if noise_std > 0:
    noise = torch.randn_like(ca_c) * noise_std
    n_c, ca_c, c_c, o_c = n_c + noise, ca_c + noise, c_c + noise, o_c + noise

  logits  = model(n_c, ca_c, c_c, o_c, seq)
  loss    = sequence_nll_loss(logits, seq, mask, model.cfg.label_smooth)

  loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()

  return {
    "loss" : loss.item(),
    "recovery" : sequence_recovery(logits, seq, mask),
  }



