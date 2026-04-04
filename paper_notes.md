# Paper Notes — Antibody Design with Structure Prediction

Reading notes on the five reference papers from the Galux job posting.
The frame: each paper lives in the same inference problem but makes different choices
about which variables to condition on, which to marginalize, and which uncertainty to collapse.

---

## Shared Problem Structure

All five papers are approximating the same integral:

```
P(Y | X) = ∫ P(Y | Z) · P(Z | X) dZ
```

| Variable | Meaning | Space |
|----------|---------|-------|
| X | antigen surface geometry (epitope context) | R^(3·n_ag) |
| Z | CDR loop conformation (the latent variable) | M_Z ⊂ R^(3·n_cdr) |
| Y | CDR amino acid sequence (design target) | {0..19}^(n_cdr) |

The integral is intractable. Every paper chooses how to handle it differently —
which term to approximate, which to point-estimate, which to model as a distribution.
That choice is the essential difference between them.

---

## 1. Hummer et al. 2022 — Locating the Bottleneck

*Current Opinion in Structural Biology*

### 1. 목적

Map which term in P(Y|X) = ∫P(Y|Z)·P(Z|X)dZ is the actual bottleneck.
Not to solve the problem — to define where the problem lives.

### 2. 모델링 설계

**데이터 단위**
Cα coordinates of CDR loops + epitope surface residues.
The contact between paratope and epitope is the signal; everything else is context.

**표현 공간**
CDR loop conformations live in R^(3·n_loop), but physical constraints
(bond lengths, angles, steric clashes) compress the realizable set to a manifold M_Z.
CDR-H3 is the least constrained loop — its manifold has the highest intrinsic dimensionality
and is the least well-characterized.

**변동 소스**
- X-side: epitope surface is relatively rigid (framework of the antigen)
- Z-side: CDR-H3 conformation is the dominant fluctuation source —
  its distribution P(Z|X) is wide and multimodal
- Y-side: sequence space {0..19}^n is discrete;
  most sequences either misfold or fail to bind

**연결 구조**
The review identifies three sequentially dependent steps:

```
X → [structure prediction] → Z* → [sequence design] → Y*
```

Each arrow is a conditional inference. Error in the first arrow
propagates into the second — if Z* is wrong, Y* is designed against the wrong geometry.

### 3. 수식과 기하학

The covariance structure of P(Z | X) for CDR-H3 is elongated —
high variance in the directions tangent to the antigen surface,
low variance in the normal direction (steric constraint from the antigen face).

```
Σ_{Z|X} = U · Λ · U^T

where Λ = diag(λ_1, λ_2, ..., λ_d),  λ_1 >> λ_2 > ... > λ_d

dominant eigenvectors u_1, u_2: tangential displacement along epitope surface
minor eigenvectors:             normal displacement (sterically constrained)
```

The ellipsoid representing P(Z|X) is flat against the antigen surface and wide along it.
Predicting Z* = E[Z|X] (the mean of this ellipsoid) for a flexible loop
loses all the off-center probability mass — which, for CDR-H3, is substantial.

### 4. Message

The bottleneck is not P(Y|Z) — given a good Z, ProteinMPNN-type models work.
The bottleneck is P(Z|X): predicting where CDR-H3 sits relative to the antigen.
The wide, elongated covariance ellipsoid of CDR-H3 | antigen means any point estimate
Z* collapses the integral to a single, possibly off-center sample.
This review sets up the precise question that Bang et al. 2024 answers empirically.

---

## 2. Watson et al. 2023 — Diffusion over the Backbone Manifold

*Nature*

### 1. 목적

Learn P(Z | X) via a generative diffusion model (RFdiffusion),
then factor out P(Y | Z) to ProteinMPNN.
The goal: make backbone generation conditioned on target geometry accessible
without exhaustive physics-based simulation.

### 2. 모델링 설계

**데이터 단위**
Backbone frames: for each residue, a rigid body (rotation R ∈ SO(3), translation t ∈ R^3).
The data unit is not atomic coordinates but the rigid-body transformation
from a canonical reference frame — this encodes geometry without dependence on
absolute position or global orientation.

**표현 공간**
Noisy backbone frame space → denoised manifold M_Z.
RFdiffusion diffuses over (R, t) space, learning to reverse the noise process
conditioned on X. The manifold M_Z is implicit — defined by the support
of the learned score function ∇_Z log p(Z|X).

**변동 소스**
- Forward process: Z_0 → Z_T via Gaussian noise on coordinates + IGSO(3) noise on rotations
- Reverse process: score network s_θ(Z_t, t, X) guides denoising toward X-conditioned structures
- ProteinMPNN receives a single sample z ~ P(Z|X) and computes P(Y|z) independently

**연결 구조**

```
X → RFdiffusion → z ~ P(Z|X) → ProteinMPNN → Y ~ P(Y|z)
                      ↑ no feedback ↑
```

Conditional independence assumption: Y ⊥ X | Z.
ProteinMPNN has no access to X once Z is fixed.
The entire antigen context is encoded in Z via the k-NN graph.

### 3. 수식과 기하학

The full generative model:

```
p_θ(Z | X) = ∫ p(Z_T) · ∏_t p_θ(Z_{t-1} | Z_t, X) dZ_{1:T}

p_θ(Y | X) ≈ p_θ(Y | z*)     z* ~ p_θ(Z | X)   [single sample approximation]
```

This is a Monte Carlo estimate of the integral with n=1 sample.
The variance of this estimator depends on the width of P(Z|X):

```
Var[P(Y|z)] ≈ E_Z[(P(Y|Z) - P(Y|X))^2] · 1/n_samples
```

For n=1, the estimator variance is entirely determined by how strongly P(Y|z)
varies across the support of P(Z|X).

Geometrically: imagine the joint distribution of (Z, Y) as an ellipsoid in product space.
The conditional independence Y ⊥ X | Z means this ellipsoid is axis-aligned
with the Z axis — the Y-Z covariance is captured, but the residual Y-X covariance
(after removing the Z effect) is assumed zero.

Whether that assumption holds depends on how precisely z captures the relevant X structure.

### 4. Message

The factorization P(Y|X) ≈ P(Y|z*)·P(z*|X) introduces a point-estimate bias.
The two models never communicate — RFdiffusion doesn't know what ProteinMPNN needs;
ProteinMPNN doesn't know how uncertain RFdiffusion was.
The quality of Y* depends entirely on how well z* represents the antigen-contact geometry
that ProteinMPNN will build its k-NN graph from.

---

## 3. Abramson et al. 2024 — Structure Prediction as Distribution

*Nature*

### 1. 목적

Move from point-estimate structure prediction (AF2: Z* = argmax P(Z|X))
to distributional structure prediction (AF3: model P(Z|X) explicitly).
Joint prediction across all molecular species in a complex — protein, DNA, RNA,
small molecules, ions — without chain-type-specific architectural assumptions.

### 2. 모델링 설계

**데이터 단위**
All-atom coordinates, unified across molecular types via a token representation.
Each residue or small-molecule fragment is a token; the interaction between tokens
is modeled through attention over all pairwise relationships simultaneously.

**표현 공간**
Pairwise representation Z_pair ∈ R^(N×N×c_z): encodes the relationship between
every pair of tokens. Single representation Z_single ∈ R^(N×c_s): encodes per-token state.
Together these define a latent space that is jointly updated by the Evoformer-style
transformer stack.

**변동 소스**
- Sequence → multiple structural conformations (aleatoric uncertainty in Z)
- Model uncertainty about the right structural arrangement (epistemic)
- AF3 separates these via the diffusion module:
  multiple samples from the diffusion process represent different realizations
  of the structural distribution P(Z|X), not just one

**연결 구조**

```
X_seq ──→ Pairwise + Single representation ──→ Diffusion module ──→ {z_1, z_2, ...z_k}
          [Evoformer / Pairformer blocks]         P(Z|X) as distribution
```

Cross-attention over all molecular species: the CDR loop and antigen are
co-represented from the beginning — Z_pair captures their joint geometry.

### 3. 수식과 기하학

AF2 output: a point z* in structure space — a delta function approximation to P(Z|X):

```
P_{AF2}(Z | X) ≈ δ(Z - z*)
```

AF3 output: a distribution — approximated by the diffusion generative model:

```
P_{AF3}(Z | X) ≈ p_θ(Z | X)

Sample z_1, z_2, ..., z_k ~ p_θ(Z|X)

→ Covariance Σ_Z = (1/k) Σ_i (z_i - z̄)(z_i - z̄)^T   [empirical ellipsoid in R^(3n)]
```

For rigid domains, Σ_Z is nearly spherical and small (low variance, well-determined).
For CDR-H3, Σ_Z is elongated — the dominant eigenvectors point along the surface
of the antigen, reflecting the degrees of freedom along which the loop can move
while remaining sterically feasible.

The AF3 design implication: if you condition ProteinMPNN on multiple z_i samples,
you are sampling from the full covariance ellipsoid rather than conditioning
on a single point. Whether the designed sequence Y is robust across this ellipsoid
is the natural next question — and it requires knowing how P(Y|z) varies
as z moves along the principal axes of Σ_Z.

### 4. Message

AF3 makes the distributional nature of Z explicit. The structural prediction is
no longer a single answer — it is an uncertainty ellipsoid in coordinate space.
For the design pipeline, this raises the question that point-estimate approaches suppress:
which z should we design against, and how sensitive is the design to that choice?
The answer depends on the shape of Σ_Z — and for CDR-H3, that shape is the problem.

---

## 4. Bang et al. 2024 — Precision in Z as a Threshold Phenomenon

*Galux Inc.*

### 1. 목적

Demonstrate empirically and mechanistically that CDR loop structure prediction accuracy
is the rate-limiting variable for zero-shot antibody design.
Show that the relationship between RMSD error and design success rate is not gradual
but threshold-like — a phase transition in the feasibility of correct antigen conditioning.

### 2. 모델링 설계

**데이터 단위**
The antigen-contact set: I(z) = { r ∈ antigen residues : r ∈ kNN(CDR-H3 residue, k=48) }

This is not a vector — it is a set. The data unit is discrete and binary per residue pair.
The antigen enters ProteinMPNN's design process only through membership in this set.

**표현 공간**
ProteinMPNN's conditioning space for antigen: {0,1}^n_antigen per CDR residue.
Whether residue r is an antigen neighbor is a binary decision made by the k-NN ball.
The space is not continuous — it is a lattice of inclusion/exclusion decisions.

**변동 소스**
- CDR-H3 position error: δz = z_predicted - z_true (Gaussian-ish, zero-mean)
- Contact-set sensitivity: ∂I/∂z — how many antigen residues sit near the k-NN boundary
- These two sources interact: the design is corrupted when δz is large enough
  to push antigen residues across the k-NN boundary

**연결 구조**

```
X_antigen ──→ [GalaxyAD: predict z*] ──→ k-NN(z*) ──→ I(z*) ──→ P(Y | I(z*))

Key: I(z*) ≠ I(z_true)  when  ||δz|| > boundary distance for any antigen residue r
```

### 3. 수식과 기하학

Define the boundary distance for antigen residue r:

```
b_r(z) = d(CDR-H3, r) - d_48(z)

where d_48(z) = distance to the 48th-nearest neighbor of CDR-H3 at structure z
```

Residue r enters the contact set when b_r(z) < 0.
The sensitivity to structural error:

```
∂[r ∈ I] / ∂z = δ(b_r(z)) · ∂b_r/∂z    [delta function at boundary crossing]
```

This is zero everywhere except at the k-NN boundary — the sensitivity is concentrated
at a lower-dimensional surface in Z-space. CDR-H3, being short and flexible,
sits near many such boundaries simultaneously.

Geometrically: the structure space around z_true contains a web of hyperplanes —
each one the boundary where one antigen residue enters or exits the contact set.
At z_true, the correct set I(z_true) is inside all the right half-spaces.
As z departs from z_true, it crosses these hyperplanes one by one,
each crossing removing a true antigen contact (false negative) or adding a wrong one.

```
RMSD error → displacement in Z-space → hyperplane crossings → ΔI(z)

GalaxyAD:  1.4 Å → displacement stays within most hyperplanes → ΔI ≈ ∅  → correct design
AlphaFold: 3.2 Å → displacement crosses many hyperplanes    → ΔI large → corrupted design
```

The success rate jump from ~1% to ~15% is not a smooth improvement —
it is a transition from the regime where z* crosses many contact boundaries
to the regime where z* stays inside the correct contact basin.

### 4. Message

Antigen conditioning in ProteinMPNN is not a gradient — it is a set membership.
The k-NN graph does not encode "how close" an antigen residue is — only whether it is
in or out of the 48-neighborhood. This makes the design landscape binary near the
boundary, and structure prediction precision determines which side of the boundary
the model operates on. GalaxyAD's 1.4 Å vs 3.2 Å is not an incremental improvement —
it is a regime change in whether the antigen contact information reaching ProteinMPNN
is correct.

---

## 5. Bang et al. 2025 — Closing the Inference Loop

*bioRxiv, Galux Inc.*

### 1. 목적

Validate the geometric claim from 2024 at atomic resolution and across diverse targets.
Show that the designed structures actually occupy the intended region of M_Z —
that the inference loop X → z* → Y* → fold(Y*) → z_actual closes with z_actual ≈ z*.
Extend to targets without experimental structures, testing P(Z|X) under distribution shift.

### 2. 모델링 설계

**데이터 단위**
cryo-EM electron density maps → atomic coordinates.
The data unit is now a physical measurement of where atoms are,
not a model prediction — it closes the loop between the computational manifold M_Z
and the physical structure manifold.

**표현 공간**
Two spaces are compared:
- Computational: z* from GalaxyAD (predicted CDR-H3 geometry)
- Physical: z_actual from cryo-EM (experimentally determined CDR-H3 geometry)

The distance ||z_actual - z*|| tests whether the model's M_Z approximates the physical one.

**변동 소스**
- Target diversity: 8 different antigens, different epitope geometries, different loop lengths
- Missing structure: some targets only have homology models as X — P(Z|X) under distribution shift
- Sequence specificity: designed antibodies must bind target but not closely related off-targets

**연결 구조**

```
X_target → GalaxyAD → z* → GaluxDesign → Y*
                                ↓
                          synthesize Y*
                                ↓
                          cryo-EM → z_actual
                                ↓
                     compare z_actual vs z*
```

The 2025 paper runs this full loop across 8 targets. The cryo-EM comparison is
the empirical test of whether the manifold approximation M_Z is physically valid.

### 3. 수식과 기하학

The full inference chain defines a composition of maps:

```
F: X → Y*    via    X → z*(X) → Y*(z*)

Test: F̃(Y*) = fold(Y*) ≈ z*(X)

where F̃ is the physical folding process (measured by cryo-EM)
```

If the manifold approximation holds, F̃(Y*) should lie in the same region of M_Z
as z*. The cryo-EM result at sub-3 Å resolution confirms this:
the designed loops adopt the predicted conformations, and the antigen contacts
are as intended.

For targets without experimental structures (homology models as X):

```
X_homology = X_true + ε_homology    where ε_homology is the modeling error

The question: does z*(X_homology) still lie in the correct contact basin?
```

The 2025 results show yes — the homology model error ε_homology does not push z*
across the contact-set boundaries identified in 2024.
The contact basin is wide enough to absorb reasonable homology modeling uncertainty.

### 4. Message

The inference loop closes physically. The computational manifold M_Z from GalaxyAD
is a valid approximation of the physical structure manifold at atomic resolution.
The discrete contact-set argument from 2024 is confirmed: designed structures
sit in the correct contact basin, and the sequences designed against those
contacts bind the intended targets. The generalization to homology models
suggests the contact basin geometry is robust to moderate X uncertainty.

---

## What the Implementation Revealed

Reading these five papers without having implemented ProteinMPNN,
the success-rate gap between RFdiffusion+ProteinMPNN (~1%) and GaluxDesign (~15%)
reads as an empirical outcome — "GalaxyAD happens to be more accurate."

After implementing ProteinMPNN from scratch — explicitly constructing the k-NN graph,
watching which residue indices fall in which neighborhood, tracing the tensor flow
from atomic coordinates through RBF encoding to the hidden state —
the gap becomes structurally necessary.

The antigen does not enter ProteinMPNN as a feature vector or an attention bias.
It enters as a set of integer indices: the antigen residues that fall within
the 48 nearest neighbors of each CDR residue. The design problem is not
"what sequence fits this antigen geometry?" — it is "what sequence fits this specific
set of k-NN edge connections?"

That reformulation makes the phase-transition argument from Bang et al. 2024 immediate:
the set I(z) is discrete and the boundary is sharp. A 2 Å displacement in CDR-H3
position does not degrade the antigen signal smoothly — it changes which residues
are in the set, discretely, at each hyperplane crossing.

In fMRI, I encountered the same structure:
functional connectivity is thresholded — two regions are "connected" or "not connected"
based on whether their correlation exceeds a boundary.
Improving the SNR of the BOLD measurement does not smoothly improve connectivity estimates —
below a threshold, the connection is invisible; above it, it is fully captured.
The design quality in Bang et al. follows the same logic,
in structure space rather than frequency space.

The implementation revealed that this threshold structure is not incidental —
it is designed into ProteinMPNN. The k-NN graph is a deliberate discretization
of a continuous spatial relationship. Understanding that discretization from the inside
is what makes the precision argument legible, rather than just empirically observed.

---

**Yuseon**
