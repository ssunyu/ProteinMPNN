# ProteinMPNN — Core Architecture from Scratch

PyTorch implementation of core architecture for [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187) (Dauparas et al., *Science* 2022).
Built from the ground up — not to reproduce numbers, but to own the design decisions.

<img width="2235" height="1122" alt="antibody_antigen_contact_analysis" src="https://github.com/user-attachments/assets/0586c3ac-0c35-4e94-913d-eedbe138508e" />

---

## Motivation

My background is in fMRI — extracting spatiotemporal structure from high-dimensional brain signals, decomposing them with ICA, and building predictive models on top of those representations. The core question was always: *given an observed pattern, what latent structure generated it?*

Inverse protein folding is the same question in a different domain. A backbone structure is observed; the sequence that folds into it is the latent variable to recover. What I wanted to understand before extending this model was exactly *how* it encodes that inference — not at the API level, but at the level of information flow and geometric transformation.

---

## Modeling Logic: X → Z → Y

The architecture is built around a three-level causal hierarchy:

```
(X) 3D structural constraints →  (Z) residue interactions  →  (Y) amino acid sequence
```

This is a deliberate inversion of physical causation. In nature, sequence determines structure. Here, structure is the condition and sequence is the inference target — a reverse inference problem.

What I found useful in framing this: the model has to separately manage two kinds of degrees of freedom.

- **Information degrees of freedom**: what constitutes signal (local residue interactions) vs. noise, and how that boundary is drawn. Message passing defines the functional unit of information — making neighboring residues approximately i.i.d. from the model's perspective.

- **Representational degrees of freedom**: how much resolution the embedding space is given to express distance relationships. RBF encoding is the key step here — expanding a 1D scalar distance into a 16-dimensional basis vector, so the model can learn which distance regimes matter for sequence identity.

Keeping these two distinct was the main thing I wanted to verify by implementing from scratch.

---

## Architecture: Tensor Flow

**Encoder** — physical space to graph embedding space

| Step | Space | Shape | Description |
|------|-------|-------|-------------|
| 1 | Physical | `(res, 5, 3)` | backbone atoms × Cartesian coordinates |
| 2 | Graph | `(res, k, 25)` | pairwise atom distances via k-NN |
| 3 | Embedding | `(res, k, 400)` | RBF expansion: 25 pairs × 16 basis = 400 |
| 4 | Hidden | `(res, 128)` | node representations after message passing |

**Decoder** — two design decisions that matter

- **Random ordering**: left-to-right generation biases early residues (no neighbor context) against late ones (full context). Random permutation order removes this asymmetry — each residue sees the same expected amount of already-decided neighborhood.
- **Causal masking**: the autoregressive mask encodes *when* information is available, not just *what* information. Decided neighbors contribute their sequence embedding; undecided ones contribute a zero vector. Time becomes a spatial gate on the graph.

---

## Validation and Visualization

**Smoke test** verifies each module independently — shape correctness, gradient flow, and sampling validity — using a synthetic backbone with no external data dependency.

**Real-data visualization** uses PDB structure 1MLC (antibody–antigen complex) to test whether k-NN contact maps based purely on Cα distances capture binding interface structure.

The finding: distance-based graphs systematically underrepresent long-range and cross-chain interactions at the binding interface. Residues that are spatially separated can be functionally coupled. In fMRI, this is the canonical motivation for functional connectivity analysis over anatomical proximity (RSA/Mantel test framework). The same gap exists here.

Implementing the model also changed how I read the surrounding literature — where ProteinMPNN sits in the current antibody design landscape, why structure prediction accuracy matters at the graph level, and what the gap between ProteinMPNN-based pipelines and tightly-coupled approaches (e.g., Bang et al. 2024, 2025) looks like from the inside. Those reading notes are in [`paper_notes.md`](./paper_notes.md).

---

## Structure

```
core_architecture/
├── config.py          # hyperparameters, constants, data containers
├── preprocessing.py   # physical space → graph feature space
├── encoder.py         # message passing encoder
├── decoder.py         # autoregressive decoder with causal masking
├── model.py           # ProteinMPNN: encode / decode / forward
└── training.py        # NLL loss with label smoothing, training step
```

---

## Usage

```bash
python -m core_architecture.smoke_test
```

```python
from core_architecture.model import ProteinMPNN
from core_architecture.config import Config

model = ProteinMPNN(Config())
enc   = model.encode(n_coords, ca_coords, c_coords, o_coords)
out   = model.decode(enc, temperature=0.1)   # out.sequence, out.log_prob
```

---

## References

Dauparas, J. et al. *Robust deep learning–based protein sequence design using ProteinMPNN.* Science 378, 49–56 (2022).

Hummer, A.M., Abanades, B. & Deane, C.M. *Advances in computational structure-based antibody design.* Current Opinion in Structural Biology 74, 102379 (2022).

Watson, J.L. et al. *De novo design of protein structure and function with RFdiffusion.* Nature 620, 1089–1100 (2023).

Abramson, J. et al. *Accurate structure prediction of biomolecular interactions with AlphaFold 3.* Nature 630, 493–500 (2024).

Bang, I. et al. *Accurate antibody loop structure prediction enables zero-shot design of target-specific antibodies.* (2024).

Bang, I. et al. *Precise, specific, and sensitive de novo antibody design across multiple cases.* bioRxiv (2025).

---

**Yuseon**
