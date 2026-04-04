# Reading Notes — Antibody Design as a Structured Inference Problem

*A modeler's view of the field, grounded in implementing ProteinMPNN from scratch.*
*My own view for integrating distributed viewpoint of antibody modeling after reading 5 reference papers*
*The papers are reference points in a picture I was already trying to draw.*

---

## The Modeler's Starting Point — Seeing the Whole Before Choosing a Model

Before setting any equation, a modeler asks:
*what phenomenon am I looking at, and what are the natural axes of that phenomenon?*

For this field, the phenomenon is:

> **Binding is not a property of the antibody or the antigen.
> It is a relational event — something that co-arises when their geometries become compatible.**

Two entities exist in their own spaces.
One is relatively fixed (antigen surface X).
One fluctuates (CDR loop Z).
A third variable is what we design (sequence Y).

The question is: in what space does their relationship live,
and what information defines whether that relationship is "binding"?

---

### The Integrated Geometry

Rather than four separate perspectives, start with one picture:

```
         Y ∈ {sequence space}
         ▲
         │  P(Y|Z) — inverse folding
         │
Z ∈ M_Z  ◀────────── Σ_Z|X  ───────────  X ∈ M_X
(CDR loop)  P(Z|X) — structure prediction  (antigen surface)

                        ↕
               B = binding manifold
          { (x, z) : contact geometry compatible }
```

The **binding manifold B** is the unifying geometry.
It is a surface in the joint space M_X × M_Z —
the set of (antigen, CDR) configurations where binding is geometrically feasible.

Everything in this field is about one question:
*can we design a Y such that fold(Y) reliably lands inside B for a given X?*

---

### Setting the Axes

The natural axes of this relationship:

```
Axis 1:  Z relative to X          (structural compatibility — does Z ∈ B_X?)
Axis 2:  Z relative to Y          (sequence-structure consistency — does fold(Y) ≈ Z?)
Axis 3:  X relative to Y (via Z)  (the design objective — does Y bind X?)
```

Each axis has a fluctuation source and a covariance structure:

| Axis | Fluctuation source | Covariance | What constrains it |
|------|--------------------|------------|-------------------|
| Z \| X | CDR loop flexibility | Σ_Z\|X — elongated, tangential to X | steric contact with antigen |
| Y \| Z | sequence diversity for a given fold | Σ_Y\|Z — concentrated near native sequences | ProteinMPNN's learned distribution |
| Y \| X | the design target | Σ_Y\|X = ∫Σ_Y\|Z · P(Z\|X)dZ | the full integral — intractable |

---

### The Covariance Picture

The joint ellipsoid over (X, Z, Y):

```
Σ_XZY = [ Σ_XX   Σ_XZ   Σ_XY ]
         [ Σ_ZX   Σ_ZZ   Σ_ZY ]
         [ Σ_YX   Σ_YZ   Σ_YY ]
```

The binding manifold B defines a **preferred subspace** of this joint ellipsoid:
the directions in (X, Z) space where binding-compatible geometry is concentrated.

What makes CDR-H3 design hard:
Σ_Z|X has large eigenvalues in the tangential directions (the loop can slide along the antigen surface) — these are exactly the directions where small changes in Z change the contact set I(z).

```
dominant eigenvectors of Σ_Z|X  ≈  directions of highest I(z) sensitivity
```

The loop's freedom and the design's sensitivity are aligned in the same subspace.
This is the core tension in the field.

---

### The Information Bottleneck

The contact set I(z) is the lossy compression of the continuous geometry into the signal ProteinMPNN actually uses:

```
continuous Σ_Z|X  →[k-NN threshold]→  discrete I(z) ∈ {0,1}^n_antigen

Information kept:   who is close to whom (topology)
Information lost:   how close, in what direction, with what relative orientation
```

This compression is not smooth — it is a step function at each k-NN boundary.
The covariance structure after compression:

```
Σ_I|X  ≈  piecewise constant in Z — flat within contact basins, jumps at boundaries
```

The binding manifold B, viewed through I(z), becomes a discrete set of contact basins.
Design succeeds when z* lands in the correct basin.
Design fails when z* is displaced across a basin boundary.

---

### The Four Perspectives as Cross-Sections

The four perspectives below are not separate frameworks —
they are cross-sections of this same joint picture,
each illuminating a different part of the geometry:

| Perspective | What it sees in the joint picture |
|------------|----------------------------------|
| Geometric | the shape of M_Z, M_X, B and their intersection |
| Probabilistic | P(Z\|X) as distribution over M_Z; P(Y\|Z) as inverse folding |
| Information | I(z) as the lossy compression of continuous geometry |
| Statistical | Σ_XZY and its rank — how much of X survives into Y |

[YOUR WORDS — before moving to the four perspectives:
in your language, what is the "event" of binding?
What does it mean geometrically for fold(Y) to land inside B?
What is the connection to how you think about phenomena as co-arising from connected relationships?]

---

## The Central Relationship

All five papers are approximating one integral:

```
P(Y | X) = ∫ P(Y | Z) · P(Z | X) dZ
```

| Variable | Entity | Space | Role |
|----------|--------|-------|------|
| X | antigen epitope geometry | R^(3·n_ag) | context — fixed |
| Z | CDR loop conformation | M_Z ⊂ R^(3·n_cdr) | latent — fluctuating bridge |
| Y | CDR amino acid sequence | {0..19}^n | target — design variable |

Every model makes a different choice about which term to approximate,
which to collapse to a point, which to treat as a distribution.
That choice determines what part of Σ_XZY survives into the design.

---

## Four Perspectives on the Same Relationship

### 1. Geometric

The relationship lives in a joint space:

```
(X, Z) ∈ R^(3·n_ag) × M_Z        M_Z ⊂ R^(3·n_cdr)

Binding = a subset B ⊂ M_Z × M_X
         where CDR geometry and antigen geometry are mutually compatible
```

The binding-compatible region B is a lower-dimensional manifold
in the full joint space — a surface, not a volume.

Key geometric fact: M_Z is high-entropy near X.
CDR-H3 has no evolutionary conservation → wide conformational distribution.
The covariance ellipsoid Σ_Z|X is elongated tangentially to the antigen surface:

```
Σ_Z|X = U · Λ · U^T

large λ   →  tangential eigenvectors  (loop can slide along antigen surface)
small λ   →  normal eigenvectors      (sterically constrained into antigen)
```

[YOUR WORDS — what does the shape of this ellipsoid mean for the binding manifold B?
Where in this joint space does the "signal" of binding live vs. the noise of loop flexibility?]

---

### 2. Probabilistic

The same relationship, written as a conditional distribution:

```
P(Y | X) = ∫ P(Y | Z) · P(Z | X) dZ
```

- P(Z | X): how does antigen geometry constrain CDR loop conformation?
  → the projection of Σ_Z|X onto the binding manifold
- P(Y | Z): given a loop conformation, what sequence is consistent with it?
  → inverse folding, conditioned on a point in M_Z

Every model in this field approximates this integral differently:
- collapse Z to a point z* (point estimate)
- model P(Z|X) as a distribution (diffusion)
- jointly optimize both terms (tight coupling)

[YOUR WORDS — in your probabilistic language: which term carries more uncertainty?
Where does the conditioning lose the most information?
How does the choice between fixed Z and random Z change what the model "sees"?]

---

### 3. Information

What is the minimal data unit that carries the binding-relevant signal?

From implementing ProteinMPNN — the answer is concrete:

```
I(z) = { r ∈ X : r ∈ kNN(CDR residue, k=48) }

The antigen enters sequence design only as set membership in a k-NN ball.
Not a continuous field. Not a distance vector. A binary set.
```

This is a specific choice about what information to preserve from the geometric relationship.
It discards: exact distances, relative orientations, long-range interactions.
It keeps: the discrete topology of who is close to whom.

The information bottleneck created by this choice:

```
continuous geometry   →[projection]→   discrete contact set I(z)

I(z) is piecewise constant in z:
  flat almost everywhere
  discontinuous at k-NN boundary surfaces
```

[YOUR WORDS — what does this data unit capture well, and what does it structurally miss?
In your language: what fluctuation source is visible through I(z),
and what fluctuation is made invisible by this embedding choice?]

---

### 4. Statistical / Matrix

The same relationship, as a covariance structure across the joint space:

```
joint distribution over (X, Z, Y):

Σ = [ Σ_XX   Σ_XZ   Σ_XY ]
    [ Σ_ZX   Σ_ZZ   Σ_ZY ]
    [ Σ_YX   Σ_YZ   Σ_YY ]
```

The design problem is: given X, infer Y.
The information channel is X → Z → Y (with Z as the latent bridge).

Y ⊥ X | Z  is the conditional independence assumed by ProteinMPNN-based pipelines:
once Z is fixed (the graph is built), X provides no additional information to Y.

This assumption holds when Σ_ZX captures all the X → Y covariance.
It fails when the Z projection discards X–Y covariance that is not mediated by contact topology.

```
rank(Σ_ZX) determines how much of the antigen geometry survives into the design
```

[YOUR WORDS — in your matrix language: what is the effective rank of this channel?
What directions of X-variation are visible in Y through this conditioning,
and what directions are marginalized out?
How does the k-NN threshold affect the rank of Σ_ZX?]

---

## Where the Papers Stand

Each paper makes a choice about which perspective to prioritize and which term to approximate:

| Paper | Axis of intervention | Core modeling choice |
|-------|---------------------|---------------------|
| Hummer et al. 2022 | Geometric | maps the width of Σ_Z\|X — identifies CDR-H3 as the high-entropy bottleneck |
| Watson et al. 2023 | Probabilistic | learns P(Z\|X) via diffusion score; treats Y\|Z as separable |
| Abramson et al. 2024 | Geometric + Statistical | replaces point z* with distribution — makes Σ_Z\|X explicit |
| Bang et al. 2024 | Information | shows I(z) is threshold-structured — z* precision determines whether G(z*) = G(z_true) |
| Bang et al. 2025 | Full closure | cryo-EM validates fold(Y*) ≈ z* — the manifold approximation holds physically |

[YOUR WORDS — which axis do you think is most underexplored?
Which choice changes the most about what information survives from X to Y?]

---

## What the Implementation Revealed

[YOUR WORDS

The four perspectives above describe the same phenomenon from the outside.
This section is what you saw from the inside — having built ProteinMPNN
and then read these papers.

Anchoring questions:
- When you built the k-NN graph: which of the four perspectives became suddenly concrete?
- When you traced the tensor flow through RBF encoding and message passing:
  what happened to Σ_ZX along the way?
- The conditional independence Y ⊥ X | Z: when did this assumption become
  structurally visible rather than just stated?
- Where did the fMRI connection appear — which axis, and in what form?]

---

**Yuseon**
