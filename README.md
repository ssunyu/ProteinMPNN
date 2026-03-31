# ProteinMPNN Core-Architecture Implementation

PyTorch implementation of Core Architecture for Protein Sequence Design. 
Inverse folding from 3D structure to amino acid sequence.
<img width="2299" height="1080" alt="concat_map_visualization" src="https://github.com/user-attachments/assets/e86f4db6-1155-488b-97f4-8045f957613f" />
## 🧠 Message Passing & Gradient Propagation
> **Local Interaction에서 Chemical Logits까지의 순방향(Forward) 및 역방향(Backward) 텐서 연산 흐름**

### 1. Computation Graph
단일 레이어 내에서 정보가 어떻게 결합, 필터링, 누적되어 최종적으로 화학적 에너지 상태(Logits)로 투영되는지를 나타내는 연산 그래프입니다.

```mermaid
flowchart TD
    classDef data fill:#f8f9fa,stroke:#ced4da,stroke-width:2px,color:#212529
    classDef op fill:#e7f5ff,stroke:#339af0,stroke-width:2px,color:#0b7285
    classDef weight fill:#fff3bf,stroke:#fcc419,stroke-width:2px,color:#862e9c
    classDef loss fill:#ffe3e3,stroke:#fa5252,stroke-width:2px,color:#c92a2a

    h_i["h_i (Self)<br/>1 x 128"]:::data
    h_j["h_j (Neighbor)<br/>1 x 128"]:::data
    e_ij["e_ij (Edge)<br/>1 x 128"]:::data

    concat(("Concat<br/>||")):::op
    h_i --> concat
    h_j --> concat
    e_ij --> concat

    z["z (Joint Info)<br/>1 x 384"]:::data
    concat --> z

    w_msg[/"W_msg<br/>384 x 128"/]:::weight
    matmul1(("MatMul<br/>*")):::op
    z --> matmul1
    w_msg --> matmul1

    m["m (Message)<br/>1 x 128"]:::data
    matmul1 --> m

    add(("Add<br/>+")):::op
    h_i -.->|"Skip Connection"| add
    m --> add

    h_out["h_out (Posterior)<br/>1 x 128"]:::data
    add --> h_out

    w_dec[/"W_dec<br/>128 x 20"/]:::weight
    matmul2(("MatMul<br/>*")):::op
    h_out --> matmul2
    w_dec --> matmul2

    L["L (Logits)<br/>1 x 20"]:::data
    matmul2 --> L

    loss_fn(("Cross<br/>Entropy")):::loss
    L --> loss_fn
    y_true[/"Y_true"/]:::weight
    y_true --> loss_fn

    J["J (Loss)<br/>Scalar"]:::data
    loss_fn --> J
---

## CORE 1: Modeling 목적 및 수학적 구조

### 1) 목적
* **단백질 3차 구조 기반 아미노산 서열 Reverse Inference**

### 2) 구조 설계 (Logic)
* **Hierarchical Layer ($Z \rightarrow X \rightarrow Y$)**
    * **(Z)** 잔기 간 Interaction $\rightarrow$ **(X)** 3차원 구조적 제약 $\rightarrow$ **(Y)** 아미노산 서열 예측.
    * 모델링 인과 흐름: 구조($X$)를 조건으로 서열($Y$)을 추론 (현상적 인과와 역행).
* **정보 및 자유도 정의**
    * **정보**: Signal(잔기 간 Local Interaction)과 Noise의 엄격한 정의.
    * **공간**: RBF Filtering을 통한 거리별 기능적 표현 자유도(Resolution) 확장.
    * **데이터**: 기능적 독립 Unit 정의 (Message Passing을 통한 i.i.d 가정 충족).
    * **자유도**: 정보의 자유도와 표현 공간의 자유도 주체를 명확히 구분.

---

## CORE 2: 딥러닝 아키텍처 구현

### 1) Encoder (Tensor Flow)
물리적 공간을 그래프 임베딩 공간으로 전환하는 데이터 Unit 흐름.

* **Step 1. Physical Space**: `(N, 5, 3)` — Backbone 잔기 $\times$ 구성 원자 $\times$ Cartesian 좌표.
* **Step 2. Graph Space**: `(N, k, 25)` — 잔기 간 거리 Graph.
* **Step 3. Embedding**: `(N, k, 25)` $\rightarrow$ `(N, k, 400)` $\rightarrow$ `(N, k, 128)` — Resolution 확장 및 임베딩.
* **Step 4. Node Update**: `(N, 128)` — 잔기(Node) 간 Local Interaction 업데이트.

### 2) Decoder (Ability Enhancement)
Random Ordering 및 Masking을 통한 생성 능력 극대화.

* **Random Ordering**: 1D Direction Bias를 극복하기 위한 무작위 추론 순서 설계.
* **Masking**: Mask Token을 활용하여 결정된 잔기의 구조적 제약 조건을 학습.

---

## CORE 3: 설계 검증 및 시가화

* **Smoke Test**: 각 Architecture Module별 설계 의도 및 수리적 타당성 검증.
* **Real Data Visualization**: PDB 실제 데이터셋 기반 가설 검증 및 시각화.
    * fMRI 기반 RSA/Mantel Test를 통한 공간적 인지 지평선 분석.

---

## Author
**Yuseon**
