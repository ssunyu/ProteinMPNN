# ProteinMPNN(Dauparas, 2022) Implementation
> PyTorch implementation of Graph-based Protein Sequence Design via Structural Constraints.
그래프 이론에 기반한 단백질 서열 설계 모델 ProteinMPNN을 바닥부터 시작하는 모델러의 입장에서 어떤 논리적 사고방식을 거쳐서 모델이 만들어졌는지에 대한 high-level의 논리 과정에 따른 모델의 Core Architecture를 바닥부터 구현합니다. 

---
## CORE1: Modeling의 목적과 수학적 구조 설계 

### 1) 목적 
* **단백질 3차 구조로 아미노산 서열 reverse inference**

### 2) 구조 설계 
* **1. Hierarchical layer (Z - X - Y)**
    * **(Z) 잔기간 interaction**이 - **(X) 단백질 3차원 구조적 제약**으로 표현되고 - **(Y) 이를 통해 아미노산 서열(Y)**을 예측할 수 있을 것이다. 
    * 실제 현상적인 인과가 아닌 **모델링의 인과 흐름** (현상적 인과 : 아미노산 서열 -> 구조)
* **2. 정보, 공간, 데이터, 자유도**
    * **a. 정보**: 모델링 목적에 따른 Signal & Noise 정의 (Signal: 잔기간 local interaction)
    * **b. 공간**: 정보가 표현 되는 공간. 정보 표현의 독립적 자유도 고려한 resolution 설정 (RBF filtering으로 거리에 다른 기능적 표현 자유도 확장)
    * **c. 데이터**: 정보를 담는 기능적 독립 unit (데이터간의 종속성이 있는경우 종속성에 관한 추가적 모델링으로 i.i.d 가정을 만족하도록)
    * **e. 자유도**: 정보의 자유도인지 표현공간의 자유도인지 주체를 명확히 구분 

---

## CORE2: 목적에 따른 딥러닝 설계 구현 

### 1) Overview
* Learning graph overview

### 2) Encoder 
* **정보 공간 전환 및 데이터 unit Tensor flow**
    1. **물리적 표현 공간**: (Nx5x3) Backbone 잔기 x 구성 원자 x Cartesian 거리좌표 
    2. **잔기간 거리 graph 공간**: (Nxkx25) 
    3. **잔기간 거리 graph resolution 확장 및 embedding**: (Nxkx25 -> Nxkx400 -> Nxkx128)
    4. **잔기(노드)간 local interaction을 통한 업데이트**: (Nx128)

### 3) Decoder
* **Random ordering & masking을 통한 decoding ability 향상**
    1. 1d direction이 아닌 **random ordering**을 통한 direction bias의 극복 
    2. **Mask Token**을 통한 masking으로 결정된 잔기의 구조적 제약 학습 

---

## CORE3: Smoke test를 통한 설계 검증 및 실제 PDB 데이터를 통한 시각화 

### 1) Smoke test 
* 각 architecture module별 설계 의도 검증 

### 2) Real data visualization 
* 코어 아키텍처 기반 및 실제 데이터셋 활용 가설 검증 및 시각화

--- 
## Author
* **Yuseon**