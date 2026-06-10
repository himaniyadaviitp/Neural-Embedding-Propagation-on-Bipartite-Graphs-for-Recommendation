# Neural-Embedding-Propagation-on-Bipartite-Graphs-for-Recommendation
# Neural Embedding Propagation on Bipartite Graphs for Recommendation
### Implementation and Analysis of NGCF and LightGCN on Amazon-Book Dataset

> B.Tech Project | Department of Computer Science and Engineering  
> Indian Institute of Technology Patna  
> Supervisor: Dr. Sourav Kumar Dandapat  
> Author: Himani Yadav (2201AI16)

---

## Overview

This project implements and comparatively evaluates two graph neural 
network based recommendation frameworks on the Amazon-Book benchmark 
dataset:

- **NGCF** (Neural Graph Collaborative Filtering) — SIGIR 2019
- **LightGCN** (Light Graph Convolutional Network) — SIGIR 2020

Both models address a fundamental limitation of traditional 
collaborative filtering: the failure to encode the rich collaborative 
signal present in the user-item interaction graph structure into the 
embedding learning process.

---

## Key Results

| Model | Source | Recall@20 | NDCG@20 | Epochs |
|---|---|---|---|---|
| MF | Paper | 0.0250 | 0.0196 | 200 |
| NeuMF | Paper | 0.0258 | 0.0200 | 200 |
| HOP-Rec | Paper | 0.0309 | 0.0232 | 200 |
| NGCF | Paper | 0.0337 | 0.0261 | 200 |
| LightGCN | Paper | 0.0411 | 0.0315 | 200 |
| **NGCF (Ours)** | **This Repo** | **0.0484** | **0.0787** | **20** |

> Our NGCF implementation surpasses all paper-reported baselines 
> by **+43.6% Recall** and **+201% NDCG** in only 20 epochs.

---

## Dataset

**Amazon-Book** (publicly available via NGCF repository)

| Property | Value |
|---|---|
| Users | 52,643 |
| Items | 91,599 |
| Interactions | 2,984,108 |
| Sparsity | 0.00062 |
| Train Split | 80% |
| Test Split | 20% |

---

## Models

### NGCF
Propagates embeddings through the user-item bipartite graph using 
learnable weight matrices and non-linear activations:

$$e_u^{(l)} = \text{LeakyReLU}\left( W_1^{(l)} e_u^{(l-1)} + 
\sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} 
\left( W_1^{(l)} e_i^{(l-1)} + W_2^{(l)}(e_i^{(l-1)} \odot e_u^{(l-1)}) 
\right) \right)$$

### LightGCN
Simplifies NGCF by removing all transformations, relying purely on 
graph structure:

$$e_u^{(l)} = \sum_{i \in \mathcal{N}_u} 
\frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e_i^{(l-1)}$$

Both models are trained using **BPR Loss**:

$$L_{BPR} = \sum_{(u,i,j)} -\ln\sigma(\hat{y}_{ui} - \hat{y}_{uj}) 
+ \lambda\|\Theta\|_2^2$$

---

## Engineering Contributions

The original NGCF codebase (TF1 + NumPy 1.x) required 4 compatibility 
fixes to run on modern environments:

| Fix | Original | Fixed |
|---|---|---|
| TF API | `import tensorflow as tf` | `import tensorflow.compat.v1 as tf` |
| Initializer | `tf.contrib.layers.xavier_initializer()` | `tf.glorot_uniform_initializer()` |
| Sparse Tensor | `np.mat([coo.row, coo.col])` | `np.asmatrix(np.vstack([...]).T)` |
| Metrics | `np.asfarray(r)` | `np.asarray(r)` |

---

