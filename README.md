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

---

## Setup and Installation

### Requirements
```bash
# Core
Python 3.12
TensorFlow 2.19
PyTorch
NumPy == 1.26.4      # Must pin — NumPy 2.0 breaks NGCF and RecBole
SciPy
Pandas == 1.5.3

# For LightGCN
recbole
ray
```

### Install
```bash
# Clone NGCF repository
git clone https://github.com/xiangwang1223/neural_graph_collaborative_filtering.git

# Install dependencies
pip install numpy==1.26.4
pip install pandas==1.5.3
pip install scipy torch
pip install recbole ray
```

> ⚠️ **Important:** Always install NumPy first and pin it to 1.26.4 
> before installing other packages. NumPy 2.0 breaks both the NGCF 
> codebase and RecBole.

---

## Running NGCF

### Apply Compatibility Patches
```python
with open('NGCF/NGCF.py', 'r') as f:
    content = f.read()

content = content.replace('import tensorflow as tf',
    'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()')
content = content.replace('tf.contrib.layers.xavier_initializer()',
    'tf.glorot_uniform_initializer()')
content = content.replace('tf.contrib.layers.', 'tf.layers.')
content = content.replace('np.mat([coo.row, coo.col]).transpose()',
    'np.asmatrix(np.vstack([coo.row, coo.col]).T)')

with open('NGCF/NGCF.py', 'w') as f:
    f.write(content)

# Fix metrics.py
with open('NGCF/utility/metrics.py', 'r') as f:
    metrics = f.read()
metrics = metrics.replace('np.asfarray', 'np.asarray')
with open('NGCF/utility/metrics.py', 'w') as f:
    f.write(metrics)
```

### Train from Scratch
```bash
cd NGCF
python NGCF.py \
    --dataset amazon-book \
    --epoch 200 \
    --save_flag 1 \
    --weights_path ../checkpoints/NGCF/ \
    --regs [1e-5]
```

### Resume from Checkpoint
```bash
python NGCF.py \
    --dataset amazon-book \
    --epoch 200 \
    --save_flag 1 \
    --pretrain 1 \
    --weights_path ../checkpoints/NGCF/ \
    --regs [1e-5]
```

---

## Running LightGCN

### Convert Data
```python
import pandas as pd, os

os.makedirs('recbole_data/amazon_book', exist_ok=True)

train_data = []
with open('Data/amazon-book/train.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2: continue
        for item_id in parts[1:]:
            train_data.append([parts[0], item_id])

test_data = []
with open('Data/amazon-book/test.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 2: continue
        for item_id in parts[1:]:
            test_data.append([parts[0], item_id])

df = pd.DataFrame(train_data + test_data, 
                  columns=['user_id:token', 'item_id:token'])
df['rating:float'] = 1.0
df['timestamp:float'] = range(len(df))
df.to_csv('recbole_data/amazon_book/amazon_book.inter', 
          sep='\t', index=False)
```

### Train LightGCN
```python
from recbole.quick_start import run_recbole

config_dict = {
    'data_path': 'recbole_data',
    'embedding_size': 64,
    'n_layers': 3,
    'epochs': 100,
    'train_batch_size': 2048,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'eval_args': {
        'split': {'RS': [0.8, 0.1, 0.1]},
        'order': 'RO',
        'group_by': 'user',
        'mode': 'full'
    },
    'metrics': ['Recall', 'NDCG', 'Precision', 'Hit'],
    'topk': [20],
    'valid_metric': 'Recall@20',
    'checkpoint_dir': 'checkpoints/LightGCN/',
    'use_gpu': True,
}

run_recbole(model='LightGCN', dataset='amazon_book', 
            config_dict=config_dict)
```

---

## Hyperparameters

| Parameter | NGCF (Ours) | NGCF (Paper) | LightGCN (Ours) |
|---|---|---|---|
| Embedding Size | 64 | 64 | 64 |
| Layers | 1 | 3 | 3 |
| Learning Rate | 0.01 | 0.0001 | 0.001 |
| Regularization | 1e-5 | 1e-5 | 1e-4 |
| Batch Size | 1024 | 1024 | 2048 |
| Optimizer | Adam | Adam | Adam |
| Loss | BPR | BPR | BPR |
| Epochs | 20 | 200 | 100 |

---

## Training on Google Colab

Both models were trained on Google Colab with Tesla T4 GPU.

| Model | Time/Epoch | Total (20 epochs) |
|---|---|---|
| NGCF | ~225 seconds | ~75 minutes |
| LightGCN | ~422 seconds | ~140 minutes |

For persistent training across sessions, weights are saved to 
Google Drive automatically. See `notebooks/BTP_CODE_1_0.ipynb` 
for the complete reproducible pipeline.

---

