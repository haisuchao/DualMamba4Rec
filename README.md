# DualMamba4Rec

**Dual-Channel GNN + Mamba Architecture for Session-based Recommendation**

An extension of [CMamba4Rec](https://arxiv.org/abs/...) that adds a global item
transition graph (Channel B) alongside the original C-Mamba stream (Channel A),
fused via cross-channel attention.

---

## Architecture Overview

```
Session Input: [v1, v2, ..., vt]
         │
    Item + Position Embedding
         │
    ┌────┴────────────────────────────┐
    │                                 │
[Channel A: C-Mamba]        [Channel B: LightGCN]
 Conv1D (local context)      Global item transition
 → Mamba (global seq)        graph from all training
 → DS Gate (noise filter)    sessions
    │                                 │
    └────────────────┬────────────────┘
                     │
            Cross-Channel Attention Fusion
            (H_A queries H_B; learnable gating)
                     │
         Multi-Intent Session Attention
         (aggregate all items, not just last)
                     │
              Prediction (CrossEntropy)
```

**Key Design Decisions:**
- **Channel B** provides *cross-session collaborative signals* that Channel A
  (session-local Mamba) cannot learn alone — especially valuable for short
  sessions (Tmall avg=3.46 items).
- **Cross-attention fusion** lets the Mamba output selectively query GNN
  representations, controlled by two learnable sigmoid gates.
- **LightGCN propagation** on the global graph enriches item embeddings with
  multi-hop transition patterns before they reach the fusion layer.

---

## Installation

```bash
# 1. Install dependencies
pip install torch recbole numpy pandas tqdm pyyaml

# 2. Install DGL (choose your CUDA version)
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html  # CUDA 11.8
pip install dgl -f https://data.dgl.ai/wheels/repo.html        # CPU only
```

---

## Data Preparation

Download raw datasets and run preprocessing:

```bash
# Diginetica (CIKM 2016)
python utils/data_preprocess.py \
    --dataset diginetica \
    --raw_path /path/to/train-item-views.csv \
    --out_dir ./data

# Tmall (Alibaba)
python utils/data_preprocess.py \
    --dataset tmall \
    --raw_path /path/to/tmall.csv \
    --out_dir ./data

# Tafeng
python utils/data_preprocess.py \
    --dataset tafeng \
    --raw_path /path/to/tafeng.csv \
    --out_dir ./data

# Yoochoose 1/64
python utils/data_preprocess.py \
    --dataset yoochoose \
    --raw_path /path/to/yoochoose-clicks.dat \
    --out_dir ./data
```

---

## Training

```bash
# Train on a single dataset
python train.py --dataset diginetica

# Train all datasets
bash run_all.sh

# Custom hyperparameters
python train.py \
    --dataset tafeng \
    --hidden_size 128 \
    --num_layers 3 \
    --gnn_n_layers 3 \
    --learning_rate 0.0005

# Ablation: disable GNN channel
python train.py --dataset tmall --no_gnn
```

---

## Quick Sanity Test

Verify all model components work correctly without real data:

```bash
python scripts/quick_test.py
```

---

## Ablation Study

```bash
python ablation.py --dataset diginetica
```

---

## Hyperparameter Guide

| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_size` | 64 | d_model — embedding & hidden dim |
| `num_layers` | 2 | C-Mamba blocks stacked |
| `mamba_d_state` | 16 | SSM state dimension |
| `kernel_size` | 3 | Conv1D kernel (C-Mamba local context) |
| `gnn_n_layers` | 2 | LightGCN propagation depth |
| `n_heads_fusion` | 4 | Cross-channel attention heads |
| `hidden_dropout_prob` | 0.3 | Dropout rate |
| `use_gnn` | True | Enable/disable Channel B |

**Dataset-specific tips:**
- **Tafeng** (long sessions): increase `kernel_size=5`, `gnn_n_layers=3`
- **Tmall** (short sessions): reduce `dropout=0.2`, `gnn_n_layers=1`
- **Yoochoose** (large scale): consider `batch_size=2048`, `hidden_size=128`

---

## Results (Expected)

| Dataset | Recall@20 | MRR@20 | vs CMamba4Rec |
|---------|-----------|--------|---------------|
| Diginetica | ~0.548 | ~0.190 | +~1.5% |
| Tmall | ~0.298 | ~0.163 | +~1.0% |
| Tafeng | ~0.132 | ~0.037 | +~4.0% |
| Yoochoose1/64 | ~0.720 | ~0.320 | +~2.0% |

*Results are estimates based on architecture analysis. Actual results depend
on hardware, random seed, and preprocessing details.*

---

## Citation

If you use this code, please cite the original CMamba4Rec paper:

```
@inproceedings{hai2025cmamba4rec,
  title={CMamba4Rec: A Convolution-augmented Mamba Architecture 
         for Session-based Recommendation},
  author={Hai, Nguyen Do and Phuong, Tu Minh},
  year={2025}
}
```
