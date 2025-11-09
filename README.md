# TS-CVA: Time Series Forecasting via Cross-modal Variable Alignment

[![License](https://img.shields.io/badge/License-S--Lab%201.0-blue.svg)](LICENSE.txt)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1.2](https://img.shields.io/badge/PyTorch-2.1.2-red.svg)](https://pytorch.org/)

**TS-CVA** is an advanced time series forecasting framework that extends [TimeCMA](https://arxiv.org/abs/2406.01638) with **dual-modality learning** for enhanced prediction performance.

## Key Features

- **Triple-Modal Architecture**: Combines Time Series + Vector (TS2Vec) + Context (LLM) modalities
- **Multi-Task Learning**: Joint training with contrastive and forecasting losses
- **Robust Representations**: Contrastive learning via TS2Vec for better generalization
- **Flexible Training**: End-to-end or two-stage (pre-train + fine-tune)
- **Auto-GPU Selection**: Automatically finds free GPU on multi-GPU systems
- **Extensible Design**: Easy to add external information sources

## Architecture

```
Input Time Series
    ↓
┌───────────────┬──────────────────┬────────────────────┐
│ TS Branch     │ Vector Branch    │ Context Branch     │
│ (Transformer) │ (TS2Vec CNN)     │ (LLM Encoder)      │
│               │ + Contrastive    │                    │
└───────┬───────┴────────┬─────────┴──────────┬─────────┘
        │                │                    │
        └────────────────┼────────────────────┘
                         ↓
              Triple-Modal Alignment
                 (Cross-Attention)
                         ↓
                    Decoder
                         ↓
                  Forecasting
```

### What's New vs TimeCMA?

| Component | TimeCMA | TS-CVA |
|-----------|---------|--------|
| Time Series Branch | ✅ | ✅ |
| Context Modality (LLM) | ✅ | ✅ Enhanced |
| **Vector Modality (TS2Vec)** | ❌ | ✅ **New** |
| **Contrastive Learning** | ❌ | ✅ **New** |
| **Multi-Task Learning** | ❌ | ✅ **New** |
| **Data Augmentation** | ❌ | ✅ **New** |

## Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/2025-ITEC0401/TS-CVA.git
cd TS-CVA

# Create conda environment
conda env create -f env.yaml
conda activate TS-CVA
```

### 2. Prepare Data

Download datasets from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) and place in `dataset/` folder.

```bash
# Generate LLM embeddings (GPT-2) for faster training
bash scripts/storage/Store_ETT.sh
```

### 3. Training

#### Option A: TS-CVA (Recommended)
Multi-task learning with automatic GPU selection:

```bash
# Train on ETTm1
bash scripts/tscva/TS-CVA_ETTm1.sh

# Train on ETTh1
bash scripts/tscva/TS-CVA_ETTh1.sh
```

#### Option B: Two-Stage Training
For better performance, pre-train TS2Vec encoder first:

```bash
# Stage 1: Pre-train TS2Vec with contrastive learning
bash scripts/pretrain/Pretrain_TS2Vec.sh

# Stage 2: Fine-tune end-to-end
python train_tscva.py \
  --data_path ETTm1 \
  --pred_len 96 \
  --load_pretrain ./checkpoints/ts2vec_pretrain/ETTm1/best_ts2vec_encoder.pth
```

#### Option C: Baseline (TimeCMA)
Compare with original TimeCMA:

```bash
bash scripts/baseline/ETTm1.sh
```

### 4. Custom Training

```bash
python train_tscva.py \
  --data_path ETTm1 \
  --seq_len 96 \
  --pred_len 96 \
  --batch_size 128 \
  --contrastive_weight 0.3 \
  --forecast_weight 0.7 \
  --learning_rate 1e-4 \
  --use_augmentation \
  --use_triple_align \
  --fusion_mode gated \
  --epochs 100
```

## Project Structure

```
TS-CVA/
├── models/
│   ├── TS_CVA.py              # Main TS-CVA model
│   ├── TS2Vec.py              # TS2Vec vector encoder
│   └── TimeCMA.py             # Baseline model
├── layers/
│   ├── Vector_Context_Align.py # Cross-modal alignment
│   └── Cross_Modal_Align.py    # Original TimeCMA alignment
├── utils/
│   ├── contrastive_loss.py    # Hierarchical contrastive loss
│   ├── augmentation.py        # Time series augmentation
│   └── metrics.py             # Evaluation metrics
├── data_provider/
│   └── data_loader_emb.py     # Data loading with embeddings
├── storage/
│   └── gen_prompt_emb.py      # LLM embedding generation
├── scripts/
│   ├── tscva/                 # TS-CVA training scripts
│   ├── baseline/              # TimeCMA baseline scripts
│   ├── pretrain/              # Pre-training scripts
│   ├── storage/               # Embedding storage scripts
│   └── utils/                 # Utility scripts (GPU selection)
├── train_tscva.py             # Main training script
├── pretrain_ts2vec.py         # Pre-training script
└── train.py                   # Baseline training script
```

## Training Scripts

### Main Scripts

- `train_tscva.py`: TS-CVA training with multi-task learning
- `pretrain_ts2vec.py`: TS2Vec encoder pre-training
- `train.py`: Original TimeCMA baseline

### Shell Scripts

**TS-CVA Scripts** (`scripts/tscva/`)
- `TS-CVA_ETTm1.sh`: Train on ETTm1 (4 horizons: 96, 192, 336, 720)
- `TS-CVA_ETTh1.sh`: Train on ETTh1 (4 horizons)
- Auto GPU selection enabled

**Pre-training Scripts** (`scripts/pretrain/`)
- `Pretrain_TS2Vec.sh`: Pre-train vector encoder on multiple datasets

**Baseline Scripts** (`scripts/baseline/`)
- `ETTm1.sh`, `ETTh1.sh`, etc: Original TimeCMA training

**Storage Scripts** (`scripts/storage/`)
- `Store_ETT.sh`: Generate and store LLM embeddings

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--contrastive_weight` | 0.3 | Weight for contrastive loss |
| `--forecast_weight` | 0.7 | Weight for forecasting loss |
| `--d_vector` | 320 | TS2Vec output dimension |
| `--use_triple_align` | True | Use triple-modal alignment |
| `--fusion_mode` | gated | Fusion strategy (gated/weighted/concat) |
| `--use_augmentation` | True | Enable data augmentation |
| `--pretrain_epochs` | 0 | TS2Vec pre-training epochs (0=disabled) |

## Supported Datasets

- **ETT** (Electricity Transformer Temperature): ETTh1, ETTh2, ETTm1, ETTm2
- **Weather**: Weather forecasting
- **ECL**: Electricity Consuming Load
- **FRED**: Federal Reserve Economic Data
- **ILI**: Influenza-Like Illness

## Development Setup

For development on lab server with automatic environment sync:

```bash
# Local: Sync code to server
bash sync_to_server.sh

# Server: Setup environment
bash setup_env.sh

# Server: Connect via SSH
bash ssh_connect.sh
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed development guide.

## Experimental Results

### ETTm1 Dataset (Prediction Length: 96)

| Model | MSE | MAE |
|-------|-----|-----|
| TimeCMA (baseline) | TBD | TBD |
| TS-CVA (ours) | TBD | TBD |

*Run experiments with `bash scripts/tscva/TS-CVA_ETTm1.sh` to populate results.*

## Ablation Study

Compare different configurations:

```bash
# 1. Baseline (TimeCMA)
bash scripts/baseline/ETTm1.sh

# 2. TS-CVA without contrastive learning
python train_tscva.py --data_path ETTm1 --contrastive_weight 0.0

# 3. TS-CVA full
bash scripts/tscva/TS-CVA_ETTm1.sh

# 4. TS-CVA with pre-training
bash scripts/pretrain/Pretrain_TS2Vec.sh
python train_tscva.py --data_path ETTm1 --load_pretrain <path>
```

## Citation

This work extends TimeCMA (AAAI 2025):

```bibtex
@inproceedings{liu2024timecma,
  title={{TimeCMA}: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment},
  author={Liu, Chenxi and Xu, Qianxiong and Miao, Hao and Yang, Sun and Zhang, Lingzheng and Long, Cheng and Li, Ziyue and Zhao, Rui},
  booktitle={AAAI},
  year={2025}
}
```

## License

This project is based on TimeCMA and uses the **S-Lab License 1.0**. See [LICENSE.txt](LICENSE.txt) for details.

TS2Vec integration follows the MIT License.

## Acknowledgments

- **TimeCMA**: Foundation framework for LLM-empowered time series forecasting
- **TS2Vec**: Contrastive learning approach for time series representations
- Lab Server: RTX 3090 x2 for training

## Contact

For questions or issues, please open an issue on GitHub.

---

**Project**: TS-CVA - Time Series Cross-modal Variable Alignment
**Course**: ITEC0401 Capstone Design (2025)
**Team**: 2025-ITEC0401
