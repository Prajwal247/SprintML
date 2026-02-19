# SprintML

Comparing prompt-learning and parameter-efficient fine-tuning methods on **TinyBERT** (`prajjwal1/bert-tiny`) across four GLUE benchmark tasks, with **differential privacy** (Opacus / manual Gaussian mechanism).

## Methods

| # | Method | What is trained |
|---|--------|-----------------|
| 1 | Soft Prompts | Learnable continuous embeddings prepended to inputs |
| 2 | Prefix Tuning | Learnable prefix vectors prepended to each layer's input |
| 3 | Full Fine-tuning | All model parameters |
| 4 | LoRA | Low-rank adapters on Q/V attention matrices |
| 5 | Single Layer | One classification head on frozen BERT |
| 6 | Soft Prompt + LoRA | Combines 1 and 4 |
| 7 | Prefix + LoRA | Combines 2 and 4 |

## Datasets

SST-2, QNLI, MNLI, and QQP from the GLUE benchmark.

## Project Structure

```
SprintML/
├── train.py              # Main entry point
├── config.py             # Hyperparameters, DP settings, task info
├── utils.py              # Seed, device, parameter counting
├── data/
│   ├── __init__.py
│   └── dataset.py        # GLUE loading and tokenization
├── models/
│   ├── __init__.py
│   ├── factory.py        # Model creation dispatcher
│   ├── soft_prompt.py
│   ├── prefix_tuning.py
│   ├── lora.py
│   ├── single_layer.py
│   ├── soft_prompt_lora.py
│   └── prefix_lora.py
├── training/
│   ├── __init__.py
│   ├── trainer.py        # Training loop with DP
│   └── dp_utils.py       # Opacus-compatible LayerNorm, model fixes
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py
```

Results are saved to `results_epsilon_8.csv`.

## Configuration

Edit `config.py` to change:

- **Model**: `MODEL_NAME` (default: `prajjwal1/bert-tiny`)
- **Privacy budget**: `EPSILON`, `DELTA`, `MAX_GRAD_NORM`
- **Methods / datasets to run**: `METHODS`, `DATASETS`
- **Per-method hyperparameters**: `HYPERPARAMS`
