# Transformer From Scratch

Implementing the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) using PyTorch.

## Project Structure

```
transformer-from-scratch/
├── src/
│   └── transformer/
│       ├── __init__.py
│       ├── attention.py        # Scaled dot-product & multi-head attention
│       ├── embeddings.py       # Token + positional embeddings
│       ├── encoder.py          # Encoder layer & encoder stack
│       ├── decoder.py          # Decoder layer & decoder stack
│       ├── feed_forward.py     # Position-wise feed-forward network
│       ├── model.py            # Full transformer model
│       └── utils.py            # Masks, padding helpers
├── tests/
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_embeddings.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_model.py
├── notebooks/                  # Exploratory notebooks
├── pyproject.toml
├── .python-version
└── README.md
```

## Setup

```bash
uv sync
```

## Run Tests

```bash
uv run pytest
```

## Paper Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017).
*Attention Is All You Need*. NeurIPS 2017. https://arxiv.org/abs/1706.03762
