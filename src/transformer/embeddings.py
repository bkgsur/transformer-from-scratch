# Token embeddings and positional encoding
# Reference: Section 3.4 and 3.5 of "Attention Is All You Need"
import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """Maps token IDs to dense vectors of size d_model (Section 3.4).

    A learned lookup table converts each integer token ID into a continuous
    vector the model can reason over. The output is scaled by sqrt(d_model)
    to keep embedding magnitudes in a similar range to the positional encodings
    that will be added on top — preventing positional signal from being drowned
    out by large embedding values.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Learnable table of shape (vocab_size, d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) integer token IDs
        # output: (batch, seq_len, d_model) — scaled so magnitudes are
        # compatible with the positional encodings added in the next step.
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):
    """Injects position information into token embeddings (Section 3.5).

    Transformers process all tokens in parallel and have no built-in sense of
    order. This module adds a fixed sinusoidal signal to each token embedding
    so the model can distinguish positions.

    Each position gets a unique fingerprint across d_model dimensions:
    - Low dimensions oscillate quickly  → fine-grained position signal
    - High dimensions oscillate slowly  → coarse-grained position signal

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout)

        # Blank encoding table: (max_seq_len, d_model)
        # One row per position, one column per embedding dimension.
        pe = torch.zeros(max_seq_len, d_model)

        # Column vector of position indices: [0, 1, 2, ..., max_seq_len-1]
        # Shape (max_seq_len, 1) so it broadcasts against div_term below.
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Computes 1 / 10000^(2i/d_model) for each even dimension i.
        # Done in log-space (exp(log(...))) for numerical stability.
        #
        # Only d_model/2 frequencies are needed because sin and cos SHARE the
        # same frequency per pair of dimensions (2i, 2i+1). arange(0,d_model,2)
        # produces [0, 2, 4, ..., d_model-2] — one value per pair, no duplicates.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Both assignments reuse the same div_term (same frequencies):
        # even cols get sin, odd cols get cos — paired by shared frequency.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension so pe broadcasts over any batch size.
        # Shape: (1, max_seq_len, d_model)
        # InputEmbeddings output is (batch, seq_len, d_model); pe is
        # (1, seq_len, d_model). PyTorch broadcasting expands the 1 to match
        # any batch size, so no explicit batch handling is needed here.
        pe = pe.unsqueeze(0)

        # register_buffer: pe moves with the model (e.g. to GPU) but is NOT
        # a trainable parameter — positional encodings are fixed, not learned.
        self.register_buffer('pe', pe)
    # x: (batch, seq_len, d_model)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        # Slice the precomputed table to the current sequence length and add
        # to token embeddings. requires_grad_(False) ensures no gradient flows
        # through the positional encoding (it is fixed).
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
 
