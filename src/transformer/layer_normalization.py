import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """Normalizes each token's embedding vector independently (Section 3.1).

    WHY WE NEED IT
    --------------
    As data flows through transformer layers, the values in each embedding
    vector can grow very large or very small — this is called "internal
    covariate shift". It makes training unstable (exploding/vanishing gradients)
    and slows convergence.

    Layer normalization fixes this by rescaling each token's vector to have
    mean=0 and std=1 after every sub-layer, keeping values in a stable range
    throughout the entire network depth.

    WHERE IT IS USED
    ----------------
    Applied after every sub-layer in both encoder and decoder using the
    "Add & Norm" pattern from Figure 1 of the paper:

        output = LayerNorm(x + SubLayer(x))

    Concretely it appears after:
      - Multi-head self-attention (encoder and decoder)
      - Cross-attention (decoder only)
      - Feed-forward network (encoder and decoder)

    HOW IT WORKS
    ------------
    For each token vector of length d_model:
      1. Compute mean and std across the d_model dimensions
      2. Subtract mean and divide by std  → normalized to mean=0, std=1
      3. Scale by alpha and shift by bias → learned rescaling so the model
         can recover expressive range if needed

    NOTE: Normalization is done across d_model (dim=-1), NOT across the batch
    or sequence. Each token is normalized independently using only its own
    d_model values — that's what makes it "layer" norm vs "batch" norm.
    """

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        # eps: small constant added to std to prevent division by zero when
        # std is very close to 0 (i.e., all values in the vector are identical)
        self.eps = eps

        # alpha and bias are learnable scalars (nn.Parameter = updated by optimizer)
        # Initialized to 1 and 0 so the initial output is just the normalized value.
        # The model can learn to scale/shift away from pure normalization if needed.
        self.alpha = nn.Parameter(torch.ones(1))   # multiplicative scale
        self.bias = nn.Parameter(torch.zeros(1))   # additive shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:      (batch, seq_len, d_model) — output of a sub-layer (attention or FFN)
        # output: (batch, seq_len, d_model) — same shape, values rescaled per token

        # Compute mean and std across the last dimension (d_model) for each token.
        # keepdim=True preserves the shape as (batch, seq_len, 1) so it broadcasts
        # correctly when subtracting/dividing against x of shape (batch, seq_len, d_model).
        mean = x.mean(dim=-1, keepdim=True)   # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True)     # (batch, seq_len, 1)

        # Normalize, then apply learned scale (alpha) and shift (bias).
        # eps is added to std to avoid division by zero.
        return self.alpha * (x - mean) / (std + self.eps) + self.bias