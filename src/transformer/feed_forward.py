# Position-wise feed-forward network
# Reference: Section 3.3 of "Attention Is All You Need"
import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """Two-layer fully connected network applied independently to each token (Section 3.3).

    WHY WE NEED IT
    --------------
    Attention is good at mixing information *across* tokens — figuring out which
    tokens should influence each other. But it has no mechanism to transform the
    *content* of each token's representation deeply on its own.

    The feed-forward block fills that gap: after attention has gathered context
    from other tokens, the FFN processes each token independently to extract
    richer features. It's where most of the model's "thinking" per token happens.

    The expansion to d_ff (typically 4x d_model, e.g. 512 → 2048) gives the
    model a larger intermediate space to compute complex non-linear transformations,
    before projecting back down to d_model.

    WHERE IT IS USED
    ----------------
    Applied once per encoder layer and once per decoder layer, after the
    attention sub-layer, using the "Add & Norm" pattern:

        output = LayerNorm(x + FeedForward(x))

    With 6 encoder + 6 decoder layers in the base model, there are 12 FFN
    blocks in total.

    HOW IT WORKS
    ------------
    For each token independently:
      1. linear_1: project d_model → d_ff         (expand)
      2. ReLU: apply non-linearity
      3. dropout: randomly zero some activations   (regularization during training)
      4. linear_2: project d_ff → d_model          (compress back)

    "Position-wise" means the same two linear layers are applied to every token
    position — the weights are shared across positions but each token is
    transformed independently (no information flows between tokens here).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        # linear_1: expands each token vector from d_model to d_ff dimensions.
        # Typical values: d_model=512, d_ff=2048 (4x expansion from the paper).
        self.linear_1 = nn.Linear(d_model, d_ff)   # (d_model → d_ff)
        self.dropout = nn.Dropout(dropout)
        # linear_2: projects back down to d_model so output shape matches input.
        # This lets the FFN slot into the residual connection: x + FFN(x).
        self.linear_2 = nn.Linear(d_ff, d_model)   # (d_ff → d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:      (batch, seq_len, d_model) — token representations after attention
        # output: (batch, seq_len, d_model) — same shape, richer per-token features
        #
        # Data flow per token vector:
        #   (d_model) → linear_1 → (d_ff) → ReLU → dropout → linear_2 → (d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
