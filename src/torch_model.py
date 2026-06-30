"""PyTorch decoder-only transformer, architecturally matched to flax_model.py.

This is a comparison baseline only -- the JAX/Flax implementation is the
main subject of this project. The architecture mirrors flax_model.py as
closely as practical: token + learned positional embeddings, pre-norm
blocks of (LayerNorm -> causal self-attention -> residual, LayerNorm ->
GELU MLP -> residual), final LayerNorm, linear LM head. Implemented from
scratch (no `transformers` dependency).

Known minor differences from the Flax model (documented for honesty, not
hidden):
  * GELU: PyTorch's default `F.gelu` is the exact erf-based formulation;
    `jax.nn.gelu` defaults to the tanh approximation. Functionally very
    close, but not bit-identical.
  * Initialization: PyTorch's default `nn.Linear`/`nn.Embedding` init
    differs from Flax's default initializers. We do not try to force
    parity, since the point of the comparison is runtime behavior, not
    bit-for-bit numerical equivalence.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, hd).transpose(1, 2)  # (B, H, T, hd)
        k = self.k_proj(x).view(B, T, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, hd).transpose(1, 2)

        attn_logits = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        attn_logits = attn_logits.masked_fill(~causal_mask, float("-inf"))
        attn_weights = self.dropout(F.softmax(attn_logits, dim=-1))

        out = attn_weights @ v  # (B, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class MLPBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLPBlock(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
