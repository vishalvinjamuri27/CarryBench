"""Decoder-only transformer implemented directly in Flax Linen (no HF deps).

Architecture (standard GPT-style pre-norm decoder):
    token_embed + pos_embed
    -> [ LayerNorm -> causal self-attention -> residual
         LayerNorm -> MLP(GELU) -> residual ] x n_layers
    -> final LayerNorm -> LM head (tied-free linear projection to vocab)

Submodules are named explicitly (q_proj/k_proj/v_proj/out_proj, fc1/fc2,
block_{i}) so that `kv_cache_jax.py` can pull the exact same parameters out
of the trained `params` pytree and reuse them in a hand-written, cached
decoding loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads


def causal_mask(seq_len: int) -> jnp.ndarray:
    """(seq_len, seq_len) boolean mask, True where position t may attend to s<=t."""
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Q, K, V are each a learned linear projection of the input. Attention
    logits are masked so position t can only attend to positions <= t
    (causal/autoregressive constraint). This is the same Q/K/V computation
    that `kv_cache_jax.py` re-implements manually for cached decoding.
    """

    cfg: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        cfg = self.cfg
        B, T, D = x.shape
        H, hd = cfg.n_heads, cfg.head_dim

        q = nn.Dense(D, name="q_proj")(x)
        k = nn.Dense(D, name="k_proj")(x)
        v = nn.Dense(D, name="v_proj")(x)

        def split_heads(t):
            return t.reshape(B, T, H, hd).transpose(0, 2, 1, 3)  # (B, H, T, hd)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        attn_logits = jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(hd)
        mask = causal_mask(T)[None, None, :, :]
        attn_logits = jnp.where(mask, attn_logits, -1e9)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_weights = nn.Dropout(rate=cfg.dropout)(attn_weights, deterministic=deterministic)

        out = jnp.einsum("bhts,bhsd->bhtd", attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return nn.Dense(D, name="out_proj")(out)


class MLPBlock(nn.Module):
    """Two-layer feed-forward block with GELU activation."""

    cfg: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        cfg = self.cfg
        h = nn.Dense(cfg.d_ff, name="fc1")(x)
        h = jax.nn.gelu(h)
        h = nn.Dense(cfg.d_model, name="fc2")(h)
        h = nn.Dropout(rate=cfg.dropout)(h, deterministic=deterministic)
        return h


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> attn -> residual, LN -> MLP -> residual."""

    cfg: TransformerConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        cfg = self.cfg
        h = nn.LayerNorm(name="ln1")(x)
        x = x + CausalSelfAttention(cfg, name="attn")(h, deterministic)
        h = nn.LayerNorm(name="ln2")(x)
        x = x + MLPBlock(cfg, name="mlp")(h, deterministic)
        return x


class DecoderOnlyTransformer(nn.Module):
    """Full decoder-only LM: embeddings -> N blocks -> final LN -> LM head."""

    cfg: TransformerConfig

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        cfg = self.cfg
        B, T = input_ids.shape
        tok_emb = nn.Embed(cfg.vocab_size, cfg.d_model, name="token_embed")(input_ids)
        positions = jnp.arange(T)[None, :]
        pos_emb = nn.Embed(cfg.max_seq_len, cfg.d_model, name="pos_embed")(positions)
        x = tok_emb + pos_emb
        x = nn.Dropout(rate=cfg.dropout)(x, deterministic=deterministic)

        for i in range(cfg.n_layers):
            x = TransformerBlock(cfg, name=f"block_{i}")(x, deterministic)

        x = nn.LayerNorm(name="ln_f")(x)
        logits = nn.Dense(cfg.vocab_size, name="lm_head")(x)
        return logits
