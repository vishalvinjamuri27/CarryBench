"""Manual KV-cache decoding for the Flax decoder-only transformer.

WHAT ARE K AND V, AND WHY CACHE THEM
-------------------------------------
In each attention layer, every token position produces a "key" (K) and a
"value" (V) vector via learned linear projections of that token's hidden
state. To compute attention for the token at position t, the model compares
a "query" (Q) at t against the K vectors of every position 0..t, and uses
the resulting weights to combine the corresponding V vectors.

Critically, the K and V vectors for a given position only depend on that
position's hidden state, which is fixed once the token has been processed.
`generate_jax.generate_naive` ignores this and recomputes K/V for the
*entire* prefix on every decoding step (O(n_tokens * T^2) total work). This
module instead computes each token's K/V exactly once, stores them in a
fixed-size per-layer buffer (the "KV cache"), and on every new step only
computes Q/K/V for the *single new token*, appends its K/V to the cache,
and attends over the cache. That turns the per-step attention cost from
O(T) recomputation + O(T) attention into O(1) new K/V + O(T) attention
(the attention sum itself is unavoidable, but the expensive K/V projections
for old tokens are never redone).

DESIGN NOTE
-----------
Rather than relying on a Flax mutable-cache abstraction, this module reads
directly from the same parameter pytree produced by
`flax_model.DecoderOnlyTransformer` (token_embed, pos_embed,
block_i/{ln1,attn,ln2,mlp}, ln_f, lm_head) and recomputes the forward pass
by hand, layer by layer. This is more verbose than a framework-provided
cache, but it makes every step of cache reuse explicit and easy to verify
for correctness against the naive path (see tests/test_generation.py).
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from .flax_model import TransformerConfig


class KVCache(NamedTuple):
    """Per-layer K/V buffers, pre-allocated to `max_len` and filled in
    incrementally as tokens are decoded.

    k, v: (n_layers, batch, max_len, n_heads, head_dim)
    """

    k: jnp.ndarray
    v: jnp.ndarray


def init_cache(cfg: TransformerConfig, batch_size: int, max_len: int) -> KVCache:
    shape = (cfg.n_layers, batch_size, max_len, cfg.n_heads, cfg.head_dim)
    return KVCache(k=jnp.zeros(shape), v=jnp.zeros(shape))


def _layer_norm(x: jnp.ndarray, p: Dict, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * p["scale"] + p["bias"]


def _dense(x: jnp.ndarray, p: Dict) -> jnp.ndarray:
    return x @ p["kernel"] + p["bias"]


def _split_heads(x: jnp.ndarray, n_heads: int, head_dim: int) -> jnp.ndarray:
    """(B, T, D) -> (B, T, H, head_dim). Same grouping as flax_model's
    split_heads (just without the transpose, since these einsums contract
    over head_dim directly regardless of axis order)."""
    B, T, D = x.shape
    return x.reshape(B, T, n_heads, head_dim)


def prefill(
    model_params: Dict, cfg: TransformerConfig, input_ids: jnp.ndarray, cache: KVCache
) -> Tuple[jnp.ndarray, KVCache, int]:
    """Run the prompt through the model once, filling the cache for positions
    [0, P), where P = input_ids.shape[1].

    Returns (logits_at_last_prompt_position, updated_cache, P).
    """
    B, P = input_ids.shape
    H, hd = cfg.n_heads, cfg.head_dim

    tok_emb = model_params["token_embed"]["embedding"][input_ids]
    pos_emb = model_params["pos_embed"]["embedding"][jnp.arange(P)][None, :, :]
    x = tok_emb + pos_emb

    new_k_layers, new_v_layers = [], []
    for layer in range(cfg.n_layers):
        lp = model_params[f"block_{layer}"]
        h = _layer_norm(x, lp["ln1"])
        q = _split_heads(_dense(h, lp["attn"]["q_proj"]), H, hd)
        k = _split_heads(_dense(h, lp["attn"]["k_proj"]), H, hd)
        v = _split_heads(_dense(h, lp["attn"]["v_proj"]), H, hd)

        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k) / jnp.sqrt(hd)
        causal = jnp.tril(jnp.ones((P, P), dtype=bool))
        attn_logits = jnp.where(causal[None, None], attn_logits, -1e9)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum("bhts,bshd->bthd", attn_weights, v).reshape(B, P, cfg.d_model)
        attn_out = _dense(attn_out, lp["attn"]["out_proj"])

        x = x + attn_out
        h = _layer_norm(x, lp["ln2"])
        mlp_h = jax.nn.gelu(_dense(h, lp["mlp"]["fc1"]))
        mlp_out = _dense(mlp_h, lp["mlp"]["fc2"])
        x = x + mlp_out

        new_k_layers.append(k)
        new_v_layers.append(v)

    x = _layer_norm(x, model_params["ln_f"])
    logits = _dense(x, model_params["lm_head"])

    k_stack = jnp.stack(new_k_layers, axis=0)  # (n_layers, B, P, H, hd)
    v_stack = jnp.stack(new_v_layers, axis=0)
    k_full = jax.lax.dynamic_update_slice(cache.k, k_stack, (0, 0, 0, 0, 0))
    v_full = jax.lax.dynamic_update_slice(cache.v, v_stack, (0, 0, 0, 0, 0))
    return logits[:, -1, :], KVCache(k=k_full, v=v_full), P


def decode_step(
    model_params: Dict, cfg: TransformerConfig, last_token: jnp.ndarray, cache: KVCache, cur_len
) -> Tuple[jnp.ndarray, KVCache]:
    """Compute logits for one new token, reusing the cache filled up to `cur_len`.

    last_token: (B, 1) int32, the most recently generated/prompted token id.
    cur_len: the position index this new token occupies (scalar, may be a
             traced value when this function is jit-compiled -- the cache
             shape stays fixed across calls, only `cur_len` changes, so JAX
             only has to compile this function once no matter how many
             tokens are decoded).
    Returns (logits_for_new_token, updated_cache).
    """
    B = last_token.shape[0]
    H, hd = cfg.n_heads, cfg.head_dim
    max_len = cache.k.shape[2]

    tok_emb = model_params["token_embed"]["embedding"][last_token]  # (B, 1, D)
    pos_emb = model_params["pos_embed"]["embedding"][cur_len][None, None, :]
    x = tok_emb + pos_emb

    new_k_layers, new_v_layers = [], []
    for layer in range(cfg.n_layers):
        lp = model_params[f"block_{layer}"]
        h = _layer_norm(x, lp["ln1"])
        q = _split_heads(_dense(h, lp["attn"]["q_proj"]), H, hd)  # (B, 1, H, hd)
        k_new = _split_heads(_dense(h, lp["attn"]["k_proj"]), H, hd)
        v_new = _split_heads(_dense(h, lp["attn"]["v_proj"]), H, hd)

        # Write only the new token's K/V into the cache -- this is the whole
        # point: we never touch positions 0..cur_len-1 again.
        k_layer = jax.lax.dynamic_update_slice(cache.k[layer], k_new, (0, cur_len, 0, 0))
        v_layer = jax.lax.dynamic_update_slice(cache.v[layer], v_new, (0, cur_len, 0, 0))
        new_k_layers.append(k_layer)
        new_v_layers.append(v_layer)

        attn_logits = jnp.einsum("bthd,bshd->bhts", q, k_layer) / jnp.sqrt(hd)  # (B, H, 1, max_len)
        valid = jnp.arange(max_len) <= cur_len
        attn_logits = jnp.where(valid[None, None, None, :], attn_logits, -1e9)
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)
        attn_out = jnp.einsum("bhts,bshd->bthd", attn_weights, v_layer).reshape(B, 1, cfg.d_model)
        attn_out = _dense(attn_out, lp["attn"]["out_proj"])

        x = x + attn_out
        h = _layer_norm(x, lp["ln2"])
        mlp_h = jax.nn.gelu(_dense(h, lp["mlp"]["fc1"]))
        mlp_out = _dense(mlp_h, lp["mlp"]["fc2"])
        x = x + mlp_out

    x = _layer_norm(x, model_params["ln_f"])
    logits = _dense(x, model_params["lm_head"])  # (B, 1, V)

    updated_cache = KVCache(k=jnp.stack(new_k_layers, axis=0), v=jnp.stack(new_v_layers, axis=0))
    return logits[:, -1, :], updated_cache


def generate_with_kv_cache(
    model_params: Dict, cfg: TransformerConfig, prompt_ids: jnp.ndarray, n_tokens: int, max_len: int
) -> jnp.ndarray:
    """Greedy generation using the manual KV cache (reference implementation,
    not jitted -- see `make_jit_prefill`/`make_jit_decode_step` below for the
    jitted versions used in benchmarking).

    prompt_ids: (B, P) int32. Returns (B, P + n_tokens) int32.
    """
    B, P = prompt_ids.shape
    cache = init_cache(cfg, B, max_len)
    last_logits, cache, cur_len = prefill(model_params, cfg, prompt_ids, cache)
    next_id = jnp.argmax(last_logits, axis=-1)
    generated = [next_id]
    for _ in range(n_tokens - 1):
        logits, cache = decode_step(model_params, cfg, next_id[:, None], cache, cur_len)
        next_id = jnp.argmax(logits, axis=-1)
        generated.append(next_id)
        cur_len += 1
    return jnp.concatenate([prompt_ids, jnp.stack(generated, axis=1)], axis=1)


def make_jit_prefill(cfg: TransformerConfig):
    """Build a jitted prefill function bound to a fixed `cfg`. Shape stays
    fixed across calls with the same prompt length, so this compiles once."""

    def fn(model_params, input_ids, cache):
        return prefill(model_params, cfg, input_ids, cache)

    return jax.jit(fn)


def make_jit_decode_step(cfg: TransformerConfig):
    """Build a jitted decode-step function bound to a fixed `cfg`.

    Because the cache shape never changes (only the scalar `cur_len`
    changes), this compiles exactly once and is then reused for every
    subsequent decoded token -- the central efficiency claim of KV-cache
    decoding, made visible in the benchmark numbers.
    """

    def fn(model_params, last_token, cache, cur_len):
        return decode_step(model_params, cfg, last_token, cache, cur_len)

    return jax.jit(fn)
