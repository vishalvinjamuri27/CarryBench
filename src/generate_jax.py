"""Naive autoregressive generation for the Flax decoder-only transformer.

This is the *un-optimized* decoding baseline: at every step the model
re-runs a full forward pass over the entire sequence generated so far,
recomputing every attention key/value pair from scratch even though most of
the prefix hasn't changed. Total work across decoding `n_tokens` is
O(n_tokens * T^2) in the sequence length T. `kv_cache_jax.py` implements the
optimized alternative (cache and reuse K/V instead of recomputing them) and
is benchmarked against this module.
"""

from __future__ import annotations

from typing import List

import jax.numpy as jnp

from . import data
from . import tokenizer as tok
from .flax_model import DecoderOnlyTransformer


def generate_naive(
    model: DecoderOnlyTransformer, params, prompt_ids: jnp.ndarray, n_tokens: int
) -> jnp.ndarray:
    """Greedily generate `n_tokens` new tokens after `prompt_ids`.

    prompt_ids: (B, P) int32 array.
    Returns: (B, P + n_tokens) int32 array (prompt followed by generated tokens).
    Decoding is greedy (argmax) for determinism -- there is no sampling
    randomness to control, which keeps the benchmark and tests reproducible.
    """
    ids = prompt_ids
    for _ in range(n_tokens):
        logits = model.apply({"params": params}, ids, deterministic=True)
        next_ids = jnp.argmax(logits[:, -1, :], axis=-1)
        ids = jnp.concatenate([ids, next_ids[:, None]], axis=1)
    return ids


def generate_one_with_eos(
    model: DecoderOnlyTransformer, params, prompt_ids: jnp.ndarray, max_new_tokens: int
) -> List[int]:
    """Single-sequence convenience wrapper that stops early at <eos>.

    prompt_ids: (1, P) int32 array.
    Returns the full sequence (prompt + generated ids) as a python list,
    stopping right after the first <eos> token if one is generated.
    """
    ids = prompt_ids
    for _ in range(max_new_tokens):
        logits = model.apply({"params": params}, ids, deterministic=True)
        next_id = int(jnp.argmax(logits[0, -1]))
        ids = jnp.concatenate([ids, jnp.array([[next_id]], dtype=ids.dtype)], axis=1)
        if next_id == tok.EOS_ID:
            break
    return [int(i) for i in ids[0]]


def evaluate_exact_match_via_generation(
    model: DecoderOnlyTransformer,
    params,
    examples: List[data.Example],
    batch_size: int = 64,
    task: data.AdditionTask = data.DEFAULT_TASK,
) -> float:
    """Exact-match accuracy when the answer is produced by actual generation
    (prompt -> greedy decode) rather than teacher-forced next-token argmax.

    This is the more honest accuracy metric: it never shows the model the
    correct previous answer digit, mirroring how the model is actually used.
    """
    correct = 0
    total = 0
    for batch in data.batch_iterator(examples, batch_size, shuffle=False, drop_last=True):
        prompt_ids = jnp.array(batch[:, : task.prompt_len])
        gen = generate_naive(model, params, prompt_ids, task.result_digits)
        pred_answer = gen[:, task.prompt_len : task.prompt_len + task.result_digits]
        true_answer = jnp.array(batch[:, task.answer_start : task.answer_end])
        matches = jnp.all(pred_answer == true_answer, axis=1)
        correct += int(jnp.sum(matches))
        total += batch.shape[0]
    return correct / total if total else float("nan")
