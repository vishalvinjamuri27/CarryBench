"""Loss and accuracy metrics shared by the JAX training/eval loops.

Two accuracy notions are tracked, by design:
  * token_accuracy: argmax-correct fraction over all non-pad label positions
    (the standard causal LM metric).
  * exact_match_accuracy: fraction of examples whose *entire* answer
    (all 4 result digits) is predicted correctly -- the metric that actually
    matters for "did the model get the addition right".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean token-level cross-entropy over positions where mask is True.

    logits: (B, T, V) unnormalized scores.
    labels: (B, T) int target ids.
    mask:   (B, T) bool/float, 1 where the position contributes to the loss.
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    mask = mask.astype(nll.dtype)
    return jnp.sum(nll * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def token_accuracy(logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Fraction of masked positions where argmax(logits) == labels."""
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == labels).astype(jnp.float32)
    mask = mask.astype(jnp.float32)
    return jnp.sum(correct * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def exact_match_accuracy(logits: jnp.ndarray, labels: jnp.ndarray, answer_mask: jnp.ndarray) -> jnp.ndarray:
    """Fraction of rows whose full answer span is predicted exactly right.

    answer_mask: (T,) bool, True at label positions belonging to the answer
    digits (see data.answer_loss_mask). Broadcast over the batch.
    """
    preds = jnp.argmax(logits, axis=-1)  # (B, T)
    answer_mask = answer_mask[None, :]
    matches = jnp.where(answer_mask, preds == labels, True)  # ignore non-answer positions
    row_exact = jnp.all(matches, axis=-1)
    return jnp.mean(row_exact.astype(jnp.float32))


def exact_match_accuracy_np(pred_ids: np.ndarray, label_ids: np.ndarray, answer_mask: np.ndarray) -> float:
    """Numpy equivalent of exact_match_accuracy, for use outside jit (e.g. PyTorch)."""
    preds = pred_ids[:, answer_mask]
    labels = label_ids[:, answer_mask]
    row_match = np.all(preds == labels, axis=1)
    return float(np.mean(row_match))
