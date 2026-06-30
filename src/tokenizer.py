"""Character-level tokenizer for the addition workload.

The vocabulary is small and fixed on purpose: this is not a general-purpose
tokenizer, it is a deterministic char->id mapping for the digits 0-9, the
'+' and '=' symbols, and three special tokens (<pad>, <bos>, <eos>). Keeping
the vocab explicit (rather than learned/dynamic) avoids any tokenizer-induced
nondeterminism in the benchmark.
"""

from __future__ import annotations

from typing import List

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"

# Explicit, ordered vocabulary. Order matters: it fixes the integer ids.
_SPECIAL_TOKENS = [PAD, BOS, EOS]
_CHAR_TOKENS = list("0123456789") + ["+", "="]
VOCAB: List[str] = _SPECIAL_TOKENS + _CHAR_TOKENS

TOKEN_TO_ID = {tok: i for i, tok in enumerate(VOCAB)}
ID_TO_TOKEN = {i: tok for i, tok in enumerate(VOCAB)}

PAD_ID = TOKEN_TO_ID[PAD]
BOS_ID = TOKEN_TO_ID[BOS]
EOS_ID = TOKEN_TO_ID[EOS]

VOCAB_SIZE = len(VOCAB)


def encode(text: str) -> List[int]:
    """Encode a plain string of single-character tokens (no specials) to ids.

    Each character in `text` must be a single-char token in the vocab
    (digit, '+', or '='). Use `encode_with_specials` if you need <bos>/<eos>.
    """
    try:
        return [TOKEN_TO_ID[ch] for ch in text]
    except KeyError as e:
        raise ValueError(f"Character {e.args[0]!r} not in tokenizer vocab") from e


def decode(ids: List[int], strip_specials: bool = True) -> str:
    """Decode a list of ids back to a string.

    If `strip_specials` is True, <pad>/<bos>/<eos> tokens are dropped from the
    output rather than rendered as literal text.
    """
    chars = []
    for i in ids:
        tok = ID_TO_TOKEN[int(i)]
        if strip_specials and tok in _SPECIAL_TOKENS:
            continue
        chars.append(tok)
    return "".join(chars)


def encode_with_specials(text: str) -> List[int]:
    """Encode `text` wrapped in <bos> ... <eos>."""
    return [BOS_ID] + encode(text) + [EOS_ID]
