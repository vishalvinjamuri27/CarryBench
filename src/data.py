"""Synthetic addition dataset.

Examples are rendered as fixed-width strings so every sequence in a given
configured task has exactly the same length, e.g. the default 3-digit task:

    007+008=0015

By default operands are zero-padded to 3 digits (0-999), and the result is
zero-padded to 4 digits because 999+999=1998 requires 4 digits. Configs may
set `operand_digits` to run longer fixed-width tasks such as 5-digit
addition. With <bos>/<eos> wrapping, every full sequence in that task has the
same length, so no padding is needed for training batches. (The PAD token
still exists in the vocab and is used when encoding prompts of a different
length for generation.)

Layout of token indices in a full sequence:

    [0]        <bos>
    [1:4]      a   (3 digits)
    [4]        '+'
    [5:8]      b   (3 digits)
    [8]        '='
    [9:13]     result (4 digits, zero-padded)
    [13]       <eos>
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np

from . import tokenizer as tok


@dataclass(frozen=True)
class AdditionTask:
    operand_digits: int = 3
    answer_order: str = "normal"

    def __post_init__(self):
        if self.answer_order not in {"normal", "reversed"}:
            raise ValueError(f"answer_order must be 'normal' or 'reversed', got {self.answer_order!r}")

    @property
    def result_digits(self) -> int:
        return self.operand_digits + 1

    @property
    def operand_max(self) -> int:
        return 10**self.operand_digits - 1

    @property
    def prompt_len(self) -> int:
        return 1 + self.operand_digits + 1 + self.operand_digits + 1

    @property
    def answer_len(self) -> int:
        return self.result_digits + 1

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.answer_len

    @property
    def answer_start(self) -> int:
        return self.prompt_len

    @property
    def answer_end(self) -> int:
        return self.prompt_len + self.result_digits


DEFAULT_TASK = AdditionTask()

OPERAND_DIGITS = DEFAULT_TASK.operand_digits
RESULT_DIGITS = DEFAULT_TASK.result_digits
OPERAND_MAX = DEFAULT_TASK.operand_max

PROMPT_LEN = DEFAULT_TASK.prompt_len
ANSWER_LEN = DEFAULT_TASK.answer_len
SEQ_LEN = DEFAULT_TASK.seq_len

# Index (into the full SEQ_LEN sequence) where the answer digits start/end.
ANSWER_START = PROMPT_LEN
ANSWER_END = PROMPT_LEN + RESULT_DIGITS  # exclusive, does not include <eos>


@dataclass
class Example:
    a: int
    b: int
    text: str  # e.g. "007+008=0015" (no specials)
    prompt: str  # e.g. "007+008=" (no specials)
    answer: str  # normal arithmetic answer, e.g. "0015"
    target_answer: str  # answer as rendered in the sequence (possibly reversed)
    input_ids: List[int]  # full sequence incl. <bos>/<eos>, length SEQ_LEN


def task_from_config(cfg: dict) -> AdditionTask:
    digits = int(cfg.get("operand_digits", OPERAND_DIGITS))
    if digits <= 0:
        raise ValueError(f"operand_digits must be positive, got {digits}")
    return AdditionTask(operand_digits=digits, answer_order=cfg.get("answer_order", "normal"))


def make_example(a: int, b: int, task: AdditionTask = DEFAULT_TASK) -> Example:
    """Build a single fixed-width addition example for operands a, b."""
    if not (0 <= a <= task.operand_max and 0 <= b <= task.operand_max):
        raise ValueError(f"operands must be in [0, {task.operand_max}], got a={a}, b={b}")
    result = a + b
    prompt = f"{a:0{task.operand_digits}d}+{b:0{task.operand_digits}d}="
    answer = f"{result:0{task.result_digits}d}"
    target_answer = answer[::-1] if task.answer_order == "reversed" else answer
    text = prompt + target_answer
    input_ids = tok.encode_with_specials(text)
    assert len(input_ids) == task.seq_len, (len(input_ids), task.seq_len)
    return Example(
        a=a,
        b=b,
        text=text,
        prompt=prompt,
        answer=answer,
        target_answer=target_answer,
        input_ids=input_ids,
    )


def _split_seed(seed: int, split: str) -> int:
    """Derive a distinct seed per split from one base seed."""
    offset = {"train": 0, "eval": 1, "test": 2}.get(split)
    if offset is None:
        offset = 3 + _stable_stage_offset(split)
    return seed * 10_000 + offset


_SPLIT_RANGES = {
    "train": (0, 80),
    "eval": (80, 90),
    "test": (90, 100),
}


def _coprime_multiplier(size: int, seed: int) -> int:
    """Return a deterministic multiplier defining a permutation modulo size."""
    if size <= 1:
        return 1
    candidate = (2 * abs(seed) + 1) % size
    candidate = candidate or 1
    while math.gcd(candidate, size) != 1:
        candidate = (candidate + 2) % size
        candidate = candidate or 1
    return candidate


def _split_index_bounds(size: int, split: str) -> Tuple[int, int]:
    """Return an exact, non-overlapping range for a standard dataset split."""
    lo_pct, hi_pct = _SPLIT_RANGES[split]
    return size * lo_pct // 100, size * hi_pct // 100


def count_carries(a: int, b: int, operand_digits: int = OPERAND_DIGITS) -> int:
    """Return how many digit positions produce a carry when adding a + b."""
    carries = 0
    carry = 0
    for _ in range(operand_digits):
        digit_sum = (a % 10) + (b % 10) + carry
        carry = 1 if digit_sum >= 10 else 0
        carries += carry
        a //= 10
        b //= 10
    return carries


def has_carry(a: int, b: int, task: AdditionTask = DEFAULT_TASK) -> bool:
    """Whether adding a + b has at least one carry in the fixed-width task."""
    return count_carries(a, b, task.operand_digits) > 0


def generate_dataset(
    n: int,
    seed: int,
    split: str = "train",
    max_operand: Optional[int] = None,
    require_carry: Optional[bool] = None,
    unique: bool = True,
    task: AdditionTask = DEFAULT_TASK,
) -> List[Example]:
    """Generate `n` addition examples deterministically for a given split.

    Standard train/eval/test splits are disjoint by construction. Pair ids are
    passed through a seeded affine permutation and each split receives a fixed
    80/10/10 slice of that permutation. Other named splits use independent
    deterministic sampling and are intended for diagnostic datasets.
    """
    if max_operand is None:
        max_operand = task.operand_max
    if not (0 <= max_operand <= task.operand_max):
        raise ValueError(f"max_operand must be in [0, {task.operand_max}], got {max_operand}")

    side = max_operand + 1
    pair_space = side * side
    rng = random.Random(_split_seed(seed, split))

    split_bounds = _split_index_bounds(pair_space, split) if split in _SPLIT_RANGES else None
    if unique and split_bounds is not None:
        split_capacity = split_bounds[1] - split_bounds[0]
        if n > split_capacity:
            raise ValueError(
                f"Requested {n} unique {split} examples, but its disjoint partition "
                f"contains only {split_capacity} pairs for max_operand={max_operand}"
            )
    elif unique and n > pair_space:
        raise ValueError(f"Requested {n} unique examples from a space of only {pair_space} pairs")

    multiplier = _coprime_multiplier(pair_space, seed)
    offset = (seed * 0x9E3779B1 + 0x85EBCA77) % pair_space

    def draw_pair() -> Tuple[int, int]:
        if split_bounds is None:
            pair_id = rng.randrange(pair_space)
        else:
            pair_id = rng.randrange(*split_bounds)
        permuted = (multiplier * pair_id + offset) % pair_space
        return divmod(permuted, side)

    examples = []
    seen = set()
    attempts = 0
    max_attempts = max(10_000, n * 10_000)
    while len(examples) < n:
        attempts += 1
        if attempts > max_attempts:
            raise ValueError(
                "Unable to satisfy dataset constraints; reduce n, disable uniqueness, "
                "or relax the carry filter"
            )
        a, b = draw_pair()
        if unique and (a, b) in seen:
            continue
        if require_carry is not None and has_carry(a, b, task) != require_carry:
            continue
        if unique:
            seen.add((a, b))
        examples.append(make_example(a, b, task))
    return examples


def make_curriculum_dataset(
    n: int,
    seed: int,
    stage_name: str,
    max_operand: Optional[int] = None,
    sampling: str = "random",
    carry_fraction: float = 0.5,
    task: AdditionTask = DEFAULT_TASK,
) -> List[Example]:
    """Generate deterministic examples for one curriculum stage.

    The text format stays fixed-width for every stage; only the sampled
    operand distribution changes. That keeps JAX shapes stable while making
    the learning problem progress from easy to harder cases.
    """
    if max_operand is None:
        max_operand = task.operand_max
    split = f"curriculum_{stage_name}"
    if sampling == "random":
        return generate_dataset(n, seed, split, max_operand=max_operand, unique=False, task=task)
    if sampling == "no_carry":
        return generate_dataset(
            n, seed, split, max_operand=max_operand, require_carry=False, unique=False, task=task
        )
    if sampling == "carry":
        return generate_dataset(
            n, seed, split, max_operand=max_operand, require_carry=True, unique=False, task=task
        )
    if sampling == "carry_heavy":
        return make_carry_heavy_dataset(n, seed + _stable_stage_offset(stage_name), unique=False, task=task)
    if sampling == "carry_mix":
        if not (0.0 <= carry_fraction <= 1.0):
            raise ValueError(f"carry_fraction must be in [0, 1], got {carry_fraction}")
        n_carry = int(round(n * carry_fraction))
        n_random = n - n_carry
        random_examples = generate_dataset(
            n_random, seed, split, max_operand=max_operand, unique=False, task=task
        )
        carry_examples = make_carry_heavy_dataset(
            n_carry, seed + _stable_stage_offset(stage_name), unique=False, task=task
        )
        examples = random_examples + carry_examples
        rng = random.Random(_split_seed(seed, split) + 17)
        rng.shuffle(examples)
        return examples
    raise ValueError(f"Unknown curriculum sampling mode {sampling!r}")


def _stable_stage_offset(stage_name: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(stage_name))


def make_carry_heavy_dataset(
    n: int, seed: int, unique: bool = True, task: AdditionTask = DEFAULT_TASK
) -> List[Example]:
    """Generate examples biased toward heavy digit-carrying additions.

    We bias each digit of `a` and `b` to be large (5-9) so that most digit
    positions overflow into a carry. This stresses the model on the
    hardest sub-case of addition (long carry chains), used as a separate
    held-out evaluation set.
    """
    rng = random.Random(seed * 10_000 + 999)
    examples = []
    seen = set()
    while len(examples) < n:
        a_digits = [rng.randint(5, 9) for _ in range(task.operand_digits)]
        b_digits = [rng.randint(5, 9) for _ in range(task.operand_digits)]
        a = int("".join(map(str, a_digits)))
        b = int("".join(map(str, b_digits)))
        if unique and (a, b) in seen:
            continue
        if unique:
            seen.add((a, b))
        examples.append(make_example(a, b, task))
    return examples


def examples_to_array(examples: List[Example]) -> np.ndarray:
    """Stack examples into a (n, SEQ_LEN) int32 array of token ids."""
    return np.array([ex.input_ids for ex in examples], dtype=np.int32)


def batch_iterator(
    examples: List[Example],
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> Iterator[np.ndarray]:
    """Yield (batch_size, SEQ_LEN) int32 arrays of token ids.

    Since every example is exactly SEQ_LEN tokens, no padding is needed
    within a batch.
    """
    idx = np.arange(len(examples))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
    n_batches = len(idx) // batch_size if drop_last else -(-len(idx) // batch_size)
    arr = examples_to_array(examples)
    for i in range(n_batches):
        batch_idx = idx[i * batch_size : (i + 1) * batch_size]
        yield arr[batch_idx]


def split_inputs_labels(batch_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a (batch, SEQ_LEN) array into causal LM inputs/labels.

    inputs = batch_ids[:, :-1], labels = batch_ids[:, 1:]
    Predicting labels[:, t] from inputs[:, :t+1].
    """
    return batch_ids[:, :-1], batch_ids[:, 1:]


def answer_loss_mask(task: AdditionTask = DEFAULT_TASK) -> np.ndarray:
    """Boolean mask (length SEQ_LEN - 1) selecting label positions that are
    answer digits (used for exact-match eval and optional loss reweighting).

    Labels are batch_ids[:, 1:], i.e. label index j corresponds to original
    sequence index j + 1. Answer digits occupy original indices
    [ANSWER_START, ANSWER_END), so the corresponding label indices are
    [ANSWER_START - 1, ANSWER_END - 1).
    """
    mask = np.zeros(task.seq_len - 1, dtype=bool)
    mask[task.answer_start - 1 : task.answer_end - 1] = True
    return mask
