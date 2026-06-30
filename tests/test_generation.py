import unittest

import jax
import jax.numpy as jnp

from src import data
from src import generate_jax as gen
from src import kv_cache_jax as kv
from src import tokenizer as tok
from src.flax_model import DecoderOnlyTransformer, TransformerConfig


def _make_tiny_model():
    cfg = TransformerConfig(
        vocab_size=tok.VOCAB_SIZE, max_seq_len=data.SEQ_LEN - 1, d_model=16, n_layers=2, n_heads=2, d_ff=32
    )
    model = DecoderOnlyTransformer(cfg)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1, data.SEQ_LEN - 1), dtype=jnp.int32)
    variables = model.init({"params": rng, "dropout": rng}, dummy, deterministic=True)
    return model, variables["params"], cfg


class TestGeneration(unittest.TestCase):
    def setUp(self):
        self.model, self.params, self.cfg = _make_tiny_model()
        self.examples = data.generate_dataset(4, seed=0, split="train")
        self.prompt_ids = jnp.array([ex.input_ids[: data.PROMPT_LEN] for ex in self.examples])

    def test_naive_generation_returns_valid_tokens(self):
        n_tokens = data.RESULT_DIGITS + 1
        out = gen.generate_naive(self.model, self.params, self.prompt_ids, n_tokens)
        self.assertEqual(out.shape, (4, data.PROMPT_LEN + n_tokens))
        self.assertTrue(bool(jnp.all((out >= 0) & (out < tok.VOCAB_SIZE))))

    def test_kv_cache_generation_matches_naive_shape_and_output(self):
        n_tokens = data.RESULT_DIGITS + 1
        naive_out = gen.generate_naive(self.model, self.params, self.prompt_ids, n_tokens)
        kv_out = kv.generate_with_kv_cache(self.params, self.cfg, self.prompt_ids, n_tokens, max_len=data.SEQ_LEN)

        self.assertEqual(naive_out.shape, kv_out.shape)
        self.assertTrue(bool(jnp.all(naive_out == kv_out)), "KV-cache decoding must match naive decoding exactly")

    def test_jitted_kv_cache_matches_reference(self):
        n_tokens = data.RESULT_DIGITS + 1
        ref_out = kv.generate_with_kv_cache(self.params, self.cfg, self.prompt_ids, n_tokens, max_len=data.SEQ_LEN)

        jit_prefill = kv.make_jit_prefill(self.cfg)
        jit_decode = kv.make_jit_decode_step(self.cfg)
        cache = kv.init_cache(self.cfg, self.prompt_ids.shape[0], data.SEQ_LEN)
        last_logits, cache, cur_len = jit_prefill(self.params, self.prompt_ids, cache)
        next_id = jnp.argmax(last_logits, axis=-1)
        generated = [next_id]
        for _ in range(n_tokens - 1):
            logits, cache = jit_decode(self.params, next_id[:, None], cache, cur_len)
            next_id = jnp.argmax(logits, axis=-1)
            generated.append(next_id)
            cur_len += 1
        jit_out = jnp.concatenate([self.prompt_ids, jnp.stack(generated, axis=1)], axis=1)

        self.assertTrue(bool(jnp.all(ref_out == jit_out)))

    def test_evaluate_exact_match_via_generation_runs(self):
        acc = gen.evaluate_exact_match_via_generation(self.model, self.params, self.examples, batch_size=4)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
