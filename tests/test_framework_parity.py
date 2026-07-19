import unittest

import jax
import jax.numpy as jnp
import numpy as np
import torch

from src import data
from src import tokenizer as tok
from src.flax_model import DecoderOnlyTransformer as FlaxTransformer
from src.flax_model import TransformerConfig
from src.torch_model import DecoderOnlyTransformer as TorchTransformer


def _copy_dense(torch_layer, flax_params):
    torch_layer.weight.data.copy_(torch.from_numpy(np.asarray(flax_params["kernel"]).T.copy()))
    torch_layer.bias.data.copy_(torch.from_numpy(np.asarray(flax_params["bias"]).copy()))


def _copy_layer_norm(torch_layer, flax_params):
    torch_layer.weight.data.copy_(torch.from_numpy(np.asarray(flax_params["scale"]).copy()))
    torch_layer.bias.data.copy_(torch.from_numpy(np.asarray(flax_params["bias"]).copy()))


def copy_flax_to_torch(params, model):
    model.token_embed.weight.data.copy_(
        torch.from_numpy(np.asarray(params["token_embed"]["embedding"]).copy())
    )
    model.pos_embed.weight.data.copy_(torch.from_numpy(np.asarray(params["pos_embed"]["embedding"]).copy()))
    for index, block in enumerate(model.blocks):
        source = params[f"block_{index}"]
        _copy_layer_norm(block.ln1, source["ln1"])
        _copy_layer_norm(block.ln2, source["ln2"])
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            _copy_dense(getattr(block.attn, name), source["attn"][name])
        _copy_dense(block.mlp.fc1, source["mlp"]["fc1"])
        _copy_dense(block.mlp.fc2, source["mlp"]["fc2"])
    _copy_layer_norm(model.ln_f, params["ln_f"])
    _copy_dense(model.lm_head, params["lm_head"])


class TestFrameworkParity(unittest.TestCase):
    def test_forward_logits_match_after_parameter_conversion(self):
        cfg = TransformerConfig(
            vocab_size=tok.VOCAB_SIZE,
            max_seq_len=data.SEQ_LEN - 1,
            d_model=16,
            n_layers=2,
            n_heads=2,
            d_ff=32,
        )
        flax_model = FlaxTransformer(cfg)
        ids = np.asarray([ex.input_ids[:-1] for ex in data.generate_dataset(4, 7, "eval")])
        variables = flax_model.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
            jnp.asarray(ids),
            deterministic=True,
        )
        torch_model = TorchTransformer(
            tok.VOCAB_SIZE,
            data.SEQ_LEN - 1,
            d_model=16,
            n_layers=2,
            n_heads=2,
            d_ff=32,
        ).eval()
        copy_flax_to_torch(variables["params"], torch_model)

        flax_logits = np.asarray(flax_model.apply(variables, jnp.asarray(ids), deterministic=True))
        with torch.no_grad():
            torch_logits = torch_model(torch.from_numpy(ids).long()).numpy()
        # CPU kernels agree more tightly, while independently implemented GPU
        # reductions can accumulate a few 1e-3 of floating-point error. Keep a
        # small absolute bound, which is especially important for logits near
        # zero and remains tight enough to detect parameter-mapping mistakes.
        np.testing.assert_allclose(torch_logits, flax_logits, rtol=2e-3, atol=3e-3)


if __name__ == "__main__":
    unittest.main()
