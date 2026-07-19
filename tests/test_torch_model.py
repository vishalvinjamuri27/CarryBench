import unittest

import torch

from src import data
from src import tokenizer as tok
from src.torch_model import DecoderOnlyTransformer
from src.train_torch import run_generation_eval


def tiny_model(use_sdpa: bool = False):
    return DecoderOnlyTransformer(
        vocab_size=tok.VOCAB_SIZE,
        max_seq_len=data.SEQ_LEN - 1,
        d_model=16,
        n_layers=2,
        n_heads=2,
        d_ff=32,
        dropout=0.0,
        use_sdpa=use_sdpa,
    )


class TestTorchModel(unittest.TestCase):
    def test_sdpa_matches_manual_attention(self):
        torch.manual_seed(0)
        manual = tiny_model(False).eval()
        sdpa = tiny_model(True).eval()
        sdpa.load_state_dict(manual.state_dict())
        ids = torch.randint(0, tok.VOCAB_SIZE, (4, data.SEQ_LEN - 1))
        with torch.no_grad():
            expected = manual(ids)
            actual = sdpa(ids)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_generation_evaluator_runs(self):
        model = tiny_model()
        examples = data.generate_dataset(5, seed=0, split="eval")
        accuracy = run_generation_eval(model, examples, 4, torch.device("cpu"), data.DEFAULT_TASK)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
