import tempfile
import unittest
from pathlib import Path

from src import utils


class TestUtils(unittest.TestCase):
    def test_invalid_config_reports_missing_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("d_model: 16\n")
            with self.assertRaisesRegex(ValueError, "Missing required config fields"):
                utils.load_config(str(path))

    def test_invalid_head_dimension_fails(self):
        cfg = {
            "d_model": 15,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 16,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0,
            "train_steps": 1,
            "eval_every": 1,
            "n_train": 2,
            "n_eval": 2,
            "n_carry_heavy": 2,
        }
        with self.assertRaisesRegex(ValueError, "divisible"):
            utils.validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
