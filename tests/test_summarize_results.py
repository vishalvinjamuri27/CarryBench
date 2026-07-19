import json
import tempfile
import unittest
from pathlib import Path

from src import summarize_results


class TestSummarizeResults(unittest.TestCase):
    def test_generated_accuracy_is_primary_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {
                "backend": "jax",
                "config": "demo_seed0",
                "seed": 0,
                "eval_generated_exact_match_accuracy": 0.75,
                "eval_teacher_forced_exact_match_accuracy": 1.0,
                "history": [{"step": 10, "eval_generated_exact_match_acc": 0.91}],
            }
            (root / "train_jax_demo_seed0.json").write_text(json.dumps(payload))
            summarize_results.summarize(root)
            table = (root / "summary_table.csv").read_text()
            self.assertIn("eval_generated_exact_match_accuracy", table)
            self.assertIn("0.75", table)
            self.assertIn(",10,", table)


if __name__ == "__main__":
    unittest.main()
