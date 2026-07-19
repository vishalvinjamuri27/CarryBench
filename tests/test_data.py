import unittest

import numpy as np

from src import data


class TestData(unittest.TestCase):
    def test_make_example_fixed_width(self):
        ex = data.make_example(7, 8)
        self.assertEqual(ex.text, "007+008=0015")
        self.assertEqual(ex.prompt, "007+008=")
        self.assertEqual(ex.answer, "0015")
        self.assertEqual(ex.target_answer, "0015")
        self.assertEqual(len(ex.input_ids), data.SEQ_LEN)

    def test_make_example_four_digit_result(self):
        ex = data.make_example(999, 999)
        self.assertEqual(ex.answer, "1998")
        self.assertEqual(ex.target_answer, "1998")
        self.assertEqual(ex.text, "999+999=1998")

    def test_make_example_variable_digit_task(self):
        task = data.AdditionTask(operand_digits=5)
        ex = data.make_example(12345, 67890, task)
        self.assertEqual(ex.text, "12345+67890=080235")
        self.assertEqual(ex.prompt, "12345+67890=")
        self.assertEqual(ex.answer, "080235")
        self.assertEqual(ex.target_answer, "080235")
        self.assertEqual(len(ex.input_ids), task.seq_len)
        self.assertEqual(task.seq_len, 20)

    def test_make_example_reversed_answer_task(self):
        task = data.AdditionTask(operand_digits=5, answer_order="reversed")
        ex = data.make_example(12345, 67890, task)
        self.assertEqual(ex.answer, "080235")
        self.assertEqual(ex.target_answer, "532080")
        self.assertEqual(ex.text, "12345+67890=532080")
        self.assertEqual(len(ex.input_ids), task.seq_len)

    def test_make_example_rejects_out_of_range_operands(self):
        with self.assertRaises(ValueError):
            data.make_example(-1, 5)
        with self.assertRaises(ValueError):
            data.make_example(5, 1000)

    def test_generate_dataset_shapes_and_determinism(self):
        ds1 = data.generate_dataset(50, seed=42, split="train")
        ds2 = data.generate_dataset(50, seed=42, split="train")
        self.assertEqual(len(ds1), 50)
        self.assertEqual([e.a for e in ds1], [e.a for e in ds2])
        self.assertEqual([e.b for e in ds1], [e.b for e in ds2])

    def test_generate_dataset_splits_differ(self):
        train = data.generate_dataset(50, seed=42, split="train")
        eval_ = data.generate_dataset(50, seed=42, split="eval")
        train_pairs = {(e.a, e.b) for e in train}
        eval_pairs = {(e.a, e.b) for e in eval_}
        self.assertEqual(len(train_pairs & eval_pairs), 0)

    def test_standard_splits_are_disjoint_at_realistic_size(self):
        for seed in range(3):
            train = data.generate_dataset(20_000, seed=seed, split="train")
            eval_ = data.generate_dataset(2_000, seed=seed, split="eval")
            test = data.generate_dataset(2_000, seed=seed, split="test")
            sets = [{(e.a, e.b) for e in split} for split in (train, eval_, test)]
            self.assertFalse(sets[0] & sets[1])
            self.assertFalse(sets[0] & sets[2])
            self.assertFalse(sets[1] & sets[2])

    def test_impossible_unique_request_fails_fast(self):
        capacity = data._standard_split_capacity(100, "train", seed=0)
        with self.assertRaisesRegex(ValueError, "partition"):
            data.generate_dataset(capacity + 1, seed=0, split="train", max_operand=9)

    def test_standard_splits_cover_operand_space_without_bands(self):
        task = data.AdditionTask(operand_digits=3)
        for split in ("train", "eval", "test"):
            examples = data.generate_dataset(10_000, seed=0, split=split, task=task)
            operands = np.array([example.a for example in examples])
            self.assertLess(abs(float(operands.mean()) - task.operand_max / 2), 20)
            self.assertLess(int(operands.min()), 20)
            self.assertGreater(int(operands.max()), task.operand_max - 20)

    def test_generate_dataset_can_limit_operand_range(self):
        ds = data.generate_dataset(50, seed=0, split="train", max_operand=9, unique=False)
        self.assertTrue(all(0 <= ex.a <= 9 and 0 <= ex.b <= 9 for ex in ds))
        self.assertTrue(all(len(ex.input_ids) == data.SEQ_LEN for ex in ds))

    def test_generate_dataset_can_filter_no_carry(self):
        ds = data.generate_dataset(50, seed=0, split="train", require_carry=False, unique=False)
        self.assertTrue(all(not data.has_carry(ex.a, ex.b) for ex in ds))

    def test_generate_dataset_variable_digit_range(self):
        task = data.AdditionTask(operand_digits=6)
        ds = data.generate_dataset(20, seed=0, split="train", task=task, unique=False)
        self.assertEqual(len(ds), 20)
        self.assertTrue(all(0 <= ex.a <= 999999 and 0 <= ex.b <= 999999 for ex in ds))
        self.assertTrue(all(len(ex.input_ids) == task.seq_len for ex in ds))

    def test_curriculum_carry_mix_contains_carry_heavy_examples(self):
        ds = data.make_curriculum_dataset(
            100,
            seed=0,
            stage_name="carry_mix_test",
            sampling="carry_mix",
            carry_fraction=0.5,
        )
        self.assertEqual(len(ds), 100)
        n_high_digit_examples = 0
        for ex in ds:
            digits = f"{ex.a:03d}" + f"{ex.b:03d}"
            if all(int(ch) >= 5 for ch in digits):
                n_high_digit_examples += 1
        self.assertGreaterEqual(n_high_digit_examples, 40)

    def test_make_carry_heavy_dataset_has_high_digits(self):
        ds = data.make_carry_heavy_dataset(20, seed=0)
        self.assertEqual(len(ds), 20)
        for ex in ds:
            for ch in f"{ex.a:03d}" + f"{ex.b:03d}":
                self.assertGreaterEqual(int(ch), 5)

    def test_examples_to_array_shape(self):
        ds = data.generate_dataset(10, seed=0, split="train")
        arr = data.examples_to_array(ds)
        self.assertEqual(arr.shape, (10, data.SEQ_LEN))
        self.assertEqual(arr.dtype, np.int32)

    def test_batch_iterator_shapes(self):
        ds = data.generate_dataset(20, seed=0, split="train")
        batches = list(data.batch_iterator(ds, batch_size=8, shuffle=True, seed=1, drop_last=True))
        self.assertEqual(len(batches), 2)  # 20 // 8 = 2
        for b in batches:
            self.assertEqual(b.shape, (8, data.SEQ_LEN))

    def test_batch_iterator_keeps_partial_batch(self):
        ds = data.generate_dataset(20, seed=0, split="train")
        batches = list(data.batch_iterator(ds, batch_size=8, shuffle=False, drop_last=False))
        self.assertEqual([len(batch) for batch in batches], [8, 8, 4])

    def test_answer_loss_mask_selects_correct_positions(self):
        mask = data.answer_loss_mask()
        self.assertEqual(mask.sum(), data.RESULT_DIGITS)
        ex = data.make_example(7, 8)
        labels = np.array(ex.input_ids[1:])  # shifted labels
        answer_chars = "".join(str(c) for c in labels[mask] - 3)  # digit ids start at 3
        self.assertEqual(answer_chars, ex.answer)

    def test_answer_loss_mask_variable_digit_task(self):
        task = data.AdditionTask(operand_digits=5)
        mask = data.answer_loss_mask(task)
        self.assertEqual(mask.sum(), task.result_digits)
        ex = data.make_example(12345, 67890, task)
        labels = np.array(ex.input_ids[1:])
        answer_chars = "".join(str(c) for c in labels[mask] - 3)
        self.assertEqual(answer_chars, ex.target_answer)

    def test_answer_loss_mask_reversed_answer_task(self):
        task = data.AdditionTask(operand_digits=5, answer_order="reversed")
        mask = data.answer_loss_mask(task)
        ex = data.make_example(12345, 67890, task)
        labels = np.array(ex.input_ids[1:])
        answer_chars = "".join(str(c) for c in labels[mask] - 3)
        self.assertEqual(answer_chars, ex.target_answer)


if __name__ == "__main__":
    unittest.main()
