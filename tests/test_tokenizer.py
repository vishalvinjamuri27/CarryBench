import unittest

from src import tokenizer as tok


class TestTokenizer(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        text = "007+008=0015"
        ids = tok.encode(text)
        self.assertEqual(tok.decode(ids), text)

    def test_encode_with_specials_round_trip(self):
        text = "999+999=1998"
        ids = tok.encode_with_specials(text)
        self.assertEqual(ids[0], tok.BOS_ID)
        self.assertEqual(ids[-1], tok.EOS_ID)
        self.assertEqual(tok.decode(ids, strip_specials=True), text)

    def test_vocab_is_explicit_and_fixed(self):
        self.assertEqual(tok.VOCAB_SIZE, len(tok.VOCAB))
        self.assertIn(tok.PAD, tok.TOKEN_TO_ID)
        self.assertIn(tok.BOS, tok.TOKEN_TO_ID)
        self.assertIn(tok.EOS, tok.TOKEN_TO_ID)
        for ch in "0123456789+=":
            self.assertIn(ch, tok.TOKEN_TO_ID)

    def test_unknown_character_raises(self):
        with self.assertRaises(ValueError):
            tok.encode("abc")

    def test_decode_strips_specials_by_default(self):
        ids = [tok.PAD_ID, tok.BOS_ID] + tok.encode("12+34=0046") + [tok.EOS_ID, tok.PAD_ID]
        self.assertEqual(tok.decode(ids), "12+34=0046")


if __name__ == "__main__":
    unittest.main()
