import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cleft.cleft_pipeline import CleftPipeline
from aligner.simalign_wrapper import SimAlignerWrapper


class TestMultiwordAndEquatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cleft_pipeline = CleftPipeline()
        cls.aligner = SimAlignerWrapper(matching_method="itermax")

    def test_multiword_mains_filter(self):
        en = "Mains filter capacitors are used in power supplies."
        ml = "മെയിൻസ് ഫിൽട്ടർ കപ്പാസിറ്ററുകൾ പവർ സപ്ലൈകളിൽ ഉപയോഗിക്കുന്നു."
        res = self.aligner.align(en, ml)
        
        # Source indices for "Mains filter capacitors": 0, 1, 2
        tgt_indices = [
            p["tgt_index"]
            for p in res["reconciled_pairs"]
            if p["src_index"] in (0, 1, 2) and p["is_lexical"]
        ]
        self.assertTrue(len(tgt_indices) >= 2, f"Should align multiple target tokens: {tgt_indices}")
        min_t, max_t = min(tgt_indices), max(tgt_indices)
        span = " ".join(res["target_tokens"][min_t : max_t + 1])
        self.assertIn("മെയിൻസ്", span)
        self.assertIn("കപ്പാസിറ്ററുകൾ", span)

    def test_multiword_low_cost(self):
        en = "Low cost options are preferred."
        ml = "കുറഞ്ഞ ചിലവിലുള്ള ഓപ്ഷനുകളാണ് മുൻഗണന നൽകുന്നത്."
        res = self.aligner.align(en, ml)
        
        tgt_indices = [
            p["tgt_index"]
            for p in res["reconciled_pairs"]
            if p["src_index"] in (0, 1) and p["is_lexical"]
        ]
        self.assertTrue(len(tgt_indices) >= 1)
        min_t, max_t = min(tgt_indices), max(tgt_indices)
        span = " ".join(res["target_tokens"][min_t : max_t + 1])
        self.assertIn("ചിലവിലുള്ള", span)

    def test_cleft_subject(self):
        # Regular verbal cleft
        tagged = "<FF>അവൻ<FF> പുസ്തകം വാങ്ങി."
        res = self.cleft_pipeline.process(tagged)
        self.assertEqual(res.status, "VALID")
        self.assertTrue(any(cop in res.copula_form for cop in ("ആണ്", "ാണ്", "യാണ്")))
        self.assertIn("വാങ്ങിയത്", res.cleft_sentence)

    def test_cleft_equative_locative(self):
        # Equative sentence (Moag §2.5, §5.4, §11.2)
        tagged = "എന്റെ വീട് <FF>കേരളത്തിൽ<FF> ആണ്."
        res = self.cleft_pipeline.process(tagged)
        self.assertIn(res.status, ("VALID", "ALLOWED"))
        self.assertTrue("കേരളത്തിലാണ്" in res.cleft_sentence or "കേരളത്തിൽ ആണ്" in res.cleft_sentence)


if __name__ == "__main__":
    unittest.main()
