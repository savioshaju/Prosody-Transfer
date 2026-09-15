"""
test_existential_cleft.py — Unit tests for existential ഉണ്ട് → ആണ് reduced cleft transformation.

Based on Tara Mohanan & K.P. Mohanan (1999): "Two Forms of 'Be' in Malayalam"
  - ഉണ്ട് (uNTE) = existential "be" → stripped during clefting
  - ആണ് (aaNE) = equative "be" / cleft marker → attached to focused constituent

Also tests copular predicate fronting for equational sentences.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from cleft.cleft_pipeline import CleftPipeline


class TestExistentialReducedCleft(unittest.TestCase):
    """Test Mohanan & Mohanan reduced cleft: ഉണ്ട് → ആണ്"""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = CleftPipeline()

    def test_existential_locative_focus(self):
        """പക്ഷെ അതിൽ പ്രശ്നമുണ്ട്. → പക്ഷെ അതിലാണ് പ്രശ്നം.
        Focus on locative 'അതിൽ', strip ഉണ്ട് from പ്രശ്നമുണ്ട് → പ്രശ്നം.
        """
        res = self.pipeline.process('പക്ഷെ <FF>അതിൽ</FF> പ്രശ്നമുണ്ട്.')
        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.cleft_sentence, "പക്ഷെ അതിലാണ് പ്രശ്നം.")

    def test_strip_existential_standalone(self):
        """ഉണ്ട് → '' (standalone)"""
        stripped = self.pipeline._strip_existential("ഉണ്ട്")
        self.assertEqual(stripped, "")

    def test_strip_existential_anusvara(self):
        """പ്രശ്നമുണ്ട് → പ്രശ്നം (anusvāra restoration)"""
        stripped = self.pipeline._strip_existential("പ്രശ്നമുണ്ട്")
        self.assertEqual(stripped, "പ്രശ്നം")

    def test_strip_existential_glide(self):
        """കുട്ടിയുണ്ട് → കുട്ടി (y-glide stripping)"""
        stripped = self.pipeline._strip_existential("കുട്ടിയുണ്ട്")
        self.assertEqual(stripped, "കുട്ടി")


class TestCopularPredicateFronting(unittest.TestCase):
    """Test copular predicate fronting for equational sentences."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = CleftPipeline()

    def test_football_predicate_fronting(self):
        """എനിക്ക് ഏറ്റവും ഇഷ്ടമുള്ള കളി കാൽപ്പന്താണ്. → കാൽപ്പന്താണ് എനിക്ക് ഏറ്റവും ഇഷ്ടമുള്ള കളി.
        Copular predicate 'കാൽപ്പന്താണ്' is fronted to clause-initial position.
        """
        res = self.pipeline.process('എനിക്ക് ഏറ്റവും ഇഷ്ടമുള്ള കളി <FF>കാൽപ്പന്താണ്</FF>.')
        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.cleft_sentence, "കാൽപ്പന്താണ് എനിക്ക് ഏറ്റവും ഇഷ്ടമുള്ള കളി.")

    def test_lawyer_subject_cleft(self):
        """<FF>അരുൺ</FF> അഭിഭാഷകനാണ്. → അരുണാണ് അഭിഭാഷകൻ.
        Subject focus in copular sentence: copula moves to focus, predicate decopularized.
        """
        res = self.pipeline.process('<FF>അരുൺ</FF> അഭിഭാഷകനാണ്.')
        self.assertEqual(res.status, "VALID")
        # ആണ് attaches to അരുൺ → അരുണാണ്, അഭിഭാഷകനാണ് decopularized → അഭിഭാഷകൻ
        self.assertIn("അരുണാണ്", res.cleft_sentence)


if __name__ == "__main__":
    unittest.main(verbosity=2)
