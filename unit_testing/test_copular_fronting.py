"""
Unit tests for Malayalam copular / equational clefting and fronting inversion.
References: Rodney F. Moag, 'Malayalam: A Complete Textbook', Lessons 2.5, 5.4, 11.2.
"""

import unittest
from cleft.cleft_pipeline import CleftPipeline


class TestCopularClefting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = CleftPipeline()

    def test_predicate_focus_fronting_lawyer(self):
        """
        Input: അരുൺ <FF>അഭിഭാഷകനാണ്</FF>.
        Expected: അഭിഭാഷകനാണ് അരുൺ.
        (Predicate nominal focus fronted to clause-initial position)
        """
        res = self.pipeline.process("അരുൺ <FF>അഭിഭാഷകനാണ്</FF>.")
        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.cleft_sentence, "അഭിഭാഷകനാണ് അരുൺ.")

    def test_subject_focus_decopularization_arun(self):
        """
        Input: <FF>അരുൺ</FF> അഭിഭാഷകനാണ്.
        Expected: അരുണാണ് അഭിഭാഷകൻ.
        (Focus attached to subject; predicate nominal decopularized to prevent double copula)
        """
        res = self.pipeline.process("<FF>അരുൺ</FF> അഭിഭാഷകനാണ്.")
        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.cleft_sentence, "അരുണാണ് അഭിഭാഷകൻ.")

    def test_moag_predicate_focus_raman(self):
        """
        Moag §2.5 Example 3 & §5.4:
        Input: അവൻ <FF>രാമനാണ്</FF>.
        Expected: രാമനാണ് അവൻ.
        """
        res = self.pipeline.process("അവൻ <FF>രാമനാണ്</FF>.")
        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.cleft_sentence, "രാമനാണ് അവൻ.")

    def test_moag_subject_focus_avan(self):
        """
        Moag §2.5 Example 2:
        Input: <FF>അവൻ</FF> രാമനാണ്.
        Expected: അവനാണ് രാമൻ.
        """
        res = self.pipeline.process("<FF>അവൻ</FF> രാമനാണ്.")
        self.assertEqual(res.status, "VALID")
        self.assertEqual(res.cleft_sentence, "അവനാണ് രാമൻ.")


if __name__ == "__main__":
    unittest.main()
