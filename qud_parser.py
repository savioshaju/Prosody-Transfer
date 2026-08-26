"""
qud_parser.py — Standalone Question Under Discussion (QUD) Parser.

Implements formal discourse pragmatics (Roberts 1996/2012 framework):
  1. Explicit QUD parsing from dialogue context
  2. Implicit QUD derivation from sentence structure & focus constituent
  3. QUD Wh-Category classification (TEMPORAL, SPATIAL, SUBJECT, OBJECT, MANNER, REASON, POLAR)
  4. Focus-QUD alignment & Background presupposition extraction
"""

import re
import sys
from typing import Dict, Any, List, Optional

# Ensure UTF-8 console output for Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


class QUDParser:
    """
    Question Under Discussion (QUD) Parser for Discourse Analysis & Prosody Transfer.
    """

    WH_PATTERNS = [
        (re.compile(r"\b(when|what time|which day|date)\b", re.IGNORECASE), "TEMPORAL", "When"),
        (re.compile(r"\b(where|which place|location)\b", re.IGNORECASE), "SPATIAL", "Where"),
        (re.compile(r"\b(who|whom|whose)\b", re.IGNORECASE), "SUBJECT_AGENT", "Who"),
        (re.compile(r"\b(what|which)\b", re.IGNORECASE), "OBJECT_PATIENT", "What"),
        (re.compile(r"\b(why|reason|because|how come)\b", re.IGNORECASE), "CAUSAL", "Why"),
        (re.compile(r"\b(how|by what means)\b", re.IGNORECASE), "MANNER", "How"),
    ]

    TEMPORAL_KEYWORDS = {"yesterday", "today", "tomorrow", "now", "then", "morning", "night", "evening", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
    SPATIAL_KEYWORDS = {"garden", "park", "house", "home", "office", "school", "room", "city", "town", "street", "table", "desk"}

    def parse(
        self,
        text: str,
        answer: Optional[str] = None,
        focus_constituent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse QUD structure from single utterance, dialogue pair, or context.
        """
        text = text.strip()

        # Check if text contains a question mark or dialogue turns
        if answer is not None:
            return self.parse_dialogue_pair(question=text, answer=answer, focus=focus_constituent)

        # Check if text contains both question and answer in one string
        if "?" in text:
            parts = text.split("?", 1)
            q_part = parts[0].strip() + "?"
            a_part = parts[1].strip()
            if a_part:
                return self.parse_dialogue_pair(question=q_part, answer=a_part, focus=focus_constituent)
            else:
                return self._parse_standalone_question(q_part)

        # Standalone statement -> Derive implicit QUD
        return self.derive_implicit_qud(statement=text, focus_constituent=focus_constituent)

    def parse_dialogue_pair(
        self,
        question: str,
        answer: str,
        focus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse explicit QUD from question-answer dialogue pair.
        """
        question_clean = question.strip()
        answer_clean = answer.strip()

        # Classify QUD Wh-category
        qud_type, wh_word = self._classify_wh_category(question_clean)

        # Infer focus from answer if not provided
        if not focus:
            focus = self._infer_focus_from_answer(question_clean, answer_clean, qud_type)

        # Background presupposition
        background = self._extract_background(answer_clean, focus)

        return {
            "is_explicit": True,
            "explicit_qud": question_clean,
            "implicit_qud": question_clean,
            "qud_type": qud_type,
            "wh_element": wh_word,
            "answer_statement": answer_clean,
            "focus_target": focus,
            "background_presupposition": background,
            "qud_tree": {
                "root_qud": question_clean,
                "qud_type": qud_type,
                "sub_qud": f"{wh_word} is the target of '{focus}'?",
                "assertion": answer_clean,
            },
        }

    def derive_implicit_qud(
        self,
        statement: str,
        focus_constituent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Derive implicit QUD from a standalone statement and focus constituent.
        """
        statement_clean = statement.strip()

        # Infer focus if not specified
        if not focus_constituent:
            focus_constituent = self._infer_default_focus(statement_clean)

        # Determine QUD category based on focus constituent semantics & syntax
        qud_type = self._determine_focus_qud_type(focus_constituent, statement_clean)

        # Reconstruct implicit QUD question
        implicit_qud, wh_word = self._reconstruct_implicit_question(statement_clean, focus_constituent, qud_type)

        background = self._extract_background(statement_clean, focus_constituent)

        return {
            "is_explicit": False,
            "explicit_qud": None,
            "implicit_qud": implicit_qud,
            "qud_type": qud_type,
            "wh_element": wh_word,
            "answer_statement": statement_clean,
            "focus_target": focus_constituent,
            "background_presupposition": background,
            "qud_tree": {
                "root_qud": implicit_qud,
                "qud_type": qud_type,
                "sub_qud": f"Implicit Wh-target: '{focus_constituent}'",
                "assertion": statement_clean,
            },
        }

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _classify_wh_category(self, text: str) -> tuple:
        for pattern, cat_name, wh_label in self.WH_PATTERNS:
            if pattern.search(text):
                return cat_name, wh_label

        if text.endswith("?"):
            return "POLAR", "Is/Did"
        return "UNKNOWN", "What"

    def _infer_focus_from_answer(self, question: str, answer: str, qud_type: str) -> str:
        ans_words = answer.strip().split()
        q_words = [w.lower().strip("?,.") for w in question.strip().split()]

        # Filter out given background words
        new_words = []
        for w in ans_words:
            w_clean = w.lower().strip("?,.")
            if w_clean not in q_words and w_clean not in {"a", "an", "the", "in", "at", "on"}:
                new_words.append(w)

        if new_words:
            return " ".join(new_words)

        return ans_words[-1] if ans_words else ""

    def _infer_default_focus(self, statement: str) -> str:
        words = statement.strip().split()
        clean_words = [w.strip(".,!?") for w in words]

        # Check for temporal words
        for w, cw in zip(words, clean_words):
            if cw.lower() in self.TEMPORAL_KEYWORDS:
                return w.strip(".,!?")

        # Check for spatial words
        for w, cw in zip(words, clean_words):
            if cw.lower() in self.SPATIAL_KEYWORDS:
                return w.strip(".,!?")

        # Default to first word (Subject)
        return words[0].strip(".,!?") if words else ""

    def _determine_focus_qud_type(self, focus: str, statement: str) -> str:
        f_lower = focus.lower()
        if any(kw in f_lower for kw in self.TEMPORAL_KEYWORDS):
            return "TEMPORAL"
        if any(kw in f_lower for kw in self.SPATIAL_KEYWORDS):
            return "SPATIAL"

        words = statement.strip().split()
        if words and focus.strip(".,!?") == words[0].strip(".,!?"):
            return "SUBJECT_AGENT"

        return "OBJECT_PATIENT"

    def _reconstruct_implicit_question(self, statement: str, focus: str, qud_type: str) -> tuple:
        stmt_clean = statement.strip().strip(".")

        if qud_type == "TEMPORAL":
            wh = "When"
            # Remove focus from statement to form background
            bg = stmt_clean.replace(focus, "").strip()
            bg = re.sub(r"\s+", " ", bg)
            return f"When did {bg}?", wh

        if qud_type == "SPATIAL":
            wh = "Where"
            bg = stmt_clean.replace(focus, "").strip()
            bg = re.sub(r"\s+", " ", bg)
            return f"Where did {bg}?", wh

        if qud_type == "SUBJECT_AGENT":
            wh = "Who"
            words = stmt_clean.split()
            rest = " ".join(words[1:]) if len(words) > 1 else stmt_clean
            return f"Who {rest}?", wh

        wh = "What"
        return f"What is the question concerning '{focus}' in '{stmt_clean}'?", wh

    def _extract_background(self, statement: str, focus: str) -> str:
        if not focus:
            return statement
        bg = statement.replace(focus, "...").strip()
        return re.sub(r"\s+", " ", bg)


def print_qud_report(res: Dict[str, Any]):
    print("\n" + "=" * 65)
    print(" QUESTION UNDER DISCUSSION (QUD) PARSER REPORT")
    print("=" * 65)
    print(f"  * Is Explicit QUD          : {res['is_explicit']}")
    if res["explicit_qud"]:
        print(f"  * Explicit QUD (Question)  : \"{res['explicit_qud']}\"")
    print(f"  * Implicit QUD (Underlying): \"{res['implicit_qud']}\"")
    print(f"  * QUD Category / Wh-Type   : {res['qud_type']} ({res['wh_element']})")
    print(f"  * Assertion Statement      : \"{res['answer_statement']}\"")
    print(f"  * Focus Target (Answer)    : \"{res['focus_target']}\"")
    print(f"  * Background Presupposition: \"{res['background_presupposition']}\"")

    tree = res.get("qud_tree", {})
    print("\n  --- QUD Discourse Tree Hierarchy ---")
    print(f"    [Root QUD]    : {tree.get('root_qud')}")
    print(f"    [Category]    : {tree.get('qud_type')}")
    print(f"    [Sub-QUD]     : {tree.get('sub_qud')}")
    print(f"    [Assertion]   : {tree.get('assertion')}")
    print("=" * 65 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Question Under Discussion (QUD) Parser")
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        default="When did Raman buy the book? Raman bought the book yesterday.",
        help="Dialogue pair or single statement text",
    )
    parser.add_argument(
        "--focus",
        "-f",
        type=str,
        default="",
        help="Optional target focus constituent",
    )

    args = parser.parse_args()

    qud_parser = QUDParser()

    print("\n" + "=" * 65)
    print(" QUD PARSER DEMONSTRATION")
    print("=" * 65)

    # 1. Test sample dialogue pair (Explicit QUD)
    sample_1 = args.text
    print(f"\n[Test 1: Explicit QUD Dialogue Pair]\n Input: \"{sample_1}\"")
    res1 = qud_parser.parse(sample_1, focus_constituent=args.focus)
    print_qud_report(res1)

    # 2. Test implicit QUD from standalone assertion
    sample_2 = "Raman bought the book yesterday."
    print(f"[Test 2: Implicit QUD Derivation for Statement]\n Statement: \"{sample_2}\" (Focus: 'yesterday')")
    res2 = qud_parser.parse(sample_2, focus_constituent="yesterday")
    print_qud_report(res2)

    # 3. Test implicit QUD for Subject focus
    sample_3 = "Raman bought the book yesterday."
    print(f"[Test 3: Implicit QUD Derivation for Subject Focus]\n Statement: \"{sample_3}\" (Focus: 'Raman')")
    res3 = qud_parser.parse(sample_3, focus_constituent="Raman")
    print_qud_report(res3)


if __name__ == "__main__":
    main()
