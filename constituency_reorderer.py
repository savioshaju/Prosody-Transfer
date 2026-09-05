"""
constituency_reorderer.py — Empirically-Grounded Constituent Reordering for Malayalam Focus Emphasis.

Implements the THREE attested positional reordering rules from Malayalam descriptive linguistics:

  Rule 1 — ADVERB FRONTING
      An adverb moved to pre-subject position signals emphasis on that adverb.
      Exception: sentence-connector adverbs (pinne, ennittu, appOL, …) are
      already clause-initial and do NOT imply emphasis.
      Example:
          njaan innale vannu           → ordinary (adverb is post-subject)
          innale njaan vannu           → EMPHASIS on innale (adverb is pre-subject)

  Rule 2 — DIRECT OBJECT FRONTING
      A direct object moved to sentence-initial / pre-subject position signals
      emphasis on that object.
      Example:
          paju pulla muzhuvan tinnu    → ordinary
          pulla muzhuvan paju tinnaloo → EMPHASIS on pulla muzhuvan (DO fronted)

  Rule 3 — VERB FRONTING
      The finite verb moved to sentence-initial position emphasises the
      fact/action expressed by the verb.

Crucially, NO OTHER constituent types receive a reordering rule here.
If the requested focus does not satisfy one of the three conditions above,
this module returns NOT_APPLICABLE with a reason string — it does NOT
invent a rule.

This module operates on the ORIGINAL (non-clefted) Malayalam sentence and
the mapped Malayalam focus constituent, forming the parallel REORDERING
branch alongside CLEFTING.

Architecture:

    ORIGINAL MAPPED SENTENCE + FOCUS TARGET
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
  CLEFTING             REORDERING (this module)
      │                       │
      │               check 3 empirical rules
      │                       │
      │           ┌───────────┴───────────┐
      │           │                       │
      │       Applicable           NOT APPLICABLE
      │           │                       │
      │           ▼                       ▼
      │     Reordered output      {"applicable": False, ...}
      ▼
  Cleft output
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Sentence-connector adverbs — these are already clause-initial and
# do NOT imply focus when they appear pre-subject.
# Source: descriptive list in the project specification.
# ═══════════════════════════════════════════════════════════════════════
SENTENCE_CONNECTORS: frozenset[str] = frozenset({
    # Transliteration anchors (kept for cross-reference)
    "pinne",   "ennittu",  "appOL",   "atinaal",
    "ennittum", "ennaal",  "anjine",

    # Malayalam script forms
    "പിന്നെ",         # pinne — "then / next"
    "എന്നിട്ട്",      # ennittu — "then / after that"
    "അപ്പോൾ",         # appOL — "then / at that time"
    "അതിനാൽ",         # atinaal — "so / therefore"
    "എന്നിട്ടും",     # ennittum — "still / even then"
    "അതിന്നുപുറമേ",   # atinna purame — "apart from that"
    "അതിന്നുപുറമ്പ്", # variant spelling
    "എന്നാൽ",         # ennaal — "if so / however"
    "അങ്ങനെ",         # anjine irikke — "meanwhile / thus"
    "അങ്ങനെയിരിക്കെ", # anjine irikke (full form)
    "നേരെമറിച്ച്",    # neere maricca — "on the other hand"
    "മറിച്ച്",        # short form
    "എന്നാൽ",
    "എന്നിരുന്നാലും",
    "ആദ്യം",          # aadyam — "first" (often positional, not emphatic)
})

# Malayalam morphological suffix patterns
_TRAILING_PUNCT: frozenset[str] = frozenset(".!?;:,")

# Verb-final suffixes (finite verb endings in Malayalam)
_FINITE_VERB_SUFFIXES: Tuple[str, ...] = (
    "ചു",   # past tense  -chu
    "ച്ചു",
    "ത്തു",
    "ൻ",    # future / habitual
    "ും",
    "ണ്ണു",
    "ിക്കുന്നു",
    "ുന്നു",
    "ന്നു",
    "ിക്കും",
    "ക്കും",
    "ോ",    # question particle on verb
    "ിനു",
    "ണ്ടു",
    "ററ",
    "ർത്തു",
    "ിരുന്നു",
    "ിരുന്നില്ല",
    "ിനാൽ",
)

# Adverbial morpheme endings (heuristic)
_ADVERB_SUFFIXES: Tuple[str, ...] = (
    "ലേ",    # -le  (e.g. innale — yesterday)
    "ൽ",     # locative sometimes adverbial
    "ായി",   # -āyi (manner)
    "ത്ത്",
    "ിൽ",
    "ോട്",
    "ക്ക്",
    "ൻ",
)

# Common temporal adverbs (closed-class supplement)
_TEMPORAL_ADVERBS: frozenset[str] = frozenset({
    "ഇന്നലെ",     # yesterday
    "ഇന്ന്",      # today
    "നാളെ",       # tomorrow
    "ഇപ്പോൾ",     # now
    "ഇന്ന്",
    "ഇന്നേ",
    "ഇന്നൊക്കെ",
    "ഇന്ന്",
    "ഇന്ന",
    "ഇന്ന",
    "മുൻപ്",      # before
    "ശേഷം",       # after/later
    "മുൻ",
    "ഇനി",        # henceforth
    "ഇതുവരെ",
    "ഇനിയും",
    "ഒരിക്കൽ",    # once
    "സദാ",        # always
    "എക്കാലവും",  # always / ever
    "എന്നും",     # always
    "ഒരുദിനം",
    "ഒരുനാൾ",
})

# Case suffixes that typically mark a direct object (accusative) in Malayalam
_ACCUSATIVE_SUFFIXES: Tuple[str, ...] = (
    "ിനെ",   # -ine  → animate accusative
    "യെ",    # -ye   → animate accusative
    "ിനെ",
    "നെ",    # -ne
    "ിനേ",
    "ിൻ",
    "ൾ",     # plural (can be DO)
    "ത്തെ",
    "ക്കൽ",
    "യ്ക്ക്", # dative-accusative overlap
    "്ക്ക്",
    "ാൽ",   # instrumental can be DO in some constructions
)


# ═══════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════

def _strip_punct(word: str) -> str:
    i = len(word)
    while i > 0 and word[i - 1] in _TRAILING_PUNCT:
        i -= 1
    return word[:i]


def _tokens(sentence: str) -> List[str]:
    return sentence.split()


def _clean(word: str) -> str:
    return _strip_punct(word).strip()


def _is_finite_verb(token: str) -> bool:
    """Heuristic: check whether a token looks like a Malayalam finite verb."""
    c = _clean(token)
    if not c:
        return False
    return any(c.endswith(sfx) for sfx in _FINITE_VERB_SUFFIXES)


def _is_adverb(token: str) -> bool:
    """
    Heuristic: is this token a temporal/manner adverb?
    Uses the closed-class temporal set plus morphological suffixes.
    """
    c = _clean(token)
    if not c:
        return False
    if c in _TEMPORAL_ADVERBS:
        return True
    return any(c.endswith(sfx) for sfx in _ADVERB_SUFFIXES)


def _is_sentence_connector(token: str) -> bool:
    c = _clean(token)
    return c in SENTENCE_CONNECTORS


def _is_accusative(token: str) -> bool:
    """Heuristic: does this token carry a Malayalam accusative suffix?"""
    c = _clean(token)
    return bool(c) and any(c.endswith(sfx) for sfx in _ACCUSATIVE_SUFFIXES)


def _find_subject_index(tokens: List[str]) -> Optional[int]:
    """
    Heuristic: in a standard SOV sentence the subject is at position 0
    (after optional sentence-connector). We look for the first token that
    is not a sentence-connector or adverb.
    """
    for i, tok in enumerate(tokens):
        c = _clean(tok)
        if not _is_sentence_connector(c) and not _is_adverb(c):
            return i
    return None


# ═══════════════════════════════════════════════════════════════════════
# ConstituencyReorderer
# ═══════════════════════════════════════════════════════════════════════

class ConstituencyReorderer:
    """
    Applies empirically-grounded constituent reordering rules for Malayalam.

    Only THREE rules are implemented (all others return NOT_APPLICABLE):
      1. ADVERB_FRONTING  — move adverb to pre-subject position
      2. DO_FRONTING      — move direct object to sentence-initial position
      3. VERB_FRONTING    — move finite verb to sentence-initial position

    Usage::

        reorderer = ConstituencyReorderer()
        result = reorderer.reorder(
            original_sentence="...",
            focused_constituent="...",
            focus_role="ADVERB",   # optional hint
        )
        if result["applicable"]:
            print(result["reordered_sentence"])
        else:
            print("Reordering not applicable:", result["reason"])
    """

    def reorder(
        self,
        original_sentence: str,
        focused_constituent: str,
        focus_role: str = "",
    ) -> Dict[str, Any]:
        """
        Main entry point.

        Args:
            original_sentence:   The non-clefted mapped Malayalam sentence.
            focused_constituent: The aligned Malayalam focus span (string).
            focus_role:          Optional hint from the pipeline about the
                                 grammatical role of the focus constituent
                                 (e.g. "SUBJECT", "DIRECT_OBJECT", "ADVERB",
                                 "VERB", "ADJECTIVE", …).

        Returns:
            Dict with keys:
              applicable          – bool
              rule_applied        – str | None  (e.g. "ADVERB_FRONTING")
              reordered_sentence  – str | None
              reason              – str         (human-readable explanation)
              focus_span_tokens   – List[str]   (tokens of the focus constituent)
              diagnostic          – str         (internal notes)
        """
        tokens = _tokens(original_sentence)
        fc_tokens = _tokens(focused_constituent)
        fc_clean = [_clean(t) for t in fc_tokens]

        if not tokens or not fc_clean:
            return self._not_applicable(
                "Empty sentence or focus constituent.",
                focused_constituent,
            )

        # ── Locate the focus span in the token list ──────────────────
        span_start, span_end = self._find_span(tokens, fc_clean)
        if span_start is None:
            return self._not_applicable(
                f"Focus constituent '{focused_constituent}' not found in sentence.",
                focused_constituent,
            )

        focus_tokens_in_sent = tokens[span_start: span_end + 1]

        # ── Try each rule in order ────────────────────────────────────
        result = self._try_adverb_fronting(
            tokens, focus_tokens_in_sent, span_start, span_end, focus_role,
        )
        if result is not None:
            return result

        result = self._try_do_fronting(
            tokens, focus_tokens_in_sent, span_start, span_end, focus_role,
        )
        if result is not None:
            return result

        result = self._try_verb_fronting(
            tokens, focus_tokens_in_sent, span_start, span_end, focus_role,
        )
        if result is not None:
            return result

        # ── No rule applies ───────────────────────────────────────────
        return self._not_applicable(
            f"Focus type '{focus_role or 'UNKNOWN'}' has no attested reordering rule in Malayalam. "
            "Only ADVERB_FRONTING, DO_FRONTING, and VERB_FRONTING are empirically established.",
            focused_constituent,
        )

    # ──────────────────────────────────────────────────────────────────
    # Rule 1: ADVERB FRONTING
    # ──────────────────────────────────────────────────────────────────

    def _try_adverb_fronting(
        self,
        tokens: List[str],
        focus_tokens: List[str],
        span_start: int,
        span_end: int,
        focus_role: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Move an adverb/temporal to pre-subject position.

        Applies if:
          (a) the focus span is heuristically an adverb (or focus_role == "ADVERB"),
          (b) the focus span is NOT a sentence connector, AND
          (c) the focus span is NOT already at position 0 (already pre-subject
              and a connector → ordinary, not emphatic).

        After fronting:
          [Focus] + [rest of sentence without Focus]
          (equivalent to "innale njaan vannu" from "njaan innale vannu")
        """
        is_adv = (
            focus_role.upper() in ("ADVERB", "ADVERBIAL", "TEMPORAL", "TIME", "MANNER")
            or all(_is_adverb(t) or _is_adverb(_clean(t)) for t in focus_tokens)
        )
        if not is_adv:
            return None

        # Check for sentence-connector exception
        if any(_is_sentence_connector(_clean(t)) for t in focus_tokens):
            return {
                "applicable": False,
                "rule_applied": None,
                "reordered_sentence": None,
                "reason": (
                    f"'{' '.join(focus_tokens)}' is a sentence connector. "
                    "Sentence connectors are already clause-initial and do NOT "
                    "imply emphasis when pre-subject (e.g. pinne, ennittu, appOL…)."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "ADVERB_FRONTING: blocked — sentence connector exception.",
            }

        # Check if already at position 0 with connector semantics → not emphatic
        if span_start == 0:
            # Already pre-subject; if it's a connector it's ordinary, otherwise it's
            # already in the emphatic position — report as already emphasised.
            return {
                "applicable": True,
                "rule_applied": "ADVERB_FRONTING",
                "reordered_sentence": " ".join(tokens),
                "reason": (
                    f"The adverb '{' '.join(focus_tokens)}' is already in pre-subject "
                    "position. This position is inherently emphatic in Malayalam."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "ADVERB_FRONTING: focus already at position 0 — emphatic.",
            }

        # Move focus span to front
        reordered = (
            focus_tokens
            + tokens[:span_start]
            + tokens[span_end + 1:]
        )
        return {
            "applicable": True,
            "rule_applied": "ADVERB_FRONTING",
            "reordered_sentence": " ".join(reordered),
            "reason": (
                f"Adverb '{' '.join(focus_tokens)}' moved to pre-subject position. "
                "In Malayalam, an adverb in pre-subject position implies emphasis "
                "(unless it is a sentence connector)."
            ),
            "focus_span_tokens": focus_tokens,
            "diagnostic": f"ADVERB_FRONTING: span [{span_start}:{span_end}] moved to front.",
        }

    # ──────────────────────────────────────────────────────────────────
    # Rule 2: DIRECT OBJECT FRONTING
    # ──────────────────────────────────────────────────────────────────

    def _try_do_fronting(
        self,
        tokens: List[str],
        focus_tokens: List[str],
        span_start: int,
        span_end: int,
        focus_role: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Move a direct object to sentence-initial position.

        Applies if:
          (a) focus_role is DIRECT_OBJECT / DO / OBJECT, OR
          (b) the head of the focus span carries an accusative suffix.

        After fronting:
          [FocusDO] + [rest of sentence without FocusDO]
          (equivalent to "pulla muzhuvan paju tinnaloo" from "paju pulla muzhuvan tinnu")
        """
        is_do = focus_role.upper() in (
            "DIRECT_OBJECT", "OBJECT", "DO", "ACCUSATIVE", "PATIENT",
        ) or _is_accusative(focus_tokens[-1])

        if not is_do:
            return None

        if span_start == 0:
            return {
                "applicable": True,
                "rule_applied": "DO_FRONTING",
                "reordered_sentence": " ".join(tokens),
                "reason": (
                    f"The direct object '{' '.join(focus_tokens)}' is already at "
                    "sentence-initial position, which is the emphatic position."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "DO_FRONTING: focus already at position 0.",
            }

        reordered = (
            focus_tokens
            + tokens[:span_start]
            + tokens[span_end + 1:]
        )
        return {
            "applicable": True,
            "rule_applied": "DO_FRONTING",
            "reordered_sentence": " ".join(reordered),
            "reason": (
                f"Direct object '{' '.join(focus_tokens)}' fronted to sentence-initial "
                "position. In Malayalam, a fronted direct object expresses emphasis "
                "on the object (e.g. 'all the grass' → pulla muzhuvan paju tinnaloo)."
            ),
            "focus_span_tokens": focus_tokens,
            "diagnostic": f"DO_FRONTING: span [{span_start}:{span_end}] moved to front.",
        }

    # ──────────────────────────────────────────────────────────────────
    # Rule 3: VERB FRONTING
    # ──────────────────────────────────────────────────────────────────

    def _try_verb_fronting(
        self,
        tokens: List[str],
        focus_tokens: List[str],
        span_start: int,
        span_end: int,
        focus_role: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Move the finite verb to sentence-initial position.

        Applies if:
          (a) focus_role is VERB / PREDICATE / VP, OR
          (b) the focus token is heuristically a finite verb.

        After fronting:
          [Verb] + [rest of sentence without Verb]
        """
        is_verb = focus_role.upper() in (
            "VERB", "FINITE_VERB", "PREDICATE", "VP", "ACTION",
        ) or (
            len(focus_tokens) == 1 and _is_finite_verb(focus_tokens[0])
        )

        if not is_verb:
            return None

        if span_start == 0:
            return {
                "applicable": True,
                "rule_applied": "VERB_FRONTING",
                "reordered_sentence": " ".join(tokens),
                "reason": (
                    f"The verb '{' '.join(focus_tokens)}' is already at sentence-initial "
                    "position, which is the emphatic position for verbs."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "VERB_FRONTING: focus already at position 0.",
            }

        reordered = (
            focus_tokens
            + tokens[:span_start]
            + tokens[span_end + 1:]
        )
        return {
            "applicable": True,
            "rule_applied": "VERB_FRONTING",
            "reordered_sentence": " ".join(reordered),
            "reason": (
                f"Verb '{' '.join(focus_tokens)}' moved to sentence-initial position. "
                "In Malayalam, verb fronting emphasises the fact/action expressed by the verb."
            ),
            "focus_span_tokens": focus_tokens,
            "diagnostic": f"VERB_FRONTING: span [{span_start}:{span_end}] moved to front.",
        }

    # ──────────────────────────────────────────────────────────────────
    # Span finder
    # ──────────────────────────────────────────────────────────────────

    def _find_span(
        self, tokens: List[str], fc_clean: List[str],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Locate the first occurrence of fc_clean (a list of clean token forms)
        as a contiguous subsequence in tokens (comparing clean forms).

        Returns (start_index, end_index) inclusive, or (None, None).
        """
        n = len(tokens)
        m = len(fc_clean)
        tok_clean = [_clean(t) for t in tokens]

        for i in range(n - m + 1):
            if tok_clean[i: i + m] == fc_clean:
                return i, i + m - 1

        # Partial / fuzzy fallback: find window with maximum overlap
        best_start, best_count = None, 0
        fc_set = set(fc_clean)
        for i in range(n - m + 1):
            count = sum(1 for j in range(m) if tok_clean[i + j] in fc_set)
            if count > best_count:
                best_count = count
                best_start = i

        if best_start is not None and best_count >= max(1, m // 2):
            return best_start, best_start + m - 1

        # Single-token fuzzy: find any token that matches any fc word
        if m == 1:
            for i, tc in enumerate(tok_clean):
                if tc == fc_clean[0]:
                    return i, i

        return None, None

    # ──────────────────────────────────────────────────────────────────
    # Not-applicable factory
    # ──────────────────────────────────────────────────────────────────

    def _not_applicable(
        self, reason: str, focused_constituent: str,
    ) -> Dict[str, Any]:
        return {
            "applicable": False,
            "rule_applied": None,
            "reordered_sentence": None,
            "reason": reason,
            "focus_span_tokens": _tokens(focused_constituent),
            "diagnostic": f"NOT_APPLICABLE: {reason}",
        }
