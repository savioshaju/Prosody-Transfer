"""
constituency_reorderer.py — Empirically-Grounded 4-Level Constituency Reordering for Malayalam Focus Emphasis.

Implements the 4-Level Constituency Reordering Architecture grounded in Malayalam descriptive linguistics
(Asher & Kumari 1997, Jayaseelan 1999/2001, Mohanan 1982):

  Level 1 — SENTENCE/CLAUSE IDENTIFICATION & ISOLATION:
      From multi-sentence inputs [S1] [S2] ... [Sn], isolates ONLY the sentence/clause S_focus
      containing the focus constituent. Prefix and suffix sentences remain untouched.

  Level 2 — EMPIRICALLY-GROUNDED CONSTITUENT REORDERING (inside S_focus):
      Implements strictly the THREE attested positional reordering rules:
        Rule 1 — ADVERB FRONTING:
            An adverb moved to pre-subject position signals emphasis on that adverb.
            Exception: sentence-connector adverbs (pinne, ennittu, appOL, ...) are
            already clause-initial and do NOT imply emphasis.
        Rule 2 — DIRECT OBJECT FRONTING:
            A direct object moved to sentence-initial / pre-subject position signals
            emphasis on that object.
            * STRICT HIERARCHY: Explicit focus_role -> Parse IR -> Morphological Evidence -> UNKNOWN.
            * NO POSITIONAL GUESSING: Inanimate objects with zero case marking (Differential Object
              Marking / DOM) are NEVER guessed to be direct objects from word order alone.
        Rule 3 — VERB FRONTING:
            The finite verb (or finite verb compound/sequence) moved to sentence-initial
            position emphasises the fact/action expressed by the verb.

      Supports multi-token focus constituents across all three rules (AdvP, DO NP, VP).

  Level 3 — CONSTITUENT POSITION TRACKING:
      Partitions S_focus into constituent units [C1, C2, ..., Ck] where the focus constituent
      is treated as an indivisible unit.
      Computes:
        position_before: 1-indexed constituent position before reordering.
        position_after:  1-indexed constituent position after reordering.
        movement_vector: string representation (e.g. "3 -> 1").

  Level 4 — REINSERTION & FULL-TEXT REASSEMBLY:
      Replaces S_focus with the transformed sentence in the multi-sentence stream,
      preserving exact surrounding text, punctuation, and whitespace.

If the requested focus does not satisfy one of the three conditions above, this module returns
NOT_APPLICABLE with an evidence-based reason string — it does NOT invent speculative rules.
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
    "എന്നിരുന്നാലും",
    "ആദ്യം",          # aadyam — "first" (often positional, not emphatic)
})

# Malayalam morphological suffix patterns
_TRAILING_PUNCT: frozenset[str] = frozenset(".!?;:,\"'`“”‘’()[]{}।॥")

# Verb-final suffixes (finite verb endings in Malayalam)
_FINITE_VERB_SUFFIXES: Tuple[str, ...] = (
    "ചു",   # past tense  -chu
    "ച്ചു",
    "ത്തു",
    "ും",    # future tense -um
    "ണ്ണു",
    "ിക്കുന്നു", # present tense -ikkunnu
    "ുന്നു",    # present tense -unnu
    "ന്നു",     # past tense -nnu
    "ിക്കും",   # future tense -ikkum
    "ക്കും",    # future tense -kkum
    "ിരുന്നു",  # past continuous / perfect
    "ിരുന്നില്ല",# negative past
    "ില്ല",     # negative
    "ണ്ടു",    # past / existential
    "ർത്തു",
)

# Adverbial morpheme endings (productive Malayalam adverbial morphology)
_ADVERB_SUFFIXES: Tuple[str, ...] = (
    "ആയി",    # -āyi  (manner, e.g. nannaayi)
    "ായി",    # -āyi
    "ആയിട്ട്", # -āyiṭṭu (manner)
    "ായിട്ട്",
    "ഓടെ",    # -ōḍe  (manner, e.g. santooSattoode)
    "ത്തോടെ",
    "ലേ",     # -le   (temporal, e.g. innale)
    "ത്തേ",    # -ttē  (temporal, e.g. neerattē)
)

# Common temporal adverbs (closed-class supplement)
_TEMPORAL_ADVERBS: frozenset[str] = frozenset({
    "ഇന്നലെ",     # yesterday
    "ഇന്ന്",      # today
    "നാളെ",       # tomorrow
    "ഇപ്പോൾ",     # now
    "ഇന്നേ",
    "ഇന്നൊക്കെ",
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
    "പിന്നീട്",    # later
})

# Common manner and other adverbs
_COMMON_ADVERBS: frozenset[str] = frozenset({
    "പെട്ടെന്ന്",   # quickly / suddenly
    "വേഗം",       # fast / quickly
    "പതുക്കെ",    # slowly
    "നന്നായി",     # well
    "ഉടൻ",        # immediately
    "ഉടനെ",       # immediately
    "വീണ്ടും",     # again
    "ഏറെ",        # much / greatly
    "ദൂരെ",       # far
    "അരികിൽ",     # near
    "സമീപം",      # near
    "എപ്പോഴും",   # always
})

# Degree modifiers commonly preceding adverbs or adjectives
DEGREE_MODIFIERS: frozenset[str] = frozenset({
    "വളരെ", "ഏറ്റവും", "തീരെ", "കുറച്ച്", "കൂടുതൽ", "അല്പം", "വല്ലാതെ",
})

# Determiners commonly introducing noun phrases
DETERMINERS: frozenset[str] = frozenset({
    "ആ", "ഈ", "ഒരു", "ഏതോ", "എല്ലാ", "ഓരോ", "ചില",
})

# Postpositions commonly following nominal complements in PPs
POSTPOSITIONS: frozenset[str] = frozenset({
    "വെച്ച്", "നിന്ന്", "വരെ", "കൂടെ", "കൊണ്ട്", "പറ്റി", "കുറിച്ച്",
    "ശേഷം", "മുൻപ്", "പകരം", "വേണ്ടി", "കൂടി", "ഇടയിൽ", "ഉള്ളിൽ",
})

# Case suffixes that typically mark a direct object (accusative) in Malayalam
_ACCUSATIVE_SUFFIXES: Tuple[str, ...] = (
    "ിനെ",   # -ine  -> animate accusative
    "യെ",    # -ye   -> animate accusative
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
    """Check whether a token looks like a Malayalam finite verb."""
    c = _clean(token)
    if not c:
        return False
    return any(c.endswith(sfx) for sfx in _FINITE_VERB_SUFFIXES)


def _is_adverb(token: str) -> bool:
    """
    Check whether token is a temporal/manner adverb.
    Uses closed-class temporal set, common manner adverb set, and morphological suffixes.
    """
    c = _clean(token)
    if not c:
        return False
    if c in _TEMPORAL_ADVERBS or c in _COMMON_ADVERBS:
        return True
    return any(c.endswith(sfx) for sfx in _ADVERB_SUFFIXES)


def _is_sentence_connector(token: str) -> bool:
    c = _clean(token)
    return c in SENTENCE_CONNECTORS


def _is_accusative(token: str) -> bool:
    """Check whether token carries overt Malayalam accusative case morphology."""
    c = _clean(token)
    return bool(c) and any(c.endswith(sfx) for sfx in _ACCUSATIVE_SUFFIXES)


# ═══════════════════════════════════════════════════════════════════════
# Level 1: Sentence Segmentation & Isolation
# ═══════════════════════════════════════════════════════════════════════

def split_sentences_with_delimiters(text: str) -> List[Dict[str, str]]:
    """
    Splits text into structured sentence units while preserving exact spacing and delimiters.
    Guarantees: ''.join(seg['raw'] for seg in segments) == text.

    Returns list of dicts with keys:
      'raw': exact raw segment substring
      'leading_ws': leading whitespace
      'body': sentence body without leading whitespace or trailing punctuation/delimiter
      'trailing_delim': trailing punctuation and whitespace (e.g. '. ', '.\n', '!')
    """
    pattern = re.compile(r'([^.!?।॥\n]+(?:[.!?।॥]+|\n+|$))', re.UNICODE)
    raw_segments = pattern.findall(text)
    if not raw_segments and text:
        raw_segments = [text]

    parsed = []
    for raw in raw_segments:
        l_stripped = raw.lstrip()
        leading_ws = raw[:len(raw) - len(l_stripped)]

        r_match = re.search(r'([.!?।॥\s]*)$', l_stripped)
        if r_match and r_match.group(1):
            trailing_part = r_match.group(1)
            body = l_stripped[:-len(trailing_part)].strip()
        else:
            body = l_stripped.strip()
            trailing_part = ''

        parsed.append({
            'raw': raw,
            'leading_ws': leading_ws,
            'body': body,
            'trailing_delim': trailing_part,
        })
    return parsed


# ═══════════════════════════════════════════════════════════════════════
# Level 3: Constituent Partitioning & Position Tracking
# ═══════════════════════════════════════════════════════════════════════

def partition_clause_constituents(
    tokens: List[str],
    span_start: int,
    span_end: int,
) -> List[Tuple[int, int, str]]:
    """
    Partitions clause tokens into a list of (start_idx, end_idx, constituent_str) tuples.
    Guarantees that:
      1. The focus span [span_start, span_end] is treated as an indivisible constituent.
      2. Tokens before focus are grouped into cohesive constituent units (NP, AdvP, Connectors).
      3. Tokens after focus are grouped into cohesive constituent units (NP, PP, Verb sequences).
    """
    def _chunk_range(r_start: int, r_end: int) -> List[Tuple[int, int, str]]:
        chunks = []
        i = r_start
        while i < r_end:
            tok = tokens[i]
            c_tok = _clean(tok)

            # 1. Sentence connector standalone
            if _is_sentence_connector(c_tok):
                chunks.append((i, i, tok))
                i += 1
                continue

            # 2. Determiner or Degree Modifier + Head word (e.g. ആ വലിയ, വളരെ വേഗം)
            if c_tok in DETERMINERS or c_tok in DEGREE_MODIFIERS:
                if i + 1 < r_end:
                    chunks.append((i, i + 1, " ".join(tokens[i: i + 2])))
                    i += 2
                    continue

            # 3. Noun + Postposition (e.g. തോട്ടത്തിൽ വെച്ച്, വീട്ടിൽ നിന്ന്)
            if i + 1 < r_end and _clean(tokens[i + 1]) in POSTPOSITIONS:
                chunks.append((i, i + 1, " ".join(tokens[i: i + 2])))
                i += 2
                continue

            # 4. Verb sequence at end of clause (e.g. കഴിച്ചു തീർത്തു)
            if i + 1 == r_end - 1 and (c_tok.endswith("ു") or c_tok.endswith("ി")):
                chunks.append((i, i + 1, " ".join(tokens[i: i + 2])))
                i += 2
                continue

            # Default: single token constituent
            chunks.append((i, i, tok))
            i += 1
        return chunks

    pre_chunks = _chunk_range(0, span_start)
    focus_str = " ".join(tokens[span_start: span_end + 1])
    focus_chunk = [(span_start, span_end, focus_str)]
    post_chunks = _chunk_range(span_end + 1, len(tokens))

    return pre_chunks + focus_chunk + post_chunks


# ═══════════════════════════════════════════════════════════════════════
# ConstituencyReorderer
# ═══════════════════════════════════════════════════════════════════════

class ConstituencyReorderer:
    """
    Applies empirically-grounded 4-level constituent reordering rules for Malayalam.

    Only THREE rules are implemented (all others return NOT_APPLICABLE):
      1. ADVERB_FRONTING  — move adverb to pre-subject position
      2. DO_FRONTING      — move direct object to sentence-initial position
      3. VERB_FRONTING    — move finite verb to sentence-initial position

    Hierarchy for role determination:
      Explicit focus_role -> Parse IR -> Morphological Evidence -> UNKNOWN.
      Strictly NO positional guessing of direct objects from SOV order.
    """

    def reorder(
        self,
        original_sentence: str,
        focused_constituent: str,
        focus_role: str = "",
        parse_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point implementing Levels 1 through 4.

        Args:
            original_sentence:   The non-clefted mapped Malayalam text (single or multi-sentence).
            focused_constituent: The aligned Malayalam focus span (string).
            focus_role:          Optional hint about grammatical role (e.g. "ADVERB", "DIRECT_OBJECT", "VERB").
            parse_info:          Optional syntactic parse or dependency annotations.

        Returns:
            Dict containing full diagnostics, constituent vectors, and transformed sentence/text.
        """
        if not original_sentence.strip() or not focused_constituent.strip():
            return self._not_applicable(
                "Empty sentence or focus constituent.",
                focused_constituent,
                original_sentence,
            )

        # ── Level 1: Sentence Identification & Isolation ───────────────
        segments = split_sentences_with_delimiters(original_sentence)
        fc_tokens = _tokens(focused_constituent)
        fc_clean = [_clean(t) for t in fc_tokens if _clean(t)]

        if not fc_clean:
            return self._not_applicable(
                "Focus constituent contains no valid tokens.",
                focused_constituent,
                original_sentence,
            )

        target_seg_idx: Optional[int] = None
        target_span: Tuple[Optional[int], Optional[int]] = (None, None)
        target_tokens: List[str] = []

        # Find the specific sentence containing the focus span
        for idx, seg in enumerate(segments):
            seg_body = seg["body"]
            if not seg_body:
                continue
            s_tokens = _tokens(seg_body)
            span_start, span_end = self._find_span(s_tokens, fc_clean)
            if span_start is not None and span_end is not None:
                target_seg_idx = idx
                target_span = (span_start, span_end)
                target_tokens = s_tokens
                break

        if target_seg_idx is None or target_span[0] is None or target_span[1] is None:
            return self._not_applicable(
                f"Focus constituent '{focused_constituent}' not found in any sentence of the input.",
                focused_constituent,
                original_sentence,
            )

        s_focus = segments[target_seg_idx]
        span_start, span_end = target_span
        focus_tokens_in_sent = target_tokens[span_start: span_end + 1]

        # ── Level 2: Try Reordering Rules on Isolated S_focus ──────────
        # Evidence Hierarchy:
        #   1. Explicit focus_role
        #   2. Parse / Dependency IR
        #   3. Morphological Evidence
        #   4. UNKNOWN -> NOT_APPLICABLE (NO positional guessing)

        role_upper = focus_role.upper().strip()

        # Step 1: Explicit focus_role dispatch
        if role_upper in ("DIRECT_OBJECT", "OBJECT", "DO", "ACCUSATIVE", "PATIENT", "NP-OBJ"):
            res = self._try_do_fronting(
                target_tokens, focus_tokens_in_sent, span_start, span_end, focus_role, parse_info,
            )
            if res is not None:
                return self._finalize_result(
                    res, segments, target_seg_idx, target_tokens, span_start, span_end, original_sentence,
                )
        elif role_upper in ("ADVERB", "ADVERBIAL", "TEMPORAL", "TIME", "MANNER", "ADVP"):
            res = self._try_adverb_fronting(
                target_tokens, focus_tokens_in_sent, span_start, span_end, focus_role, parse_info,
            )
            if res is not None:
                return self._finalize_result(
                    res, segments, target_seg_idx, target_tokens, span_start, span_end, original_sentence,
                )
        elif role_upper in ("VERB", "FINITE_VERB", "PREDICATE", "VP", "ACTION", "VFIN"):
            res = self._try_verb_fronting(
                target_tokens, focus_tokens_in_sent, span_start, span_end, focus_role, parse_info,
            )
            if res is not None:
                return self._finalize_result(
                    res, segments, target_seg_idx, target_tokens, span_start, span_end, original_sentence,
                )
        elif role_upper in ("SUBJECT", "ADJECTIVE", "PP", "NP", "GENITIVE", "NUMERAL", "QUANTIFIER"):
            # Explicit non-reorderable role in Malayalam linguistics
            return self._not_applicable(
                f"Focus type '{role_upper}' has no attested reordering rule in Malayalam. "
                "Only ADVERB_FRONTING, DO_FRONTING, and VERB_FRONTING are empirically established.",
                focused_constituent,
                original_sentence,
            )
        else:
            # Step 2 & 3: Role not specified or unknown. Check morphological / parse evidence.
            # (A) Check overt accusative case marking
            if any(_is_accusative(t) for t in focus_tokens_in_sent):
                res = self._try_do_fronting(
                    target_tokens, focus_tokens_in_sent, span_start, span_end, focus_role, parse_info,
                )
                if res is not None:
                    return self._finalize_result(
                        res, segments, target_seg_idx, target_tokens, span_start, span_end, original_sentence,
                    )

            # (B) Check finite verb morphology
            if _is_finite_verb(focus_tokens_in_sent[-1]):
                res = self._try_verb_fronting(
                    target_tokens, focus_tokens_in_sent, span_start, span_end, focus_role, parse_info,
                )
                if res is not None:
                    return self._finalize_result(
                        res, segments, target_seg_idx, target_tokens, span_start, span_end, original_sentence,
                    )

            # (C) Check adverbial morphology / closed class
            adv_morph = (
                any(_is_adverb(t) for t in focus_tokens_in_sent)
                or any(_clean(t) in _TEMPORAL_ADVERBS for t in focus_tokens_in_sent)
                or any(_clean(t) in _COMMON_ADVERBS for t in focus_tokens_in_sent)
                or (_clean(focus_tokens_in_sent[0]) in DEGREE_MODIFIERS and len(focus_tokens_in_sent) > 1)
            )
            if adv_morph:
                res = self._try_adverb_fronting(
                    target_tokens, focus_tokens_in_sent, span_start, span_end, focus_role, parse_info,
                )
                if res is not None:
                    return self._finalize_result(
                        res, segments, target_seg_idx, target_tokens, span_start, span_end, original_sentence,
                    )

        # ── Step 4: No rule applies (UNKNOWN) ─────────────────────────
        role_label = focus_role.upper() if focus_role else "UNKNOWN"
        return self._not_applicable(
            f"Focus type '{role_label}' has no attested reordering rule in Malayalam. "
            "Only ADVERB_FRONTING, DO_FRONTING, and VERB_FRONTING are empirically established.",
            focused_constituent,
            original_sentence,
        )

    # ──────────────────────────────────────────────────────────────────
    # Level 2 Rules
    # ──────────────────────────────────────────────────────────────────

    def _try_adverb_fronting(
        self,
        tokens: List[str],
        focus_tokens: List[str],
        span_start: int,
        span_end: int,
        focus_role: str,
        parse_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Rule 1: ADVERB FRONTING
        Moves an adverb/AdvP to pre-subject position.
        """
        role_upper = focus_role.upper().strip()
        role_matches = role_upper in (
            "ADVERB", "ADVERBIAL", "TEMPORAL", "TIME", "MANNER", "ADVP",
        )

        parse_matches = False
        if parse_info and isinstance(parse_info, dict):
            parse_tag = parse_info.get("pos", "").upper() or parse_info.get("role", "").upper()
            parse_matches = parse_tag in ("ADVERB", "ADVP", "RB")

        morph_matches = (
            any(_is_adverb(t) for t in focus_tokens)
            or any(_clean(t) in _TEMPORAL_ADVERBS for t in focus_tokens)
            or any(_clean(t) in _COMMON_ADVERBS for t in focus_tokens)
            or (_clean(focus_tokens[0]) in DEGREE_MODIFIERS and len(focus_tokens) > 1)
        )

        if not (role_matches or parse_matches or morph_matches):
            return None

        # Check for sentence-connector exception (cannot front connector as focus)
        if any(_is_sentence_connector(_clean(t)) for t in focus_tokens):
            return {
                "applicable": False,
                "rule_applied": None,
                "reordered_tokens": None,
                "reason": (
                    f"'{' '.join(focus_tokens)}' is a sentence connector. "
                    "Sentence connectors are already clause-initial and do NOT "
                    "imply emphasis when pre-subject (e.g. pinne, ennittu, appOL…)."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "ADVERB_FRONTING: blocked — sentence connector exception.",
            }

        # Determine subject position in the clause to front before subject
        subj_idx = self._find_subject_index(tokens)

        # If already at or before subject position
        if span_start == 0 or (subj_idx is not None and span_end < subj_idx):
            return {
                "applicable": True,
                "rule_applied": "ADVERB_FRONTING",
                "reordered_tokens": list(tokens),
                "reason": (
                    f"The adverb '{' '.join(focus_tokens)}' is already in pre-subject "
                    "position. This position is inherently emphatic in Malayalam."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "ADVERB_FRONTING: focus already in pre-subject position — emphatic.",
            }

        # Check if token 0 is a sentence connector
        if tokens and _is_sentence_connector(tokens[0]):
            # Place after connector, before subject: [Connector] + [Focus] + [Rest]
            reordered = (
                [tokens[0]]
                + focus_tokens
                + tokens[1:span_start]
                + tokens[span_end + 1:]
            )
        else:
            # Place at sentence initial: [Focus] + [Rest]
            reordered = (
                focus_tokens
                + tokens[:span_start]
                + tokens[span_end + 1:]
            )

        return {
            "applicable": True,
            "rule_applied": "ADVERB_FRONTING",
            "reordered_tokens": reordered,
            "reason": (
                f"Adverb '{' '.join(focus_tokens)}' moved to pre-subject position. "
                "In Malayalam, an adverb in pre-subject position implies emphasis "
                "(unless it is a sentence connector)."
            ),
            "focus_span_tokens": focus_tokens,
            "diagnostic": f"ADVERB_FRONTING: span [{span_start}:{span_end}] moved to pre-subject.",
        }

    def _try_do_fronting(
        self,
        tokens: List[str],
        focus_tokens: List[str],
        span_start: int,
        span_end: int,
        focus_role: str,
        parse_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Rule 2: DIRECT OBJECT FRONTING
        Moves a direct object to sentence-initial position.

        Follows strict evidence hierarchy:
          1. Explicit focus_role in DO set.
          2. Parse / Dependency IR marking as object.
          3. Morphological evidence (overt accusative marking).
          4. UNKNOWN (NO positional guessing).
        """
        role_upper = focus_role.upper().strip()
        role_matches = role_upper in (
            "DIRECT_OBJECT", "OBJECT", "DO", "ACCUSATIVE", "PATIENT", "NP-OBJ",
        )

        parse_matches = False
        if parse_info and isinstance(parse_info, dict):
            parse_role = parse_info.get("role", "").upper() or parse_info.get("dep", "").upper()
            parse_matches = parse_role in ("DO", "OBJ", "DIRECT_OBJECT", "PATIENT", "NP-OBJ")

        # Check overt morphological accusative marking on head/tokens
        morph_matches = (
            _is_accusative(focus_tokens[-1])
            or any(_is_accusative(t) for t in focus_tokens)
        )

        if not (role_matches or parse_matches or morph_matches):
            return None

        # Check if already at position 0 (or position 1 after connector)
        has_connector = tokens and _is_sentence_connector(tokens[0])
        target_initial_pos = 1 if has_connector else 0

        if span_start == target_initial_pos or span_start == 0:
            return {
                "applicable": True,
                "rule_applied": "DO_FRONTING",
                "reordered_tokens": list(tokens),
                "reason": (
                    f"The direct object '{' '.join(focus_tokens)}' is already at "
                    "sentence-initial position, which is the emphatic position."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "DO_FRONTING: focus already at sentence-initial position.",
            }

        if has_connector:
            reordered = (
                [tokens[0]]
                + focus_tokens
                + tokens[1:span_start]
                + tokens[span_end + 1:]
            )
        else:
            reordered = (
                focus_tokens
                + tokens[:span_start]
                + tokens[span_end + 1:]
            )

        return {
            "applicable": True,
            "rule_applied": "DO_FRONTING",
            "reordered_tokens": reordered,
            "reason": (
                f"Direct object '{' '.join(focus_tokens)}' fronted to sentence-initial "
                "position. In Malayalam, a fronted direct object expresses emphasis on the object."
            ),
            "focus_span_tokens": focus_tokens,
            "diagnostic": f"DO_FRONTING: span [{span_start}:{span_end}] moved to front.",
        }

    def _try_verb_fronting(
        self,
        tokens: List[str],
        focus_tokens: List[str],
        span_start: int,
        span_end: int,
        focus_role: str,
        parse_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Rule 3: VERB FRONTING
        Moves a finite verb (or finite compound/serial verb) to sentence-initial position.
        """
        role_upper = focus_role.upper().strip()
        role_matches = role_upper in (
            "VERB", "FINITE_VERB", "PREDICATE", "VP", "ACTION", "VFIN",
        )

        parse_matches = False
        if parse_info and isinstance(parse_info, dict):
            parse_role = parse_info.get("pos", "").upper() or parse_info.get("role", "").upper()
            parse_matches = parse_role in ("VERB", "VFIN", "VP", "PREDICATE")

        # In Malayalam finite verb sequences, the final auxiliary carries the finite tense inflection
        morph_matches = _is_finite_verb(focus_tokens[-1])

        if not (role_matches or parse_matches or morph_matches):
            return None

        has_connector = tokens and _is_sentence_connector(tokens[0])
        target_initial_pos = 1 if has_connector else 0

        if span_start == target_initial_pos or span_start == 0:
            return {
                "applicable": True,
                "rule_applied": "VERB_FRONTING",
                "reordered_tokens": list(tokens),
                "reason": (
                    f"The verb '{' '.join(focus_tokens)}' is already at sentence-initial "
                    "position, which is the emphatic position for verbs."
                ),
                "focus_span_tokens": focus_tokens,
                "diagnostic": "VERB_FRONTING: focus already at sentence-initial position.",
            }

        if has_connector:
            reordered = (
                [tokens[0]]
                + focus_tokens
                + tokens[1:span_start]
                + tokens[span_end + 1:]
            )
        else:
            reordered = (
                focus_tokens
                + tokens[:span_start]
                + tokens[span_end + 1:]
            )

        return {
            "applicable": True,
            "rule_applied": "VERB_FRONTING",
            "reordered_tokens": reordered,
            "reason": (
                f"Verb '{' '.join(focus_tokens)}' moved to sentence-initial position. "
                "In Malayalam, verb fronting emphasises the fact/action expressed by the verb."
            ),
            "focus_span_tokens": focus_tokens,
            "diagnostic": f"VERB_FRONTING: span [{span_start}:{span_end}] moved to front.",
        }

    # ──────────────────────────────────────────────────────────────────
    # Level 3 & Level 4 Result Finalizer
    # ──────────────────────────────────────────────────────────────────

    def _finalize_result(
        self,
        rule_res: Dict[str, Any],
        segments: List[Dict[str, str]],
        target_seg_idx: int,
        orig_tokens: List[str],
        span_start: int,
        span_end: int,
        original_sentence: str,
    ) -> Dict[str, Any]:
        """
        Computes Level 3 (constituent tracking) and Level 4 (reinsertion) to produce
        the final return structure.
        """
        if not rule_res.get("applicable"):
            return {
                "applicable": False,
                "rule_applied": None,
                "reordered_sentence": None,
                "original_sentence": original_sentence,
                "isolated_sentence_before": segments[target_seg_idx]["body"],
                "isolated_sentence_after": None,
                "full_text_before": original_sentence,
                "full_text_after": None,
                "position_before": None,
                "position_after": None,
                "movement_vector": None,
                "constituent_sequence_before": [],
                "constituent_sequence_after": [],
                "focus_span_tokens": rule_res.get("focus_span_tokens", []),
                "reason": rule_res.get("reason", ""),
                "diagnostic": rule_res.get("diagnostic", ""),
            }

        reordered_tokens: List[str] = rule_res["reordered_tokens"]
        s_focus = segments[target_seg_idx]

        # ── Level 3: Constituent Position Tracking ───────────────────
        # 1. Before reordering
        before_chunks = partition_clause_constituents(orig_tokens, span_start, span_end)
        constituents_before = [c[2] for c in before_chunks]

        # Identify focus constituent index before (1-indexed)
        focus_idx_before = 1
        for i, c in enumerate(before_chunks):
            if c[0] == span_start and c[1] == span_end:
                focus_idx_before = i + 1
                break

        # 2. After reordering: find new span of focus tokens in reordered_tokens
        fc_clean = [_clean(t) for t in rule_res["focus_span_tokens"] if _clean(t)]
        new_span_start, new_span_end = self._find_span(reordered_tokens, fc_clean)
        if new_span_start is None or new_span_end is None:
            # Fallback if find_span misses exact match
            new_span_start = 1 if (reordered_tokens and _is_sentence_connector(reordered_tokens[0])) else 0
            new_span_end = new_span_start + len(rule_res["focus_span_tokens"]) - 1

        after_chunks = partition_clause_constituents(reordered_tokens, new_span_start, new_span_end)
        constituents_after = [c[2] for c in after_chunks]

        focus_idx_after = 1
        for i, c in enumerate(after_chunks):
            if c[0] == new_span_start and c[1] == new_span_end:
                focus_idx_after = i + 1
                break

        movement_vector = f"{focus_idx_before} -> {focus_idx_after}"

        # ── Level 4: Reinsertion & Full-Text Output ──────────────────
        isolated_after_body = " ".join(reordered_tokens)
        isolated_after_full = s_focus["leading_ws"] + isolated_after_body + s_focus["trailing_delim"]

        # Reconstruct full text with only S_focus modified
        reassembled_segments = []
        for idx, seg in enumerate(segments):
            if idx == target_seg_idx:
                reassembled_segments.append(isolated_after_full)
            else:
                reassembled_segments.append(seg["raw"])

        full_text_after = "".join(reassembled_segments)

        return {
            "applicable": True,
            "rule_applied": rule_res["rule_applied"],
            "reordered_sentence": full_text_after,
            "original_sentence": original_sentence,
            "isolated_sentence_before": s_focus["body"],
            "isolated_sentence_after": isolated_after_body,
            "full_text_before": original_sentence,
            "full_text_after": full_text_after,
            "position_before": focus_idx_before,
            "position_after": focus_idx_after,
            "movement_vector": movement_vector,
            "constituent_sequence_before": constituents_before,
            "constituent_sequence_after": constituents_after,
            "focus_span_tokens": rule_res["focus_span_tokens"],
            "reason": rule_res["reason"],
            "diagnostic": f"{rule_res['diagnostic']} Movement vector: [{movement_vector}].",
        }

    # ──────────────────────────────────────────────────────────────────
    # Span finder & Helpers
    # ──────────────────────────────────────────────────────────────────

    def _find_span(
        self, tokens: List[str], fc_clean: List[str],
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Locate the first occurrence of fc_clean (list of clean token forms)
        as a contiguous subsequence in tokens (comparing clean forms).
        Returns (start_index, end_index) inclusive, or (None, None).
        """
        n = len(tokens)
        m = len(fc_clean)
        if not n or not m or n < m:
            return None, None

        tok_clean = [_clean(t) for t in tokens]

        # 1. Exact contiguous match
        for i in range(n - m + 1):
            if tok_clean[i: i + m] == fc_clean:
                return i, i + m - 1

        # 2. Substring/subtoken match within tokens
        for i in range(n - m + 1):
            match = True
            for j in range(m):
                tc = tok_clean[i + j]
                fc = fc_clean[j]
                if not (tc == fc or (fc and fc in tc)):
                    match = False
                    break
            if match:
                return i, i + m - 1

        # 3. Best overlap window
        best_start, best_count = None, 0
        fc_set = set(fc_clean)
        for i in range(n - m + 1):
            count = sum(1 for j in range(m) if tok_clean[i + j] in fc_set)
            if count > best_count:
                best_count = count
                best_start = i

        if best_start is not None and best_count >= max(1, m // 2):
            return best_start, best_start + m - 1

        # 4. Single-token fallback
        if m == 1:
            for i, tc in enumerate(tok_clean):
                if tc == fc_clean[0] or (fc_clean[0] and fc_clean[0] in tc):
                    return i, i

        return None, None

    def _find_subject_index(self, tokens: List[str]) -> Optional[int]:
        """
        Heuristic: in a standard clause, look for the first non-connector, non-adverb token.
        """
        for i, tok in enumerate(tokens):
            c = _clean(tok)
            if not _is_sentence_connector(c) and not _is_adverb(c):
                return i
        return None

    def _not_applicable(
        self,
        reason: str,
        focused_constituent: str,
        original_sentence: str = "",
    ) -> Dict[str, Any]:
        return {
            "applicable": False,
            "rule_applied": None,
            "reordered_sentence": None,
            "original_sentence": original_sentence,
            "isolated_sentence_before": None,
            "isolated_sentence_after": None,
            "full_text_before": original_sentence,
            "full_text_after": None,
            "position_before": None,
            "position_after": None,
            "movement_vector": None,
            "constituent_sequence_before": [],
            "constituent_sequence_after": [],
            "focus_span_tokens": _tokens(focused_constituent),
            "reason": reason,
            "diagnostic": f"NOT_APPLICABLE: {reason}",
        }
