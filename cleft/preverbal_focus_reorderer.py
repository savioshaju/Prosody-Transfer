"""
preverbal_focus_reorderer.py — Preverbal Position as Focus Reordering for Malayalam (Jayaseelan 2023).

Theoretical Foundation:
  K. A. Jayaseelan (2023), "Dravidian Word Order and the Clausal Peripheries",
  The Oxford Handbook of Dravidian Languages:
  - In SOV languages like Malayalam, the structural focus position is immediately
    to the left of the finite verb (Spec, FocP immediately dominating vP).
  - Unmarked word order is SOV (canonical order: Subject > IO/Adjuncts > DO > V).
  - When a constituent (NP, PP, AdvP, VP, or word) is focused, it surfaces in the
    immediate preverbal position (Spec, FocP).
  - All non-focused constituents move past this position into topic positions (TopP*),
    preserving their relative unmarked order before the focus phrase.
  - Wh-phrases and contrastive/informational focused constituents both target Spec, FocP.
  - Wh-questions themselves are excluded from focus reordering pipelines.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from aligner.alignment_utils import strip_punctuation
from cleft.phrase_chunker import MalayalamPhraseChunker, PhraseChunk
from cleft.malayalam_pipeline import MalayalamPipeline


# ═══════════════════════════════════════════════════════════════════════
# WH-Question Words & Interrogative Markers
# ═══════════════════════════════════════════════════════════════════════

EN_WH_WORDS = frozenset({
    "who", "whom", "whose", "what", "which", "where", "when", "why", "how"
})

ML_WH_STEMS = frozenset({
    "ആര്", "ആരാണ്", "ആരെ", "ആരെയാണ്", "ആർക്ക്", "ആർക്കാണ്", "ആരുടെ", "ആരുടെയാണ്",
    "എന്ത്", "എന്താണ്", "എന്തിനാണ്", "എന്തിനാ", "എന്തുകൊണ്ട്",
    "എവിടെ", "എവിടെയാണ്", "എങ്ങോട്ട്", "എങ്ങോട്ടാണ്", "എവിടെനിന്ന്", "എവിടെനിന്നാണ്",
    "എപ്പോൾ", "എപ്പോഴാണ്", "എപ്പഴാണ്",
    "എങ്ങനെ", "എങ്ങനെയാണ്", "എങ്ങനെയുണ്ട്",
    "ഏത്", "ഏതാണ്", "ഏതിനെ", "ഏതിന്",
    "എത്ര", "എത്രയാണ്", "എത്രപേർ", "എത്രയെണ്ണം"
})


def is_wh_question(text: str, lang: str = "auto") -> Tuple[bool, str]:
    """
    Detect whether an English or Malayalam sentence is a WH-question or interrogative.
    Returns:
        (is_wh, reason_string)
    """
    if not text:
        return False, ""

    clean_text = text.strip()
    words = [strip_punctuation(w).lower() for w in clean_text.split() if strip_punctuation(w)]

    if not words:
        return False, ""

    # Check English WH-interrogatives
    has_question_mark = "?" in clean_text

    # English check
    if any(w in EN_WH_WORDS for w in words):
        first_word = words[0]
        if first_word in EN_WH_WORDS:
            return True, f"English sentence begins with WH-interrogative '{first_word}'."
        if has_question_mark and any(w in EN_WH_WORDS for w in words):
            wh_found = [w for w in words if w in EN_WH_WORDS]
            return True, f"English interrogative contains WH-word(s): {wh_found}."

    # Malayalam check
    for w in words:
        # Direct stem match
        if w in ML_WH_STEMS:
            return True, f"Malayalam sentence contains WH-interrogative '{w}'."
        # Suffix/prefix interrogative forms (e.g. എന്തിനാണു്, ആർക്കെങ്കിലും, എവിടെയെങ്കിലും)
        if any(w.startswith(stem) for stem in ("ആരാ", "എന്താ", "എവിടെയാ", "എപ്പോഴാ", "എങ്ങനെയാ", "എന്തിനാ", "എത്രയാ")):
            return True, f"Malayalam sentence contains WH-interrogative form '{w}'."

    # Verbal question particle -oo accompanied by question mark
    if has_question_mark:
        for w in words:
            if any(w.endswith(sfx) for sfx in ("വോ", "യോ", "ണോ", "ല്ലോ", "തോ")):
                return True, f"Malayalam sentence contains interrogative question particle in '{w}'."
        return True, "Sentence ends with interrogative question mark."

    return False, ""


# ═══════════════════════════════════════════════════════════════════════
# Sentence Unit Splitter & Reassembler (preserving delimiters/whitespace)
# ═══════════════════════════════════════════════════════════════════════

_SPLIT_PATTERN = re.compile(
    r"([^.;:!?]+(?:[.;:!?]+(?:\s+|$)|$))",
    re.UNICODE,
)


def split_sentences_structured(text: str) -> List[Dict[str, str]]:
    """
    Split multi-sentence text into structured segments preserving whitespace and punctuation.
    """
    raw_segments = _SPLIT_PATTERN.findall(text)
    if not raw_segments:
        stripped = text.strip()
        return [{
            "leading_ws": text[: len(text) - len(text.lstrip())],
            "body": stripped,
            "trailing_delim": text[len(text.rstrip()):],
            "original": text,
        }]

    results: List[Dict[str, str]] = []
    for seg in raw_segments:
        lead_len = len(seg) - len(seg.lstrip())
        leading_ws = seg[:lead_len]
        remainder = seg[lead_len:]

        trail_match = re.search(r"([.;:!?\s]+)$", remainder)
        if trail_match:
            body = remainder[: trail_match.start()]
            trailing_delim = trail_match.group(1)
        else:
            body = remainder
            trailing_delim = ""

        if body or trailing_delim:
            results.append({
                "leading_ws": leading_ws,
                "body": body,
                "trailing_delim": trailing_delim,
                "original": seg,
            })

    return results if results else [{
        "leading_ws": "",
        "body": text,
        "trailing_delim": "",
        "original": text,
    }]


# ═══════════════════════════════════════════════════════════════════════
# Preverbal Focus Reorderer (Jayaseelan 2023)
# ═══════════════════════════════════════════════════════════════════════

class PreverbalFocusReorderer:
    """
    Empirically grounded Dravidian Syntactic Focus Reorderer (Jayaseelan 2023).
    Places focused constituent into the structural immediate preverbal position (Spec, FocP)
    within an SOV clause structure.
    """

    def __init__(self):
        self.phrase_chunker = MalayalamPhraseChunker()
        self.ml_pipeline = MalayalamPipeline()

    def is_sentence_sov(self, tokens: List[str], main_verb: Optional[str] = None) -> Tuple[bool, int, str]:
        """
        Check whether the sentence has an SOV structure:
        Specifically, whether the finite verb is at or near the clause end (verb-final),
        allowing only quotative particles (e.g. എന്ന്) or trailing copula/auxiliaries.
        Returns:
            (is_sov, verb_index, reason)
        """
        n = len(tokens)
        if n == 0:
            return False, -1, "Empty token sequence."
        if n == 1:
            return False, 0, "Single token sentence; cannot reorder."

        clean_tokens = [strip_punctuation(t) for t in tokens]

        verb_idx = -1
        if main_verb:
            clean_mv = strip_punctuation(main_verb)
            for i in range(n - 1, -1, -1):
                if clean_tokens[i] == clean_mv or clean_tokens[i].startswith(clean_mv):
                    verb_idx = i
                    break

        if verb_idx == -1:
            # Look at the end of the sentence (last 3 tokens)
            for i in range(n - 1, max(-1, n - 4), -1):
                t = clean_tokens[i]
                if not t:
                    continue
                # Common Malayalam finite verb endings
                if (
                    any(t.endswith(sfx) for sfx in ("ുന്നു", "ിച്ചു", "തു", "ി", "ണം", "ും", "ാം", "യുന്നു", "ണ്ടു", "ന്നു", "ഞ്ഞു", "ത്തു", "പ്പെട്ടു", "പെട്ടു", "ആണ്", "ാണ്", "ഉണ്ട്", "ഇല്ല"))
                    or t in ("കണ്ടു", "വാങ്ങി", "ചെയ്തു", "പോയി", "വന്നു", "കൊടുത്തു", "പറഞ്ഞു", "എഴുതി", "തീരുമാനിച്ചു")
                ):
                    verb_idx = i
                    break

        if verb_idx == -1:
            # Fallback: assume rightmost token is the predicate head
            verb_idx = n - 1

        # Check verb finality (within last 3 tokens of the clause)
        trailing_count = (n - 1) - verb_idx
        if trailing_count > 2:
            return False, verb_idx, f"Verb at index {verb_idx} is not clause-final (followed by {trailing_count} tokens)."

        return True, verb_idx, f"SOV structure confirmed: Finite verb at index {verb_idx} ('{tokens[verb_idx]}')."

    def _chunk_token_span(self, tokens: List[str], base_offset: int = 0) -> List[Dict[str, Any]]:
        """Chunk a sub-span of tokens into syntactic constituents."""
        if not tokens:
            return []
        
        chunks = self.phrase_chunker.chunk_sentence(tokens)
        constituents = []
        covered = set()
        
        for c in chunks:
            # Check for spurious multi-argument merges (e.g. nominative agent + accusative/dative object)
            # If a chunk contains multiple case-marked or independent nominal heads, split them
            c_toks = tokens[c.start_idx : c.end_idx + 1]
            if len(c_toks) > 1:
                # If any internal token (not the last) carries an accusative/dative/locative case, split
                split_points = []
                for idx_within, t in enumerate(c_toks[:-1]):
                    c_clean = strip_punctuation(t)
                    if any(c_clean.endswith(sfx) for sfx in ("നെ", "യെ", "ിനെ", "ക്ക്", "യ്ക്ക്", "്ക്ക്", "ന്", "ിന്", "ിൽ", "ത്തിൽ", "ൽ")):
                        split_points.append(idx_within + 1)
                
                if split_points:
                    prev_sp = 0
                    for sp in split_points + [len(c_toks)]:
                        sub_toks = c_toks[prev_sp:sp]
                        abs_indices = list(range(base_offset + c.start_idx + prev_sp, base_offset + c.start_idx + sp))
                        constituents.append({
                            "type": c.chunk_type,
                            "text": " ".join(sub_toks),
                            "token_indices": abs_indices,
                            "tokens": sub_toks,
                            "is_verb": False,
                        })
                        covered.update(range(c.start_idx + prev_sp, c.start_idx + sp))
                        prev_sp = sp
                    continue

            abs_indices = list(range(base_offset + c.start_idx, base_offset + c.end_idx + 1))
            constituents.append({
                "type": c.chunk_type,
                "text": " ".join(c_toks),
                "token_indices": abs_indices,
                "tokens": c_toks,
                "is_verb": False,
            })
            covered.update(range(c.start_idx, c.end_idx + 1))

        for i, t in enumerate(tokens):
            if i not in covered and strip_punctuation(t):
                constituents.append({
                    "type": "XP",
                    "text": t,
                    "token_indices": [base_offset + i],
                    "tokens": [t],
                    "is_verb": False,
                })

        constituents.sort(key=lambda x: x["token_indices"][0])
        return constituents

    def partition_into_constituents(
        self,
        tokens: List[str],
        verb_idx: int,
        focused_constituent: str,
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Partition tokens into syntactic constituents respecting the focus constituent boundaries:
          - Identifies the exact token span of the focus constituent.
          - Chunks the pre-focus span, the focus constituent itself, and the post-focus pre-verbal span.
          - Appends the finite verb constituent.
        Returns:
            (constituents, focus_constituent_idx)
        """
        n = len(tokens)
        clean_tokens = [strip_punctuation(t) for t in tokens]
        clean_focus_words = [strip_punctuation(w) for w in focused_constituent.split() if strip_punctuation(w)]

        pre_verb_tokens = clean_tokens[:verb_idx]
        pre_verb_raw = tokens[:verb_idx]

        # 1. Locate focus span in pre_verb_tokens
        f_start, f_end = -1, -1
        flen = len(clean_focus_words)

        if flen > 0:
            # Exact slice match
            for i in range(len(pre_verb_tokens) - flen + 1):
                if pre_verb_tokens[i : i + flen] == clean_focus_words:
                    f_start = i
                    f_end = i + flen - 1
                    break

            # Fallback substring match
            if f_start == -1:
                clean_fc_str = " ".join(clean_focus_words)
                for i, t in enumerate(pre_verb_tokens):
                    if clean_fc_str == t or clean_fc_str in t or t in clean_fc_str:
                        f_start = i
                        f_end = i
                        break

            # Token overlap fallback
            if f_start == -1:
                for i, t in enumerate(pre_verb_tokens):
                    if any(fw == t or fw in t for fw in clean_focus_words):
                        f_start = i
                        f_end = i
                        break

        # If focus not found in pre-verb tokens, check if it was the verb itself
        if f_start == -1:
            return [], None

        # Expand focus span to include adjacent postposition (e.g. തോട്ടത്തിൽ + വെച്ച്)
        if f_end + 1 < len(pre_verb_tokens):
            from cleft.cleft_pipeline import POSTPOSITIONS
            next_w = pre_verb_tokens[f_end + 1]
            if next_w in POSTPOSITIONS:
                f_end += 1

        # 2. Build constituents: pre-focus, focus, post-focus
        constituents: List[Dict[str, Any]] = []

        # Pre-focus constituents
        if f_start > 0:
            pre_consts = self._chunk_token_span(pre_verb_raw[:f_start], base_offset=0)
            constituents.extend(pre_consts)

        # Focus constituent
        focus_c_idx = len(constituents)
        focus_indices = list(range(f_start, f_end + 1))
        focus_raw_toks = pre_verb_raw[f_start : f_end + 1]
        
        # Determine constituent type
        f_type = "NP"
        f_text_clean = " ".join(clean_tokens[f_start : f_end + 1])
        from cleft.phrase_chunker import TIME_WORDS, POSTPOSITIONS
        if any(w in TIME_WORDS for w in clean_tokens[f_start : f_end + 1]) or any(f_text_clean.endswith(sfx) for sfx in ("മായി", "ആയി", "ഓടെ")):
            f_type = "AdvP"
        elif any(w in POSTPOSITIONS for w in clean_tokens[f_start : f_end + 1]) or any(f_text_clean.endswith(sfx) for sfx in ("നിന്ന്", "ൽനിന്ന്", "ലേക്ക്", "ിലേക്ക്")):
            f_type = "PP"

        constituents.append({
            "type": f_type,
            "text": " ".join(focus_raw_toks),
            "token_indices": focus_indices,
            "tokens": focus_raw_toks,
            "is_verb": False,
        })

        # Post-focus pre-verb constituents
        if f_end + 1 < verb_idx:
            post_consts = self._chunk_token_span(pre_verb_raw[f_end + 1 : verb_idx], base_offset=f_end + 1)
            constituents.extend(post_consts)

        # Verb constituent
        verb_indices = list(range(verb_idx, n))
        verb_tokens = tokens[verb_idx:n]
        constituents.append({
            "type": "V",
            "text": " ".join(verb_tokens),
            "token_indices": verb_indices,
            "tokens": verb_tokens,
            "is_verb": True,
        })

        return constituents, focus_c_idx

    def reorder_clause(
        self,
        clause_text: str,
        focused_constituent: str,
        main_verb: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reorder a single clause placing the focused constituent into the immediate preverbal position.
        """
        raw_tokens = clause_text.strip().split()
        if not raw_tokens:
            return {
                "applicable": False,
                "reordered_sentence": clause_text,
                "reason": "Empty clause.",
            }

        # 1. SOV Structure Check
        is_sov, verb_idx, sov_reason = self.is_sentence_sov(raw_tokens, main_verb)
        if not is_sov:
            return {
                "applicable": False,
                "reordered_sentence": clause_text,
                "reason": f"Clause does not satisfy SOV condition: {sov_reason}",
            }

        # 2. Partition into constituents
        constituents, focus_c_idx = self.partition_into_constituents(raw_tokens, verb_idx, focused_constituent)

        if focus_c_idx is None:
            return {
                "applicable": False,
                "reordered_sentence": clause_text,
                "reason": f"Focused constituent '{focused_constituent}' could not be resolved into clause constituents.",
            }

        num_constituents = len(constituents)
        verb_c_idx = num_constituents - 1  # Last constituent is V

        pos_before = focus_c_idx + 1  # 1-indexed
        c_focus = constituents[focus_c_idx]

        # 3. Check if already preverbal (immediately preceding V)
        if focus_c_idx == verb_c_idx - 1:
            # Already immediately preverbal
            return {
                "applicable": True,
                "reordered_sentence": clause_text,
                "already_preverbal": True,
                "position_before": pos_before,
                "position_after": pos_before,
                "movement_vector": "in-situ (preverbal)",
                "focused_constituent": c_focus["text"],
                "constituent_type": c_focus["type"],
                "constituent_sequence_before": [f"{c['type']}({c['text']})" for c in constituents],
                "constituent_sequence_after": [f"{c['type']}({c['text']})" for c in constituents],
                "reason": (
                    f"Constituent '{c_focus['text']}' [{c_focus['type']}] already occupies "
                    "the canonical immediate preverbal focus position (Spec-FocP)."
                ),
            }

        # 4. Perform Syntactic Preverbal Movement (Jayaseelan 2023 Spec-FocP)
        # Separate remaining non-focused constituents (topics) in their original relative order
        topics = [c for i, c in enumerate(constituents) if i != focus_c_idx and i != verb_c_idx]
        v_const = constituents[verb_c_idx]

        # Preverbal order: [Topics] + [Focused Constituent] + [Verb]
        reordered_constituents = topics + [c_focus, v_const]
        pos_after = len(topics) + 1  # 1-indexed position right before V

        reordered_clause = " ".join(c["text"] for c in reordered_constituents)

        return {
            "applicable": True,
            "reordered_sentence": reordered_clause,
            "already_preverbal": False,
            "position_before": pos_before,
            "position_after": pos_after,
            "movement_vector": f"{pos_before} -> {pos_after}",
            "focused_constituent": c_focus["text"],
            "constituent_type": c_focus["type"],
            "constituent_sequence_before": [f"{c['type']}({c['text']})" for c in constituents],
            "constituent_sequence_after": [
                (f"FOC:{c['type']}({c['text']})" if c is c_focus else f"{c['type']}({c['text']})")
                for c in reordered_constituents
            ],
            "reason": (
                f"Moved focused constituent '{c_focus['text']}' [{c_focus['type']}] "
                f"from position {pos_before} to immediate preverbal position {pos_after} (Spec-FocP), "
                "with non-focused constituents raised to topic positions (TopP*) preserving relative order (Jayaseelan 2023)."
            ),
        }

    def reorder(
        self,
        original_sentence: str,
        focused_constituent: str,
        main_verb: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for multi-sentence / multi-clause inputs.
        Isolates the target clause containing the focus constituent,
        applies preverbal focus reordering, and reassembles full text.
        """
        clean_full = original_sentence.strip()
        if not clean_full:
            return {
                "applicable": False,
                "reordered_sentence": original_sentence,
                "reason": "Empty input string.",
            }

        # Check WH-question blocking upfront
        is_wh, wh_reason = is_wh_question(clean_full)
        if is_wh:
            return {
                "applicable": False,
                "reordered_sentence": original_sentence,
                "status": "BLOCKED_WH_QUESTION",
                "reason": f"WH_QUESTION_BLOCKED: {wh_reason}",
            }

        # Multi-sentence / clause isolation
        segments = split_sentences_structured(clean_full)
        clean_focus = strip_punctuation(focused_constituent).strip()

        # Find target segment containing focus
        target_idx = None
        for i, seg in enumerate(segments):
            seg_body = seg["body"]
            if clean_focus in seg_body or any(w in seg_body.split() for w in clean_focus.split() if w):
                target_idx = i
                break

        if target_idx is None:
            # Fallback to first segment if not found
            target_idx = 0

        target_seg = segments[target_idx]
        clause_res = self.reorder_clause(
            clause_text=target_seg["body"],
            focused_constituent=focused_constituent,
            main_verb=main_verb,
        )

        if not clause_res.get("applicable"):
            return {
                "applicable": False,
                "reordered_sentence": original_sentence,
                "isolated_sentence_before": target_seg["body"],
                "reason": clause_res.get("reason", "Not applicable."),
            }

        # Reassemble full text
        reordered_body = clause_res["reordered_sentence"]
        assembled_parts = []
        for i, seg in enumerate(segments):
            if i == target_idx:
                assembled_parts.append(f"{seg['leading_ws']}{reordered_body}{seg['trailing_delim']}")
            else:
                assembled_parts.append(seg["original"])

        final_reordered = "".join(assembled_parts).strip()

        clause_res["reordered_sentence"] = final_reordered
        clause_res["isolated_sentence_before"] = target_seg["body"]
        clause_res["isolated_sentence_after"] = reordered_body
        return clause_res
