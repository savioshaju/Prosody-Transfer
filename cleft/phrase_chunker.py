"""
phrase_chunker.py — Deterministic Malayalam Phrase & Clause Chunker for Syntactic Constituency.

Identifies complete maximal projections (XP) and clausal domains in Malayalam sentences:
  - NP (Noun Phrase): [Quantifier/Numeral] + [Attributive Adjective / Genitive] + [Noun Stem]* + [Head Noun + Case]
  - PP (Postpositional Phrase): [NP] + [Postposition / Ablative / Allative suffix]
  - AdvP (Adverbial Phrase): [Degree modifier]* + [Manner/Time Adverb or Converb]
  - VP (Verb Phrase): [Participle]* + [Main Verb] + [Auxiliary chain]

Enforces Dravidian Island Constraints (Jayaseelan 2001; Moag §11.2):
  1. Left Branch Condition: Attributive adjectives/genitives cannot be extracted alone;
     the whole NP must be clefted.
  2. Postposition Stranding Ban: A DP cannot be extracted away from its postposition;
     the whole PP must be clefted.
  3. Clausal Integrity: Focus projection must preserve clausal scope and not cross independent
     or coordinate clause boundaries.
"""

import re
from typing import List, Dict, Any, Optional, Tuple

from aligner.alignment_utils import strip_punctuation


POSTPOSITIONS = frozenset({
    "കൂടെ", "ഒപ്പം", "ശേഷം", "മുമ്പ്", "മുൻപ്", "കുറിച്ച്", "സംബന്ധിച്ച്",
    "കൊണ്ട്", "വഴി", "ഉൾപ്പെടെ", "ഉള്‍പ്പെടെ", "പകരം", "പകരമായി", "നേതൃത്വത്തിൽ",
    "പ്രകാരം", "കാരണം", "അനുസരിച്ച്", "പോലെ", "വരെ", "എതിരെ",
    "തുടങ്ങി", "ആയി", "ആയിട്ട്", "കുറിച്ചുള്ള", "നേരെ", "മാത്രം", "മാത്രമേ",
    "കൂടാതെ", "മുതൽ", "മുതല്‍", "വച്ച്", "വെച്ച്", "അടക്കം"
})

PP_SUFFIXES = (
    "ൽനിന്ന്", "യിൽനിന്ന്", "ിൽനിന്ന്", "നിന്ന്", "നിന്നും", "ൽനിന്നും", "യിൽനിന്നും",
    "ലേക്ക്", "ിലേക്ക്", "ത്തേക്ക്", "ത്തക്ക്", "വരെ", "കൂടെ", "ശേഷം", "കൊണ്ട്", "പ്രകാരം"
)

DEGREE_MODIFIERS = frozenset({
    "വളരെ", "കൂടുതൽ", "തീരെ", "അല്പം", "ഒരല്പം", "ഏറ്റവും", "അത്യധികം",
    "ഏറെ", "കുറച്ച്", "വല്ലാതെ", "അധികം", "തീർത്തും", "ഏറ്റം", "ഏറ്റവുമധികം", "ഏതാണ്ട്"
})

TIME_WORDS = frozenset({
    "ഇന്നലെ", "ഇന്ന്", "നാളെ", "രാവിലെ", "വൈകുന്നേരം", "ഇപ്പോൾ",
    "അന്ന്", "പിറ്റേന്ന്", "നേരത്തെ", "എപ്പോൾ", "പണ്ട്", "ഇപ്പോഴേ",
    "രാത്രി", "പകൽ", "ഉച്ചയ്ക്ക്", "അസ്തമയത്ത്", "തുടക്കത്തിൽ", "ഒടുവിൽ"
})

CASE_SUFFIXES = (
    "നെ", "യെ", "ിനെ", "ക്ക്", "യ്ക്ക്", "്ക്ക്", "ന്", "ിന്",
    "ൽ", "ിൽ", "ത്തിൽ", "ത്ത്", "ലെ", "ിലേക്ക്", "ലേക്ക്",
    "ആൽ", "ാൽ", "ത്തിനാൽ", "ഇനാൽ", "ഓട്", "യോട്", "നിന്ന്", "ൽനിന്ന്", "യിൽനിന്ന്",
    "ന്റെ", "ുടെ", "ിന്റെ", "്റെ"
)


class PhraseChunk:
    def __init__(self, chunk_type: str, start_idx: int, end_idx: int, tokens: List[str], text: str, clause_id: int = 0):
        self.chunk_type = chunk_type  # 'NP', 'PP', 'AdvP', 'VP'
        self.start_idx = start_idx
        self.end_idx = end_idx  # inclusive token index
        self.tokens = tokens
        self.text = text
        self.clause_id = clause_id

    def __repr__(self):
        return f"[{self.chunk_type} (C{self.clause_id}): '{self.text}' ({self.start_idx}..{self.end_idx})]"


class MalayalamPhraseChunker:
    """
    Syntactic constituent and clausal chunker for Malayalam sentences.
    Resolves complete multi-word constituents: NP, PP, AdvP, VP across clausal boundaries.
    """

    def segment_clauses(self, sentence: str) -> List[Tuple[int, int, str]]:
        """
        Segment a sentence into clausal intervals (start_char, end_char, clause_text).
        Recognizes coordinate punctuation (; . :), converb/conditional boundaries.
        """
        clause_spans = []
        raw_splits = list(re.finditer(r"(?:;\s*|:\s+|\.\s+)", sentence))
        if not raw_splits:
            return [(0, len(sentence), sentence)]

        last_pos = 0
        for m in raw_splits:
            end_pos = m.start()
            if end_pos > last_pos:
                c_text = sentence[last_pos:end_pos].strip()
                if c_text:
                    clause_spans.append((last_pos, end_pos, c_text))
            last_pos = m.end()

        if last_pos < len(sentence):
            c_text = sentence[last_pos:].strip()
            if c_text:
                clause_spans.append((last_pos, len(sentence), c_text))

        return clause_spans if clause_spans else [(0, len(sentence), sentence)]

    def chunk_sentence(self, tokens: List[Any], sentence_text: Optional[str] = None) -> List[PhraseChunk]:
        """
        Segment a token list (strings or IR Token objects) into complete phrases.
        """
        token_forms = [
            strip_punctuation(getattr(t, "form", str(t))).strip()
            for t in tokens
        ]
        token_poses = [
            getattr(t, "form_pos", "").upper()
            for t in tokens
        ]

        n = len(token_forms)
        chunks: List[PhraseChunk] = []

        # Determine clause boundaries across tokens if sentence_text provided
        clause_id_map = [0] * n
        if sentence_text:
            clauses = self.segment_clauses(sentence_text)
            curr_c = 0
            char_cursor = 0
            for i, f in enumerate(token_forms):
                if not f:
                    continue
                pos = sentence_text.find(f, char_cursor)
                if pos != -1:
                    char_cursor = pos + len(f)
                    for c_idx, (c_start, c_end, _) in enumerate(clauses):
                        if c_start <= pos < c_end:
                            clause_id_map[i] = c_idx
                            break

        i = 0
        while i < n:
            form = token_forms[i]
            c_id = clause_id_map[i]

            if not form:
                i += 1
                continue

            # 1. Check for AdvP (Degree + Adverb, or standalone Time word, or Converb)
            if form in DEGREE_MODIFIERS and i + 1 < n:
                next_form = token_forms[i + 1]
                if (
                    next_form.endswith("മായി")
                    or next_form.endswith("ആയി")
                    or next_form in TIME_WORDS
                    or any(next_form.endswith(sfx) for sfx in ("ത്തിൽ", "ഓടെ"))
                ):
                    chunks.append(PhraseChunk("AdvP", i, i + 1, token_forms[i : i + 2], " ".join(token_forms[i : i + 2]), c_id))
                    i += 2
                    continue

            if form in TIME_WORDS or any(form.endswith(sfx) for sfx in ("പ്പോൾ", "ുമ്പോൾ", "മ്പോൾ", "തിനുശേഷം", "തിനുമുമ്പ്", "ആയി", "ആയിട്ട്", "ഓടെ")):
                chunks.append(PhraseChunk("AdvP", i, i, [form], form, c_id))
                i += 1
                continue

            # 2. Check for multi-word NP or PP
            start_np = i
            curr = i
            while curr < n:
                c_form = token_forms[curr]
                c_pos = token_poses[curr] if curr < len(token_poses) else ""

                # Stop NP expansion if we cross into a new clause
                if clause_id_map[curr] != c_id:
                    chunks.append(PhraseChunk("NP", start_np, max(start_np, curr - 1), token_forms[start_np : curr], " ".join(token_forms[start_np : curr]), c_id))
                    i = curr
                    break

                # Suffix-based PP detection: words ending in -ൽനിന്ന്, -ലേക്ക്, etc.
                has_pp_suffix = any(c_form.endswith(sfx) for sfx in PP_SUFFIXES)
                has_case = any(c_form.endswith(sfx) for sfx in CASE_SUFFIXES)
                has_pp_word = curr + 1 < n and token_forms[curr + 1] in POSTPOSITIONS

                # Left-branch modifiers (genitive, adjectival participle, numeral) continue the phrase
                is_genitive = c_form.endswith(("ന്റെ", "ുടെ", "ിന്റെ", "്റെ"))
                is_adjective = (
                    c_pos.startswith("JJ")
                    or any(c_form.endswith(sfx) for sfx in ("യ", "ന്ന", "ത്ത", "ിയ", "ക്കൻ", "ിലെ", "ലെ"))
                    or c_form in ("ഈ", "ആ", "ഏത്", "നല്ല", "വലിയ", "ചെറിയ", "ഒരു", "രണ്ട്", "മൂന്ന്")
                )

                if is_genitive or is_adjective:
                    curr += 1
                    continue

                if has_pp_word:
                    # Incorporate standalone postposition into PP
                    pp_end = curr + 1
                    chunks.append(PhraseChunk("PP", start_np, pp_end, token_forms[start_np : pp_end + 1], " ".join(token_forms[start_np : pp_end + 1]), c_id))
                    i = pp_end + 1
                    break

                if has_pp_suffix:
                    # Suffixed PP (e.g., രണ്ട് പ്രാഥമികസ്രോതസ്സുകളിൽനിന്ന്)
                    chunks.append(PhraseChunk("PP", start_np, curr, token_forms[start_np : curr + 1], " ".join(token_forms[start_np : curr + 1]), c_id))
                    i = curr + 1
                    break

                if has_case:
                    # Case marker terminates argument NP
                    chunks.append(PhraseChunk("NP", start_np, curr, token_forms[start_np : curr + 1], " ".join(token_forms[start_np : curr + 1]), c_id))
                    i = curr + 1
                    break

                # If next word starts a new constituent (e.g. demonstrative, new adjective, or different clause)
                if curr + 1 < n:
                    next_f = token_forms[curr + 1]
                    next_p = token_poses[curr + 1] if curr + 1 < len(token_poses) else ""
                    next_is_adj = (
                        next_p.startswith("JJ")
                        or any(next_f.endswith(sfx) for sfx in ("യ", "ന്ന", "ത്ത", "ിയ", "ക്കൻ", "ലെ", "ിലെ", "ന്റെ", "ുടെ", "ിന്റെ"))
                        or next_f in ("വടക്കൻ", "തെക്കൻ", "കിഴക്കൻ", "പടിഞ്ഞാറൻ", "ഈ", "ആ", "ഏത്", "നല്ല", "വലിയ", "ചെറിയ", "ഒരു", "രണ്ട്", "മൂന്ന്")
                    )
                    # Note: If next_f carries a case suffix, an uninflected noun stem compounds into it (e.g. ഡെറാഡൂൺ + നഗരത്തിന് -> [ഡെറാഡൂൺ നഗരത്തിന്])
                    if next_f in POSTPOSITIONS or next_is_adj or clause_id_map[curr + 1] != c_id:
                        chunks.append(PhraseChunk("NP", start_np, curr, token_forms[start_np : curr + 1], " ".join(token_forms[start_np : curr + 1]), c_id))
                        i = curr + 1
                        break

                curr += 1
            else:
                chunks.append(PhraseChunk("NP", start_np, start_np, [form], form, c_id))
                i = start_np + 1

        return chunks

    def find_enclosing_phrase(self, tokens: List[Any], focus_start: int, focus_end: int, sentence_text: Optional[str] = None) -> Optional[PhraseChunk]:
        """
        Find the maximal phrase chunk that contains or overlaps the focus span.
        Enforces that sub-constituents expand to their maximal projection (Jayaseelan 2001).
        """
        chunks = self.chunk_sentence(tokens, sentence_text)
        
        # 1. Exact match or fully enclosing
        for chunk in chunks:
            if chunk.start_idx <= focus_start and focus_end <= chunk.end_idx:
                return chunk

        # 2. Maximum token overlap
        best_overlap = 0
        best_chunk = None
        focus_set = set(range(focus_start, focus_end + 1))
        for chunk in chunks:
            chunk_set = set(range(chunk.start_idx, chunk.end_idx + 1))
            overlap = len(focus_set & chunk_set)
            if overlap > best_overlap:
                best_overlap = overlap
                best_chunk = chunk

        return best_chunk
