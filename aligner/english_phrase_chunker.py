"""
english_phrase_chunker.py — English Phrase & Clause Chunker for Cross-Lingual Alignment.

Uses spaCy (en_core_web_trf) to identify:
  - Clauses: Main/Matrix clause, Relative clause (relcl), Adverbial clause (advcl), Complement clause (ccomp)
  - Phrases:
      * NP (Noun Phrase): [Determiner/Modifier]* + [Noun/Pronoun/ProperNoun]+
      * PP (Prepositional Phrase): [Preposition] + [Complement NP]
      * AdvP (Adverbial Phrase): [Adverb / Adverbial modifier chain]
      * VP (Verb Phrase): [Auxiliaries]* + [Main Verb] + [Particles]

Given an English focus word or span, finds:
  1. The enclosing clause
  2. The maximal syntactic phrase (XP) corresponding to the focus
"""

import re
from typing import List, Dict, Any, Optional, Tuple

_NLP = None

def get_spacy_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except Exception:
                try:
                    _NLP = spacy.load("en_core_web_trf")
                except Exception:
                    _NLP = None
        except Exception:
            _NLP = None
    return _NLP


class EnglishChunk:
    def __init__(self, chunk_type: str, start_char: int, end_char: int, start_token: int, end_token: int, text: str, clause_id: int = 0):
        self.chunk_type = chunk_type  # 'NP', 'PP', 'AdvP', 'VP', 'CLAUSE'
        self.start_char = start_char
        self.end_char = end_char
        self.start_token = start_token
        self.end_token = end_token  # inclusive
        self.text = text
        self.clause_id = clause_id

    def __repr__(self):
        return f"[{self.chunk_type} (C{self.clause_id}): '{self.text}' ({self.start_token}..{self.end_token})]"


class EnglishPhraseChunker:
    """
    Syntactic constituent and clause parser for English sentences.
    """

    def __init__(self):
        self.nlp = get_spacy_nlp()

    def parse(self, sentence: str) -> Dict[str, Any]:
        """
        Parse sentence into token list, clauses, and typed phrase chunks.
        """
        if not self.nlp:
            return self._fallback_parse(sentence)

        doc = self.nlp(sentence)
        tokens = [t.text for t in doc]

        # 1. Identify clauses based on verb heads and dependency edges
        clause_heads = []
        for token in doc:
            if token.pos_ in ("VERB", "AUX") and token.dep_ in ("ROOT", "advcl", "relcl", "ccomp", "xcomp", "conj"):
                clause_heads.append(token)

        # Map each token to a clause id
        clause_map = {}
        for i, token in enumerate(doc):
            curr = token
            # Climb up to nearest clause head
            while curr.head != curr and curr not in clause_heads:
                curr = curr.head
            clause_id = clause_heads.index(curr) if curr in clause_heads else 0
            clause_map[i] = clause_id

        chunks: List[EnglishChunk] = []

        # 2. Extract Prepositional Phrases (PP): Prep + POBJ NP
        pp_token_spans = set()
        for token in doc:
            if token.pos_ == "ADP" and token.dep_ in ("prep", "agent"):
                pobj_children = [c for c in token.children if c.dep_ in ("pobj", "pcomp")]
                if pobj_children:
                    pobj = pobj_children[0]
                    # Subtree spanning from token to end of pobj's subtree
                    start_t = token.i
                    # Find maximal end of pobj subtree
                    end_t = max(t.i for t in pobj.subtree)
                    pp_text = doc[start_t : end_t + 1].text
                    c_id = clause_map.get(start_t, 0)
                    chunks.append(EnglishChunk("PP", doc[start_t].idx, doc[end_t].idx + len(doc[end_t].text), start_t, end_t, pp_text, c_id))
                    for idx in range(start_t, end_t + 1):
                        pp_token_spans.add(idx)

        # 3. Extract Noun Chunks (NP)
        for nc in doc.noun_chunks:
            # If the noun chunk is strictly inside a PP, we still keep it or let PP take precedence for prepositional focus
            c_id = clause_map.get(nc.start, 0)
            chunks.append(EnglishChunk("NP", nc.start_char, nc.end_char, nc.start, nc.end - 1, nc.text, c_id))

        # 4. Extract Adverbial Phrases (AdvP)
        for token in doc:
            if token.pos_ == "ADV" and token.dep_ in ("advmod", "npadvmod"):
                adv_tokens = sorted([t.i for t in token.subtree if t.pos_ in ("ADV", "PART") or t.dep_ in ("advmod", "neg")])
                if adv_tokens:
                    s_t, e_t = min(adv_tokens), max(adv_tokens)
                    adv_text = doc[s_t : e_t + 1].text
                    c_id = clause_map.get(s_t, 0)
                    chunks.append(EnglishChunk("AdvP", doc[s_t].idx, doc[e_t].idx + len(doc[e_t].text), s_t, e_t, adv_text, c_id))

        # Sort chunks by start token
        chunks.sort(key=lambda c: (c.start_token, -(c.end_token - c.start_token)))

        return {
            "tokens": tokens,
            "chunks": chunks,
            "num_clauses": len(clause_heads) if clause_heads else 1,
            "doc": doc
        }

    def _fallback_parse(self, sentence: str) -> Dict[str, Any]:
        """Simple regex fallback when spaCy is unavailable."""
        tokens = sentence.split()
        chunks = []
        for i, t in enumerate(tokens):
            chunks.append(EnglishChunk("NP", 0, 0, i, i, t, 0))
        return {
            "tokens": tokens,
            "chunks": chunks,
            "num_clauses": 1,
            "doc": None
        }

    def find_focus_phrase(self, sentence: str, focus_text: str) -> Optional[EnglishChunk]:
        """
        Given the English focus expression, find the best-matching maximal constituent
        (PP if preposition is included in focus, otherwise NP, AdvP, etc.).
        """
        parsed = self.parse(sentence)
        chunks = parsed["chunks"]
        f_clean = re.sub(r"[^\w\s]", "", focus_text).strip().lower()
        if not f_clean:
            return None

        # 1. Look for exact string match among chunks (preferring PP > NP > AdvP if focus contains prep)
        for c in chunks:
            c_clean = re.sub(r"[^\w\s]", "", c.text).strip().lower()
            if c_clean == f_clean:
                return c

        # 2. Look for enclosing chunk where focus is a sub-phrase
        # e.g., focus "two primary sources" inside PP "from two primary sources" or NP "two primary sources"
        candidates = []
        for c in chunks:
            c_clean = re.sub(r"[^\w\s]", "", c.text).strip().lower()
            if f_clean in c_clean or c_clean in f_clean:
                overlap = len(set(f_clean.split()) & set(c_clean.split()))
                candidates.append((overlap, c))

        if candidates:
            # Sort by overlap descending, then by length closest to focus
            candidates.sort(key=lambda x: (x[0], -abs(len(x[1].text) - len(focus_text))), reverse=True)
            return candidates[0][1]

        # 3. Fallback: create an ad-hoc chunk from focus_text
        doc = parsed.get("doc")
        if doc and focus_text in sentence:
            start_char = sentence.find(focus_text)
            end_char = start_char + len(focus_text)
            span = doc.char_span(start_char, end_char)
            if span:
                c_type = "PP" if span[0].pos_ == "ADP" else ("AdvP" if span[0].pos_ == "ADV" else "NP")
                return EnglishChunk(c_type, start_char, end_char, span.start, span.end - 1, span.text, 0)

        return EnglishChunk("NP", 0, len(focus_text), 0, len(focus_text.split()) - 1, focus_text, 0)
