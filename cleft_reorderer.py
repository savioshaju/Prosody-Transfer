"""
Clause-Aware Cleft Positional Reorderer for Malayalam.

Takes the output of AwesomeCleftPipeline.process() and produces three positional
variants of the Malayalam cleft sentence:

    1. Preverbal / Contrastive:    [BG] + [Focus+ആണ്] + [NomVerb]
    2. Postverbal / Information:   [BG] + [NomVerb]   + [Focus+ആണ്]
    3. Clause-Initial / Strong:    [Focus+ആണ്] + [BG] + [NomVerb]

Architecture:
    FULL EMPHASIZED SENTENCE
    │
    ├── Pre-Cleft Clauses   → preserved in position
    ├── Cleft Clause        → reordered (only this clause is touched)
    │    ├── Background tokens
    │    ├── Focus + ആണ്
    │    └── Nominalized Verb
    └── Post-Cleft Clauses  → preserved in position

The module never modifies existing pipeline code — it is a pure downstream
consumer of the dictionary returned by AwesomeCleftPipeline.process().
"""

from typing import List, Dict, Any, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# Nominalization suffix patterns (used to detect nominalized verbs)
# ═══════════════════════════════════════════════════════════════════════
NOMINALIZATION_SUFFIXES = (
    "ുന്നത്",   # present nominalization  (e.g. കുടിക്കുന്നത്)
    "ന്നത്",    # present nominalization variant
    "ിയത്",    # past nominalization     (e.g. വാങ്ങിയത്)
    "ത്തത്",    # past nominalization variant
    "ച്ചത്",    # past nominalization     (e.g. കുടിച്ചത്)
    "യത്",     # past nominalization variant
    "ണ്ടത്",    # past nominalization     (e.g. കൊണ്ടത്)
    "ടത്",     # past nominalization variant
    "ന്ത്",    # past nominalization variant
)


# ═══════════════════════════════════════════════════════════════════════
# NP-modifier surface heuristics
# (lightweight replica of AwesomeCleftPipeline._is_np_modifier)
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# NP-modifier surface heuristics & Subordinate Clause Markers
# ═══════════════════════════════════════════════════════════════════════
_RP_SUFFIXES = (
    "ിട്ടുള്ള", "ഉള്ള", "ുന്ന", "ാത്ത", "പ്പെട്ട",
    "ായ", "ിയ", "ച്ച", "ത്ത", "ന്ന", "മായ", "മായുള്ള",
    "പരമായ", "പരമായി", "സംബന്ധമായ", "സംബന്ധിച്ച്",
)
_GENITIVE_SUFFIXES = ("യുടെ", "ുടെ", "ന്റെ", "ിന്റെ", "്റെ")
_ADJ_LOCATIVE_SUFFIXES = ("ിലെ", "ലെ", "ത്തെ")
_ADVERBIAL_SUFFIXES = ("മായി", "മായും", "ായി", "ായും", "പൂർവ്വം", "വണ്ണം")
_DEMONYM_ADJECTIVES = frozenset({
    "ഇന്ത്യൻ", "അമേരിക്കൻ", "ഉത്തരേന്ത്യൻ", "ദക്ഷിണേന്ത്യൻ", "പാശ്ചാത്യൻ",
    "ജർമ്മൻ", "റഷ്യൻ", "ഏഷ്യൻ", "കേരളീയൻ", "യൂറോപ്യൻ", "ബ്രിട്ടീഷ്",
    "ചൈനീസ്", "ജാപ്പനീസ്",
})
_ADJ_SUFFIXES = ("ീയ", "ിക", "ികമായ", "ത്ത")
_DETERMINERS_NUMERALS = frozenset({
    "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്",
    "ഏഴ്", "എട്ട്", "ഒൻപത്", "പത്ത്", "ഇരുപത്", "മുപ്പത്",
    "നൂറ്", "ആയിരം", "ദി", "ദ", "ദ്", "ഒരു", "എ",
    "മറ്റ്", "മറ്റു", "ചില", "പല", "എല്ലാ", "ഓരോ",
})
_DEGREE_ADVERBS = frozenset({
    "ഏറ്റവും", "ഏറെ", "അത്യന്തം", "വളരെ", "കൂടുതൽ", "ഏറ്റം",
    "പ്രധാനമായും", "പ്രത്യേകിച്ചും", "സാധാരണയായി", "പൊതുവെ",
})

_SUBORDINATE_MARKERS = (
    "എങ്കിലും", "ആണെങ്കിലും", "കിലും", "ഉണ്ടെങ്കിലും",
    "പ്പോൾ", "ുമ്പോൾ", "ിക്കുമ്പോൾ", "ന്നപ്പോൾ", "യപ്പോൾ", "ച്ചപ്പോൾ",
    "തിനുശേഷം", "തിന് ശേഷം", "ശേഷം", "മുമ്പ്", "തിനുമുമ്പ്",
    "തുകൊണ്ട്", "തിനാൽ", "കാരണം",
    "യാണെങ്കിൽ", "എങ്കിൽ", "ച്ചാൽ", "ന്നാൽ",
)

_TRAILING_PUNCT = frozenset(".!?;:,")


# ═══════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════

def _strip_punct(word: str) -> str:
    """Strip trailing sentence/clause punctuation from a token."""
    i = len(word)
    while i > 0 and word[i - 1] in _TRAILING_PUNCT:
        i -= 1
    return word[:i]


def _is_np_modifier(word: str) -> bool:
    """
    Surface-level check if a Malayalam token is an NP-internal modifier
    (determiner, adjective, relative participle, genitive, degree adverb, numeral).
    """
    clean = _strip_punct(word).strip()
    if not clean:
        return False
    if clean in _DETERMINERS_NUMERALS or clean in _DEGREE_ADVERBS or clean in _DEMONYM_ADJECTIVES or clean.isdigit():
        return True
    if any(clean.endswith(sfx) for sfx in _RP_SUFFIXES):
        return True
    if any(clean.endswith(sfx) for sfx in _GENITIVE_SUFFIXES):
        return True
    if any(clean.endswith(sfx) for sfx in _ADJ_LOCATIVE_SUFFIXES):
        return True
    if any(clean.endswith(sfx) for sfx in _ADVERBIAL_SUFFIXES):
        return True
    if any(clean.endswith(sfx) for sfx in _ADJ_SUFFIXES):
        return True
    return False


def _is_copula_token(token: str) -> bool:
    """Check if a token bears the copula ആണ്."""
    clean = _strip_punct(token)
    return bool(clean) and (clean.endswith("ാണ്") or clean.endswith("ആണ്"))


def _is_nominalized_verb(token: str) -> bool:
    """Check if a token is a nominalized verb form."""
    clean = _strip_punct(token)
    return bool(clean) and any(clean.endswith(sfx) for sfx in NOMINALIZATION_SUFFIXES)


def _is_subordinate_clause_boundary(token: str) -> bool:
    """Check if a token marks a subordinate clause boundary."""
    if token.endswith(",") or token.endswith(";"):
        return True
    clean = _strip_punct(token)
    return any(clean.endswith(marker) for marker in _SUBORDINATE_MARKERS)


# ═══════════════════════════════════════════════════════════════════════
# CleftReorderer
# ═══════════════════════════════════════════════════════════════════════

class CleftReorderer:
    """
    Clause-aware positional reorderer for Malayalam cleft sentences.

    Usage::

        pipeline  = AwesomeCleftPipeline()
        result    = pipeline.process(en, ml, en_focus=...)
        reorderer = CleftReorderer()
        variants  = reorderer.reorder(result)

        print(variants["preverbal_focus"])
        print(variants["postverbal_focus"])
        print(variants["clause_initial_focus"])
    """

    def reorder(self, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.

        Args:
            pipeline_result: The full dict returned by AwesomeCleftPipeline.process().

        Returns:
            Dict with keys:
                preverbal_focus      – [BG] + [Focus+ആണ്] + [NomVerb]
                postverbal_focus     – [BG] + [NomVerb]   + [Focus+ആണ്]
                clause_initial_focus – [Focus+ആണ്] + [BG] + [NomVerb]
                components           – structural decomposition breakdown
        """
        emphasized = pipeline_result.get("emphasized_malayalam_sentence", "")
        cleft_dict = pipeline_result.get("cleft_pipeline_output", {})
        focused_constituent = (
            pipeline_result.get("focused_malayalam_constituent", "")
            or cleft_dict.get("focused_constituent", "")
        )

        if not emphasized:
            return self._fallback_result(emphasized)

        # Step 1 ── Segment into clauses
        clauses = self._segment_clauses(emphasized)

        # Step 2 ── Find the cleft clause
        cleft_idx, has_nom_verb = self._find_cleft_clause(clauses, cleft_dict)

        if cleft_idx is None:
            return self._fallback_result(emphasized)

        # Step 3 ── Decompose the cleft clause
        pre_clauses  = clauses[:cleft_idx]
        cleft_tokens = clauses[cleft_idx]
        post_clauses = clauses[cleft_idx + 1:]

        decomp = self._decompose_cleft_clause(cleft_tokens, cleft_dict, focused_constituent)

        background   = decomp["background"]
        focus_copula = decomp["focus_copula"]
        nom_verb     = decomp["nominalized_verb"]
        trailing_p   = decomp["trailing_punct"]

        # Step 4 ── Produce 3 orderings
        preverbal = self._reassemble(
            pre_clauses, background, focus_copula, nom_verb,
            trailing_p, post_clauses, "preverbal",
        )
        postverbal = self._reassemble(
            pre_clauses, background, focus_copula, nom_verb,
            trailing_p, post_clauses, "postverbal",
        )
        clause_initial = self._reassemble(
            pre_clauses, background, focus_copula, nom_verb,
            trailing_p, post_clauses, "clause_initial",
        )

        return {
            "preverbal_focus": preverbal,
            "postverbal_focus": postverbal,
            "clause_initial_focus": clause_initial,
            "components": {
                "pre_clauses": [" ".join(c) for c in pre_clauses],
                "cleft_clause": {
                    "background": background,
                    "focus_copula": focus_copula,
                    "nominalized_verb": nom_verb,
                    "trailing_punct": trailing_p,
                },
                "post_clauses": [" ".join(c) for c in post_clauses],
            },
        }

    # ──────────────────────────────────────────────────────────────
    # 1. Clause Segmentation
    # ──────────────────────────────────────────────────────────────

    def _segment_clauses(self, sentence: str) -> List[List[str]]:
        """
        Segment sentence into clause token-lists.

        A clause boundary is signaled by a comma/semicolon or a subordinate
        marker (e.g. "ലഭ്യമാണെങ്കിലും", "വന്നതിനുശേഷം,").
        """
        tokens = sentence.split()
        clauses: List[List[str]] = []
        current: List[str] = []

        for token in tokens:
            current.append(token)
            if _is_subordinate_clause_boundary(token):
                clauses.append(current)
                current = []

        if current:
            clauses.append(current)

        return clauses

    # ──────────────────────────────────────────────────────────────
    # 2. Cleft Clause Identification
    # ──────────────────────────────────────────────────────────────

    def _find_cleft_clause(
        self, clauses: List[List[str]], cleft_dict: dict,
    ) -> Tuple[Optional[int], bool]:
        """
        Identify which clause index contains the cleft.

        Uses the copula form from cleft_dict as the primary anchor,
        falling back to surface suffix matching.

        Returns:
            (clause_index, has_nominalized_verb)
        """
        copula_form   = (cleft_dict.get("aanu_attachment") or {}).get("attached_form", "")
        nom_verb_form = cleft_dict.get("nominalized_verb", "")

        # ── Primary: exact match on known copula form ──
        copula_clause_idx = None
        for i, clause_tokens in enumerate(clauses):
            for token in clause_tokens:
                if copula_form and _strip_punct(token) == copula_form:
                    copula_clause_idx = i
                    break
            if copula_clause_idx is not None:
                break

        # ── Fallback: surface suffix match ──
        if copula_clause_idx is None:
            for i, clause_tokens in enumerate(clauses):
                for token in clause_tokens:
                    if _is_copula_token(token):
                        copula_clause_idx = i
                        break
                if copula_clause_idx is not None:
                    break

        if copula_clause_idx is None:
            return None, False

        # ── Check if nominalized verb lives in the same clause ──
        has_nom = False
        for token in clauses[copula_clause_idx]:
            clean = _strip_punct(token)
            if nom_verb_form and clean == nom_verb_form:
                has_nom = True
                break
            elif not nom_verb_form and _is_nominalized_verb(token):
                has_nom = True
                break

        return copula_clause_idx, has_nom

    # ──────────────────────────────────────────────────────────────
    # 3. Cleft Clause Decomposition
    # ──────────────────────────────────────────────────────────────

    def _decompose_cleft_clause(
        self, clause_tokens: List[str], cleft_dict: dict, focused_constituent_str: str = "",
    ) -> Dict[str, Any]:
        """
        Decompose a cleft clause's tokens into:

            background       – non-focus, non-verb tokens
            focus_copula     – focus constituent + ആണ് (preserves full multi-word NP)
            nominalized_verb – nominalized verb token(s)
            trailing_punct   – punctuation stripped from the clause's last token
        """
        tokens = list(clause_tokens)

        # ── Strip trailing punctuation from the last token ──
        trailing_punct = ""
        if tokens:
            last = tokens[-1]
            i = len(last)
            while i > 0 and last[i - 1] in _TRAILING_PUNCT:
                i -= 1
            if i < len(last):
                trailing_punct = last[i:]
                tokens[-1] = last[:i]
                if not tokens[-1]:
                    tokens.pop()

        # ── Locate copula-bearing token ──
        copula_form   = (cleft_dict.get("aanu_attachment") or {}).get("attached_form", "")
        nom_verb_form = cleft_dict.get("nominalized_verb", "")

        copula_idx = None
        for i, t in enumerate(tokens):
            if copula_form and _strip_punct(t) == copula_form:
                copula_idx = i
                break
        if copula_idx is None:
            for i, t in enumerate(tokens):
                if _is_copula_token(t):
                    copula_idx = i
                    break

        if copula_idx is None:
            return {
                "background": tokens,
                "focus_copula": [],
                "nominalized_verb": [],
                "trailing_punct": trailing_punct,
            }

        # ── Locate nominalized verb token ──
        nom_idx = None
        for i, t in enumerate(tokens):
            if i == copula_idx:
                continue
            clean = _strip_punct(t)
            if nom_verb_form and clean == nom_verb_form:
                nom_idx = i
                break
        if nom_idx is None:
            for i, t in enumerate(tokens):
                if i == copula_idx:
                    continue
                if _is_nominalized_verb(t):
                    nom_idx = i
                    break

        # ── Determine focus span width ──
        # Strategy A: Use the explicit focused_constituent words.
        # If the pipeline selected a multi-token constituent (e.g. "പ്രധാനമായും ഉത്തരേന്ത്യൻ ഭക്ഷണം"),
        # its words immediately precede copula_idx.
        fc_words = [
            _strip_punct(w).strip()
            for w in (focused_constituent_str or cleft_dict.get("focused_constituent", "")).split()
            if _strip_punct(w).strip()
        ]

        focus_start = copula_idx
        if len(fc_words) > 1:
            n_prefix = len(fc_words) - 1
            cand_start = copula_idx - n_prefix
            if cand_start >= 0:
                token_prefix = [_strip_punct(tokens[j]) for j in range(cand_start, copula_idx)]
                # Exact or case-stripped match with fc_words prefix
                if token_prefix == fc_words[:-1]:
                    focus_start = cand_start

        # Strategy B: If Strategy A didn't match (or single-word focus),
        # greedily absorb leftward NP-modifiers (adjectives, determiners, adverbs)
        if focus_start == copula_idx:
            k = copula_idx - 1
            while k >= 0:
                if nom_idx is not None and k == nom_idx:
                    break
                if _is_np_modifier(tokens[k]):
                    focus_start = k
                    k -= 1
                else:
                    break

        # Safety: Don't cross nominalized verb
        if nom_idx is not None and focus_start <= nom_idx < copula_idx:
            focus_start = nom_idx + 1

        # ── Build component lists ──
        focus_indices = set(range(focus_start, copula_idx + 1))
        nom_indices   = {nom_idx} if nom_idx is not None else set()

        focus_copula = [tokens[i] for i in sorted(focus_indices)]
        nom_verb     = [tokens[i] for i in sorted(nom_indices)]
        background   = [
            tokens[i] for i in range(len(tokens))
            if i not in focus_indices and i not in nom_indices
        ]

        return {
            "background": background,
            "focus_copula": focus_copula,
            "nominalized_verb": nom_verb,
            "trailing_punct": trailing_punct,
        }

    # ──────────────────────────────────────────────────────────────
    # 4. Sentence Reassembly
    # ──────────────────────────────────────────────────────────────

    def _reassemble(
        self,
        pre_clauses: List[List[str]],
        background: List[str],
        focus_copula: List[str],
        nom_verb: List[str],
        trailing_punct: str,
        post_clauses: List[List[str]],
        order: str,
    ) -> str:
        """
        Reassemble the full sentence with the cleft clause in the specified
        positional order, preserving pre/post clauses intact.

        Orders:
            preverbal      – [BG] + [Focus+ആണ്] + [NomVerb]
            postverbal     – [BG] + [NomVerb]   + [Focus+ആണ്]
            clause_initial – [Focus+ആണ്] + [BG] + [NomVerb]
        """
        # Determine cleft-clause token order
        if order == "preverbal":
            cleft = background + focus_copula + nom_verb
        elif order == "postverbal":
            if not nom_verb:
                # Copular sentence: no verb to reorder around
                cleft = background + focus_copula
            else:
                cleft = background + nom_verb + focus_copula
        elif order == "clause_initial":
            cleft = focus_copula + background + nom_verb
        else:
            cleft = background + focus_copula + nom_verb

        # Reattach trailing punctuation to the last token of the cleft clause
        if trailing_punct and cleft:
            cleft = list(cleft)
            cleft[-1] = cleft[-1] + trailing_punct

        # Assemble: pre-clauses → cleft clause → post-clauses
        parts: List[str] = []
        for clause in pre_clauses:
            parts.extend(clause)
        parts.extend(cleft)
        for clause in post_clauses:
            parts.extend(clause)

        return " ".join(parts)

    # ──────────────────────────────────────────────────────────────
    # Fallback
    # ──────────────────────────────────────────────────────────────

    def _fallback_result(self, sentence: str) -> Dict[str, Any]:
        """Return unchanged sentence for all orderings when decomposition fails."""
        return {
            "preverbal_focus": sentence,
            "postverbal_focus": sentence,
            "clause_initial_focus": sentence,
            "components": {
                "pre_clauses": [],
                "cleft_clause": {
                    "background": sentence.split() if sentence else [],
                    "focus_copula": [],
                    "nominalized_verb": [],
                    "trailing_punct": "",
                },
                "post_clauses": [],
            },
            "diagnostic": "Could not identify cleft components.",
        }
