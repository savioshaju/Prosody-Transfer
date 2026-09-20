"""
multi_target_emphasis_generator.py — Multi-Target Emphasis Candidate Generator.

Takes an English sentence and its Malayalam translation as input, discovers
all valid emphasis target candidates (N possibilities) across both word-level
and syntactic phrase-level constituents (NP, PP, AdvP, VP), determines whether
an established Malayalam emphasis strategy (Clefting & Copular fronting) can
realize each, and outputs rows with the 7 required columns matching
Synthetic_data - Sheet1.csv:

1. English Input
2. Malayalam Input
3. Malayalam After Emphasis
4. Strategy Used
5. Malayalam Emphasized Word/Phrase
6. English Emphasized Word/Phrase
7. English Prosody Labels
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Tuple, Set

# Reconfigure stdout for UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir) if os.path.basename(_curr_dir) == "cleft" else _curr_dir
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from aligner.alignment_utils import strip_punctuation, is_word_match
from cleft.phrase_chunker import MalayalamPhraseChunker, PhraseChunk, is_malayalam_verb
from cleft.copula_pipeline import attach_aanu_surface, split_punctuation
from cleft.cleft_pipeline import CleftPipeline
from verb_norm.verb_normalizer import VerbNormalizer
from aligner.morpho_simalign import MorphoSimAligner

DEFAULT_SPREADSHEET_ID = "1f1Ja8LXMzPhgy0rKDZP9MohyjPwVPkTRUCDmL6WXc5U"
DEFAULT_SERVICE_ACCOUNT_FILE = os.path.join(PROJECT_ROOT, "service_account.json")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Phonological Sandhi & Verb Normalization Explainer (Learned from Dataset)
# ═══════════════════════════════════════════════════════════════════════════

def explain_sandhi_rule(stem: str, aanu_form: str) -> str:
    """
    Generates the exact linguistic sandhi rule explanation for attaching 'ആണ്',
    matching the notation learned from Synthetic_data - Sheet1.csv.
    """
    clean_stem = strip_punctuation(stem).strip()
    clean_aanu = strip_punctuation(aanu_form).strip()
    if not clean_stem:
        return f"{stem} + ആണ് → {aanu_form}"

    # Year / Locative pattern: 2015 ലെ → 2015 ലാണ്
    if re.search(r"\d{1,4}\s*ലെ$", clean_stem) or clean_stem.endswith("ത്തിലെ") or clean_stem.endswith("ിലെ") or clean_stem.endswith("ത്തെ"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — Year + ലെ + ആണ് → Year ലാണ്" if re.search(r"\d{1,4}", clean_stem) else f"{clean_stem} + ആണ് → {clean_aanu} — ലെ + ആണ് → ലാണ്"

    # Chillu Consonants (Vyañjana-Svara Sandhi)
    if clean_stem.endswith("ൽ") or clean_stem.endswith("ല്"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — ൽ + ആണ്  →  ലാണ് (consonant ൽ → insert ല)"
    if clean_stem.endswith("ൾ") or clean_stem.endswith("ള്"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — ൾ + ആണ്  → ളാണ് (consonant ൾ → insert ള)"
    if clean_stem.endswith("ൺ") or clean_stem.endswith("ണ്"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — ൺ + ആണ്  → ണാണ്"
    if clean_stem.endswith("ൻ") or clean_stem.endswith("ന്"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — ൻ + ആണ്  → നാണ്"
    if clean_stem.endswith("ർ") or clean_stem.endswith("ര്"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — ർ + ആണ്  → രാണ്"

    # Anusvāra (ം → മാണ്)
    if clean_stem.endswith("ം"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — ം + ആണ്  → മാണ്"

    # Ablative case / geminate -നിന്ന് / -ൽനിന്ന്
    if clean_stem.endswith("നിന്ന്"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — C് + ആ → Cാ"

    # Accusative -നെ / -യെ
    if clean_stem.endswith("നെ"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — നെ + ആണ് → നെയാണ്( vowel എ → insert യ് )"
    if clean_stem.endswith("യെ"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — യെ + ആണ് → യെയാണ്( vowel എ → insert യ് )"

    # Generic Virāma deletion: C് + ആണ് → Cാണ്
    if clean_stem.endswith("്"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — C് + ആണ് → Cാണ്"

    # Adverbial / predicate complement -ായി
    if clean_stem.endswith("ായി"):
        return f"{clean_stem} + ആണ് → {clean_aanu} — യായി + ആണ് → യായിയാണ് (vowel ഇ → insert യ്)"

    # Rounded vowels (ഉ, ൂ, ഒ, ോ) -> Vakāra-āgamam (insert വ)
    if any(clean_stem.endswith(v) for v in ("ു", "ൂ", "ൊ", "ോ", "ഉ", "ഊ", "ഒ", "ഓ")):
        last_syll = clean_stem[-2:] if len(clean_stem) >= 2 else clean_stem
        return f"{clean_stem} + ആണ് → {clean_aanu} — {last_syll} + ആണ് → {clean_aanu[-3:]} (vowel ഉ → insert വ)"

    # Front/central vowels (ി, ീ, െ, േ, ാ, അ) -> Yakāra-āgamam (insert യ്)
    last_char = clean_stem[-1]
    return f"{clean_stem} + ആണ് → {clean_aanu} — {last_char} + ആണ് → {clean_aanu[-2:]} (vowel {last_char} → insert യ്)"


def explain_verb_norm_rule(verb_orig: str, verb_norm: str) -> str:
    """
    Generates the exact verb nominalization rule explanation,
    matching the notation learned from Synthetic_data - Sheet1.csv.
    """
    clean_v = strip_punctuation(verb_orig).strip()
    clean_vn = strip_punctuation(verb_norm).strip()
    if not clean_v or not clean_vn:
        return f"{verb_orig} → {verb_norm}"

    if clean_v.endswith("യിരുന്നു"):
        return f"{clean_v} → {clean_vn} — -യിരുന്നു → -യിരുന്ന + ത്"
    if clean_v.endswith("ുന്നു"):
        return f"{clean_v} → {clean_vn} — -ുന്നു → -ുന്ന + ത്"
    if clean_v.endswith("ി"):
        return f"{clean_v} → {clean_vn} — -ി → -ിയ + ത്"
    if clean_v.endswith("ത്തി") or clean_v.endswith("ത്തു"):
        return f"{clean_v} → {clean_vn} — ത്തി → ത്തിയ + ത്"
    if clean_v.endswith("ിച്ചു") or clean_v.endswith("ച്ചു"):
        return f"{clean_v} → {clean_vn} — -ച്ചു → -ച്ച + ത്"
    if clean_v.endswith("ഞ്ഞു"):
        return f"{clean_v} → {clean_vn} — -ഞ്ഞു → -ഞ്ഞ + ത്"
    if clean_v.endswith("ന്നു"):
        return f"{clean_v} → {clean_vn} — -ന്നു → -ന്ന + ത്"
    if clean_v.endswith("ണം") or clean_v.endswith("ൂ"):
        return f"{clean_v} → {clean_vn} — -ണം → -േണ്ടത്"
    if clean_v.endswith("ും"):
        return f"{clean_v} → {clean_vn} — -ും → -ുന്നത്"

    return f"{clean_v} → {clean_vn} — + ത്"


# ═══════════════════════════════════════════════════════════════════════════
# 1b. Phonetic Transliteration & Loanword Matching
# ═══════════════════════════════════════════════════════════════════════════

MALAYALAM_CONSONANT_MAP = {
    'ക': 'k', 'ഖ': 'k', 'ഗ': 'g', 'ഘ': 'g', 'ങ': 'ng',
    'ച': 'c', 'ഛ': 'c', 'ജ': 'j', 'ഝ': 'j', 'ഞ': 'nj',
    'ട': 't', 'ഠ': 't', 'ഡ': 'd', 'ഢ': 'd', 'ണ': 'n',
    'ത': 't', 'ഥ': 't', 'ദ': 'd', 'ധ': 'd', 'ന': 'n',
    'പ': 'p', 'ഫ': 'p', 'ബ': 'b', 'ഭ': 'b', 'മ': 'm',
    'യ': 'y', 'ര': 'r', 'റ': 'r', 'ല': 'l', 'ള': 'l', 'ഴ': 'z',
    'വ': 'v', 'ശ': 's', 'ഷ': 's', 'സ': 's', 'ഹ': 'h',
    'ൽ': 'l', 'ൾ': 'l', 'ർ': 'r', 'ൺ': 'n', 'ൻ': 'n', 'ൿ': 'k', 'ം': 'm',
}

def ml_consonant_skeleton(word: str) -> str:
    res = []
    for ch in word:
        if ch in MALAYALAM_CONSONANT_MAP:
            res.append(MALAYALAM_CONSONANT_MAP[ch])
    dedup = []
    for ch in res:
        if not dedup or dedup[-1] != ch:
            dedup.append(ch)
    return "".join(dedup)

def en_consonant_skeleton(word: str) -> str:
    w = word.lower()
    w = re.sub(r'ph', 'f', w)
    w = re.sub(r'th', 't', w)
    w = re.sub(r'c(?=[eiy])', 's', w)
    w = re.sub(r'c', 'k', w)
    w = re.sub(r'q', 'k', w)
    w = re.sub(r'w', 'v', w)
    w = re.sub(r'z', 's', w)
    res = [ch for ch in w if ch in 'bdfghjklmnprstvy']
    dedup = []
    for ch in res:
        if not dedup or dedup[-1] != ch:
            dedup.append(ch)
    return "".join(dedup)

def is_phonetic_translit_match(en_w: str, ml_w: str) -> bool:
    en_s = en_consonant_skeleton(en_w)
    ml_s = ml_consonant_skeleton(ml_w)
    if not en_s or not ml_s:
        return False
    if en_s == ml_s:
        return True
    if len(en_s) >= 3 and len(ml_s) >= 3:
        if en_s in ml_s or ml_s in en_s:
            return True
        en_no_y = en_s.replace('y', '').replace('v', '')
        ml_no_y = ml_s.replace('y', '').replace('v', '')
        if en_no_y and en_no_y == ml_no_y:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 2. Candidate Target Extraction & Filtering
# ═══════════════════════════════════════════════════════════════════════════

class EmphasisTargetCandidateExtractor:
    """
    Extracts all candidate emphasis constituents at word and syntactic phrase levels.
    Enforces Left Branch Condition and Postposition Stranding Ban.
    """
    def __init__(self):
        self.ml_chunker = MalayalamPhraseChunker()

    def extract_candidates(self, ml_sentence: str) -> List[Dict[str, Any]]:
        """
        Extracts syntactic phrase chunks (NP, PP, AdvP), noun compounds, and
        content words from a Malayalam sentence.
        """
        raw_tokens = ml_sentence.split()
        if not raw_tokens:
            return []

        clean_tokens = [strip_punctuation(t) for t in raw_tokens]
        n = len(raw_tokens)

        chunks = self.ml_chunker.chunk_sentence(raw_tokens, sentence_text=ml_sentence)
        candidates: List[Dict[str, Any]] = []
        seen_spans: Set[Tuple[int, int]] = set()

        # Identify sentence-final finite verb index to protect it from clefting
        final_verb_idx = None
        for k in range(n - 1, -1, -1):
            if is_malayalam_verb(clean_tokens[k]):
                final_verb_idx = k
                break
        if final_verb_idx is None:
            final_verb_idx = n - 1

        def add_cand(s: int, e: int, level: str, c_type: str):
            if s < 0 or e >= n or s > e:
                return
            # Do not cleft the matrix verb (it must be nominalized instead)
            if s == final_verb_idx and e == final_verb_idx:
                return
            # Prevent postposition / case particle stranding
            # e.g., if token at e+1 is 'ലെ', 'ന്റെ', 'ുടെ', cannot strand it
            if e + 1 < n and clean_tokens[e + 1] in ("ലെ", "ന്റെ", "ുടെ", "ിന്റെ", "്റെ"):
                return
            # Standalone particle cannot be clefted alone
            if s == e and clean_tokens[s] in ("ലെ", "ന്റെ", "ുടെ", "ിന്റെ", "്റെ", "ആണ്", "ആയിരുന്നു", "ഒരു", "എന്ന"):
                return

            span = (s, e)
            if span not in seen_spans:
                seen_spans.add(span)
                cand_text = " ".join(clean_tokens[s : e + 1])
                candidates.append({
                    "text": cand_text,
                    "start_token": s,
                    "end_token": e,
                    "level": level,
                    "type": c_type,
                })

        ADNOMINAL_SUFFIXES = ("യിലെ", "ത്തിലെ", "ിലെ", "ന്റെ", "ിന്റെ", "്റെ", "ുടെ", "ക്കൻ")
        CASE_TERMINAL_SUFFIXES = ("ൽ", "ിൽ", "ത്തിൽ", "ത്ത്", "ന്", "ിന്", "ക്ക്", "യ്ക്ക്", "നെ", "യെ", "ആൽ", "കൊണ്ട്", "ഓട്")

        def ends_with_any(w: str, sfxs: tuple) -> bool:
            clean = strip_punctuation(w)
            return any(clean.endswith(s) for s in sfxs)

        # 1. Syntactic Phrase Chunks (NP, PP, AdvP) from Chunker
        for c in chunks:
            if c.chunk_type == "VP":
                continue
            add_cand(c.start_idx, c.end_idx, "phrase", c.chunk_type)

        # 2. Adnominal modifier pairs: word + 'ലെ' (e.g. '2015 ലെ')
        for i in range(n - 1):
            if clean_tokens[i + 1] == "ലെ":
                add_cand(i, i + 1, "phrase", "NP")

        # 3. Adnominal Modifiers + Head Noun / PP (e.g. 'ഇന്ത്യയിലെ ഡൽഹിയിൽ', 'വടക്കൻ അരിസോണയിൽ')
        for i in range(n - 1):
            if i != final_verb_idx and i + 1 != final_verb_idx:
                w1, w2 = clean_tokens[i], clean_tokens[i + 1]
                if w1 != "ലെ" and (ends_with_any(w1, ADNOMINAL_SUFFIXES) or w1 in ("വടക്കൻ", "തെക്കൻ", "കിഴക്കൻ", "പടിഞ്ഞാറൻ")) and not is_malayalam_verb(w2):
                    add_cand(i, i + 1, "phrase", "PP" if ends_with_any(w2, CASE_TERMINAL_SUFFIXES) else "NP")

        # 4. Noun Compounds (2 to 3 word nominal sequences within an NP)
        for i in range(n - 1):
            if i != final_verb_idx and i + 1 != final_verb_idx:
                w1, w2 = clean_tokens[i], clean_tokens[i + 1]
                if (
                    w1 not in ("ലെ", "ന്റെ", "ുടെ")
                    and not ends_with_any(w1, CASE_TERMINAL_SUFFIXES)
                    and not ends_with_any(w1, ADNOMINAL_SUFFIXES)
                    and not ends_with_any(w2, ("ി", "ു", "ുന്നു", "ത്തു", "പ്പെട്ടു"))
                    and not is_malayalam_verb(w2)
                    and not ends_with_any(w2, ADNOMINAL_SUFFIXES)
                ):
                    add_cand(i, i + 1, "phrase", "NP")

        for i in range(n - 2):
            if i + 2 != final_verb_idx:
                w1, w2, w3 = clean_tokens[i], clean_tokens[i + 1], clean_tokens[i + 2]
                if (
                    not ends_with_any(w1, CASE_TERMINAL_SUFFIXES)
                    and not ends_with_any(w1, ADNOMINAL_SUFFIXES)
                    and not ends_with_any(w2, CASE_TERMINAL_SUFFIXES)
                    and not ends_with_any(w2, ADNOMINAL_SUFFIXES)
                    and not is_malayalam_verb(w3)
                    and not ends_with_any(w3, ADNOMINAL_SUFFIXES)
                ):
                    add_cand(i, i + 2, "phrase", "NP")

        # 5. Individual Content Words (Nouns, Named Entities, Adverbs, Temporal words)
        skip_words = frozenset({
            "ആണ്", "ആയിരുന്നു", "ഉണ്ട്", "ഉണ്ടായിരുന്നു", "അല്ല", "ഇല്ല",
            "എന്ന", "എന്ന്", "ഒരു", "എന്നാൽ", "പക്ഷേ", "കൂടി", "മാത്രം", "ലെ"
        })
        for i in range(n):
            w = clean_tokens[i]
            if w and w not in skip_words and i != final_verb_idx:
                add_cand(i, i, "word", "WORD")

        # Sort candidates in structural order of appearance, tighter spans first
        candidates.sort(key=lambda x: (x["start_token"], x["end_token"]))
        return candidates


# ═══════════════════════════════════════════════════════════════════════════
# 3. Main Multi-Target Emphasis Generator Class
# ═══════════════════════════════════════════════════════════════════════════

def map_tokens_to_words(tokens: List[str], words: List[str]) -> List[int]:
    """
    Maps fine-grained aligner token indices to corresponding coarse whitespace word indices.
    """
    token_to_word = []
    w_idx = 0
    for tok in tokens:
        c_tok = strip_punctuation(tok).lower()
        matched_w = None
        if c_tok:
            for j in range(w_idx, len(words)):
                c_word = strip_punctuation(words[j]).lower()
                if c_tok in c_word:
                    matched_w = j
                    w_idx = j
                    break
        if matched_w is None:
            matched_w = min(w_idx, len(words) - 1) if words else 0
        token_to_word.append(matched_w)
    return token_to_word


class MultiTargetEmphasisGenerator:
    """
    End-to-End Multi-Target Emphasis Generator.
    """
    def __init__(self, aligner_type: str = "morpho", aligner_model: Optional[str] = None):
        self.aligner_type = (aligner_type or "morpho").lower()
        self.aligner = MorphoSimAligner(model_name=aligner_model or "bert-base-multilingual-cased")

        self.cleft_pipeline = CleftPipeline()
        self.verb_normalizer = VerbNormalizer()
        self.candidate_extractor = EmphasisTargetCandidateExtractor()

    def _project_ml_to_en(
        self,
        ml_start_idx: int,
        ml_end_idx: int,
        en_tokens: List[str],
        ml_tokens: List[str],
        alignments: List[Dict[str, Any]],
        align_src_tokens: Optional[List[str]] = None,
        align_tgt_tokens: Optional[List[str]] = None,
    ) -> Tuple[str, List[int]]:
        """
        Maps a Malayalam constituent span [ml_start_idx..ml_end_idx] to the
        corresponding contiguous English token span and word-level prosody labels.
        """
        token_to_word_en = (
            map_tokens_to_words(align_src_tokens, en_tokens)
            if align_src_tokens and len(align_src_tokens) != len(en_tokens)
            else list(range(len(en_tokens)))
        )
        token_to_word_ml = (
            map_tokens_to_words(align_tgt_tokens, ml_tokens)
            if align_tgt_tokens and len(align_tgt_tokens) != len(ml_tokens)
            else list(range(len(ml_tokens)))
        )

        target_ml_indices = set(range(ml_start_idx, ml_end_idx + 1))
        matched_en_indices = set()

        # 1. First pass: alignment-based mapping (excluding synthetic CASE_BIND prepositions)
        for a in alignments:
            if str(a.get("matched_stem", "")).startswith("[CASE_BIND:"):
                continue

            tgt_t_idx = a.get("tgt_index")
            src_t_idx = a.get("src_index")
            if tgt_t_idx is None or src_t_idx is None:
                continue

            tgt_w_idx = token_to_word_ml[tgt_t_idx] if 0 <= tgt_t_idx < len(token_to_word_ml) else tgt_t_idx
            src_w_idx = token_to_word_en[src_t_idx] if 0 <= src_t_idx < len(token_to_word_en) else src_t_idx

            if tgt_w_idx in target_ml_indices and 0 <= src_w_idx < len(en_tokens):
                src_word = strip_punctuation(en_tokens[src_w_idx]).lower()
                if src_word not in ("is", "are", "was", "were", "been", "held", "be"):
                    matched_en_indices.add(src_w_idx)

        # 2. Phonetic / Transliteration Anchor Recovery for unaligned or misaligned target tokens
        for ml_idx in target_ml_indices:
            ml_w = ml_tokens[ml_idx]
            for en_idx, en_w in enumerate(en_tokens):
                if is_phonetic_translit_match(en_w, ml_w):
                    matched_en_indices.add(en_idx)

        # 3. Monotonic syntactic modifier expansion for multi-word constituents
        if matched_en_indices:
            min_en = min(matched_en_indices)
            ml_len = ml_end_idx - ml_start_idx + 1
            if ml_len > len(matched_en_indices) and min_en > 0:
                prev_en = strip_punctuation(en_tokens[min_en - 1]).lower()
                if prev_en not in ("in", "at", "on", "to", "from", "of", "by", "with", "during", "the", "a", "an", "were", "was", "is", "are"):
                    matched_en_indices.add(min_en - 1)

        # If all were auxiliaries or empty, try all matched
        if not matched_en_indices:
            for a in alignments:
                tgt_t_idx = a.get("tgt_index")
                src_t_idx = a.get("src_index")
                if tgt_t_idx is None or src_t_idx is None:
                    continue
                tgt_w_idx = token_to_word_ml[tgt_t_idx] if 0 <= tgt_t_idx < len(token_to_word_ml) else tgt_t_idx
                src_w_idx = token_to_word_en[src_t_idx] if 0 <= src_t_idx < len(token_to_word_en) else src_t_idx
                if tgt_w_idx in target_ml_indices and 0 <= src_w_idx < len(en_tokens):
                    matched_en_indices.add(src_w_idx)

        # Fallback to positional ratio if alignment is totally empty
        if not matched_en_indices and ml_tokens and en_tokens:
            ml_ratio = (ml_start_idx + ml_end_idx) / (2.0 * max(1, len(ml_tokens) - 1))
            approx_en_idx = int(round(ml_ratio * (len(en_tokens) - 1)))
            matched_en_indices.add(min(approx_en_idx, len(en_tokens) - 1))

        if not matched_en_indices:
            return "", [0] * len(en_tokens)

        # Cluster matched indices to eliminate outlier jumps across the sentence
        sorted_idx = sorted(matched_en_indices)
        clusters: List[List[int]] = [[sorted_idx[0]]]
        for x in sorted_idx[1:]:
            if x <= clusters[-1][-1] + 2:  # allow 1 intervening word (e.g. preposition or 'of')
                clusters[-1].append(x)
            else:
                clusters.append([x])

        # Pick the most prominent / longest cluster
        clusters.sort(key=lambda c: len(c), reverse=True)
        best_cluster = sorted(clusters[0])
        min_idx = best_cluster[0]
        max_idx = best_cluster[-1]

        # Preposition stripping: if min_idx < max_idx and leading token is a preposition
        # and Malayalam constituent has no overt postposition, strip it
        OVERT_POSTPOSITIONS = {"വെച്ച്", "വച്ച്", "ഉള്ളിൽ", "അകത്ത്", "പുറത്ത്", "കീഴിൽ", "മുകളിൽ", "നേരെ", "വരെ"}
        has_postpos = any(strip_punctuation(ml_tokens[i]) in OVERT_POSTPOSITIONS for i in range(ml_start_idx, ml_end_idx + 1))
        if min_idx < max_idx and not has_postpos:
            first_w = strip_punctuation(en_tokens[min_idx]).lower()
            if first_w in ("in", "at", "on", "to", "from", "of", "by", "with", "during"):
                min_idx += 1

        # Expand to leading determiner (The, A, An) if present at clause/NP start
        if min_idx > 0 and en_tokens[min_idx - 1].lower() in ("the", "a", "an") and min_idx - 1 not in matched_en_indices:
            if ml_start_idx == 0 or (min_idx == 1 and ml_start_idx <= 1):
                min_idx -= 1

        # Form clean English target phrase
        en_span_tokens = en_tokens[min_idx : max_idx + 1]
        en_phrase = " ".join(en_span_tokens)
        clean_en_phrase = strip_punctuation(en_phrase).strip()

        # Binary prosody labels: 1 for tokens in the emphasized span, 0 otherwise
        prosody_labels = [1 if min_idx <= i <= max_idx else 0 for i in range(len(en_tokens))]

        return clean_en_phrase, prosody_labels

    def process_pair(self, english_sentence: str, malayalam_sentence: str) -> List[Dict[str, Any]]:
        """
        Generates all valid emphasis targets (N possibilities) for an English-Malayalam sentence pair.
        Returns a list of dicts with the 7 required columns:
          1. English Input
          2. Malayalam Input
          3. Malayalam After Emphasis
          4. Strategy Used
          5. Malayalam Emphasized Word/Phrase
          6. English Emphasized Word/Phrase
          7. English Prosody Labels (bracketed list string)
        and matching aliases for Synthetic_data - Sheet1.csv (Source_Prosody_labels, etc.).
        """
        clean_en = english_sentence.strip()
        clean_ml = malayalam_sentence.strip()
        if not clean_en or not clean_ml:
            return []

        en_tokens = clean_en.split()
        ml_tokens = clean_ml.split()

        # 1. Compute alignments (English -> Malayalam)
        align_res = self.aligner.align(clean_en, clean_ml)
        alignments = align_res.get("reconciled_pairs") or align_res.get("src_to_tgt_alignments") or []
        align_src_tokens = align_res.get("source_tokens") or []
        align_tgt_tokens = align_res.get("target_tokens") or []

        # 2. Extract candidate emphasis targets
        candidates = self.candidate_extractor.extract_candidates(clean_ml)

        successful_rows: List[Dict[str, Any]] = []
        seen_targets: Set[Tuple[str, str]] = set()

        for cand in candidates:
            c_text = cand["text"]
            s_idx = cand["start_token"]
            e_idx = cand["end_token"]

            # Construct tagged sentence with <FF> around constituent
            tagged_tokens = []
            for i, tok in enumerate(ml_tokens):
                if i == s_idx:
                    tagged_tokens.append(f"<FF>{tok}")
                elif i == e_idx:
                    tagged_tokens.append(f"{tok}<FF>")
                else:
                    tagged_tokens.append(tok)
            if s_idx == e_idx:
                tagged_tokens[s_idx] = f"<FF>{ml_tokens[s_idx]}<FF>"

            tagged_sent = " ".join(tagged_tokens)

            # Process via CleftPipeline
            res = self.cleft_pipeline.process(tagged_sent)
            if res.status != "VALID":
                continue

            cleft_sent = res.cleft_sentence
            if not cleft_sent or cleft_sent == clean_ml:
                continue

            # Determine emphasized word/phrase in Malayalam
            aanu_focus = attach_aanu_surface(c_text)

            # Project to English
            en_target, prosody_labels = self._project_ml_to_en(
                s_idx, e_idx, en_tokens, ml_tokens, alignments,
                align_src_tokens=align_src_tokens,
                align_tgt_tokens=align_tgt_tokens,
            )
            if not en_target:
                continue

            # Deduplicate by target pair (Malayalam Focus Form, English Target Phrase)
            target_key = (aanu_focus, en_target)
            if target_key in seen_targets:
                continue
            seen_targets.add(target_key)

            # Determine main verb and nominalized verb
            main_verb = getattr(res, "main_verb", "") or ""
            if not main_verb and ml_tokens:
                main_verb = strip_punctuation(ml_tokens[-1])

            # Extract nominalized verb from the end of cleft_sent or compute via verb_normalizer
            cleft_tokens = cleft_sent.split()
            last_cleft_tok = strip_punctuation(cleft_tokens[-1]) if cleft_tokens else ""
            if last_cleft_tok.endswith("ത്"):
                nom_verb = last_cleft_tok
            else:
                v_res = self.verb_normalizer.normalize(main_verb)
                nom_verb = v_res.get("normalized") or main_verb

            # Build Strategy Used string matching Synthetic_data format:
            # Focus : <Stem> + ആണ് → <AanuForm> — <Sandhi Rule>
            # Verb: <VerbOrig> → <VerbNorm> — <Norm Rule>
            focus_line = f"Focus : {explain_sandhi_rule(c_text, aanu_focus)}"
            verb_line = f"Verb: {explain_verb_norm_rule(main_verb, nom_verb)}"
            strategy_used = f"{focus_line}\n{verb_line}"

            # Format Python list string matching Synthetic_data - Sheet1.csv: "[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]"
            prosody_labels_str = str(prosody_labels)

            successful_rows.append({
                "Source_Sentence": clean_en,
                "Malayalam Input": clean_ml,
                "Target_Sentence": cleft_sent,
                "Prosody_strategy (Emphasis/rising contour)" : "Emphasis",
                "Strategy_in_Target": strategy_used,
                "Target_Prosody_word": aanu_focus,
                "Source_Prosody_word": en_target,
                "Source_Prosody_labels": prosody_labels_str,
            })

        return successful_rows

    def export_to_google_sheet(
        self,
        rows_or_df: Any,
        spreadsheet_id_or_url: str = DEFAULT_SPREADSHEET_ID,
        credentials_path: str = DEFAULT_SERVICE_ACCOUNT_FILE,
        sheet_name: Optional[str] = None,
        mode: str = "overwrite",
    ) -> bool:
        """Directly writes generated emphasis candidate rows to the Google Sheet."""
        return export_to_google_sheet(
            rows_or_df,
            spreadsheet_id_or_url=spreadsheet_id_or_url,
            credentials_path=credentials_path,
            sheet_name=sheet_name,
            mode=mode,
        )

    def process_dataset(
        self,
        input_csv_or_df: Any,
        output_csv: Optional[str] = None,
        gsheet_id_or_url: Optional[str] = None,
        en_col: str = "eng",
        ml_col: str = "mal",
        max_rows: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Batch-processes a CSV or DataFrame of English-Malayalam pairs and returns a DataFrame
        with all successful emphasis target rows.
        """
        if isinstance(input_csv_or_df, str):
            df = pd.read_csv(input_csv_or_df)
        else:
            df = input_csv_or_df.copy()

        # Find columns
        en_candidates = [c for c in df.columns if "eng" in c.lower() or "source" in c.lower()]
        ml_candidates = [c for c in df.columns if "mal" in c.lower() or "target" in c.lower() or "s1" in c.lower()]
        actual_en = en_candidates[0] if en_candidates else df.columns[0]
        actual_ml = ml_candidates[0] if ml_candidates else df.columns[1]

        all_rows = []
        limit = max_rows if max_rows is not None else len(df)
        subset = df.iloc[:limit]

        for i, (_, row) in enumerate(subset.iterrows()):
            en_sent = str(row[actual_en]).strip()
            ml_sent = str(row[actual_ml]).strip()
            if not en_sent or not ml_sent or en_sent == "nan" or ml_sent == "nan":
                continue
            rows = self.process_pair(en_sent, ml_sent)
            all_rows.extend(rows)

        out_df = pd.DataFrame(all_rows)
        if output_csv:
            out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"Saved {len(out_df)} emphasis candidate rows to local CSV: {output_csv}")

        if gsheet_id_or_url:
            export_to_google_sheet(out_df, spreadsheet_id_or_url=gsheet_id_or_url)

        return out_df


# ═══════════════════════════════════════════════════════════════════════════
# 4. Google Sheets Direct Export Integration
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SPREADSHEET_ID = "1f1Ja8LXMzPhgy0rKDZP9MohyjPwVPkTRUCDmL6WXc5U"
DEFAULT_SERVICE_ACCOUNT_FILE = os.path.join(PROJECT_ROOT, "service_account.json")


def export_to_google_sheet(
    rows_or_df: Any,
    spreadsheet_id_or_url: str = DEFAULT_SPREADSHEET_ID,
    credentials_path: str = DEFAULT_SERVICE_ACCOUNT_FILE,
    sheet_name: Optional[str] = None,
    mode: str = "overwrite",
    max_retries: int = 3,
) -> bool:
    """
    Directly writes generated emphasis candidates to a Google Spreadsheet using
    the provided service account credentials.
    Supports both raw spreadsheet IDs and full Google Docs URLs.
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        print(">>> Warning: 'gspread' or 'google-auth' not installed. Cannot write to Google Sheets.")
        return False

    if not os.path.exists(credentials_path):
        print(f">>> Warning: Service account credentials file not found at: {credentials_path}")
        return False

    # Extract ID if a full Google Sheets URL was passed
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", spreadsheet_id_or_url)
    sheet_id = m.group(1) if m else spreadsheet_id_or_url.strip()

    try:
        creds = Credentials.from_service_account_file(
            credentials_path,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"
            ]
        )
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(sheet_name) if sheet_name else sh.get_worksheet(0)
    except Exception as e:
        print(f">>> [GoogleSheet] Authorization or connection failed: {e}")
        return False

    if isinstance(rows_or_df, pd.DataFrame):
        df = rows_or_df.copy()
    else:
        df = pd.DataFrame(rows_or_df)

    if df.empty:
        print(">>> [GoogleSheet] No rows to export.")
        return True

    headers = list(df.columns)
    data_rows = [[("" if pd.isna(x) else str(x)) for x in row] for row in df.values]
    full_matrix = [headers] + data_rows

    import time
    for attempt in range(max_retries):
        try:
            if mode == "append":
                existing = ws.get_all_values()
                if not existing:
                    ws.update(values=full_matrix, range_name="A1")
                else:
                    ws.append_rows(data_rows)
            else:
                # Overwrite sheet with full matrix
                ws.clear()
                ws.update(values=full_matrix, range_name="A1")
            print(f">>> Successfully exported {len(df)} rows to Google Sheet: '{sh.title}' ({ws.title})")
            print(f">>> View sheet at: https://docs.google.com/spreadsheets/d/{sheet_id}/edit")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f">>> [GoogleSheet] Attempt {attempt + 1} encountered: {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                print(f">>> [GoogleSheet] Failed to write to Google Sheets after {max_retries} attempts: {e}")
                return False
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Target Emphasis Candidate Generator")
    parser.add_argument("--en", type=str, default="", help="English input sentence")
    parser.add_argument("--ml", type=str, default="", help="Malayalam input sentence")
    parser.add_argument("--csv", type=str, default="", help="Path to input CSV file")
    parser.add_argument("--output", "-out", type=str, default="multi_target_output.csv", help="Output CSV path")
    parser.add_argument("--no-local", action="store_true", help="Do not write locally, write only to Google Sheets")
    parser.add_argument("--gsheet", type=str, default=DEFAULT_SPREADSHEET_ID, help="Google Spreadsheet ID or URL (default: linked doc)")
    parser.add_argument("--no-gsheet", action="store_true", help="Disable writing directly to Google Sheets")
    parser.add_argument("--creds", type=str, default=DEFAULT_SERVICE_ACCOUNT_FILE, help="Path to service account JSON")
    parser.add_argument("--aligner", type=str, default="morpho", choices=["morpho"])
    args = parser.parse_args()

    generator = MultiTargetEmphasisGenerator(aligner_type=args.aligner)

    gsheet_target = None if args.no_gsheet else args.gsheet

    if args.csv:
        out_csv = None if args.no_local else args.output
        generator.process_dataset(args.csv, output_csv=out_csv, gsheet_id_or_url=gsheet_target)
    else:
        en = args.en or "The 2015 Commonwealth Karate Games were held in Delhi, India."
        ml = args.ml or "2015 ലെ കോമൺവെൽത്ത് കരാട്ടെ ഗെയിംസ് ഇന്ത്യയിലെ ഡൽഹിയിൽ നടത്തി."
        print("=" * 70)
        print("English Input  :", en)
        print("Malayalam Input:", ml)
        print("=" * 70)
        rows = generator.process_pair(en, ml)
        print(f"\nGenerated {len(rows)} successful emphasis target candidates:\n")
        for idx, r in enumerate(rows, 1):
            print(f"[{idx}] Focus: {r['Target_Prosody_word']} ↔ {r['Source_Prosody_word']}")
            print(f"    Labels: {r['Source_Prosody_labels']}")
            print(f"    Cleft : {r['Target_Sentence']}")
            print(f"    Strategy:\n    " + r['Strategy_in_Target'].replace('\n', '\n    '))
            print("-" * 70)

        df = pd.DataFrame(rows)
        if not args.no_local:
            df.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"\nSaved local results to: {args.output}")

        if gsheet_target:
            print("\nWriting directly to Google Sheet...")
            export_to_google_sheet(df, spreadsheet_id_or_url=gsheet_target, credentials_path=args.creds)


if __name__ == "__main__":
    main()
