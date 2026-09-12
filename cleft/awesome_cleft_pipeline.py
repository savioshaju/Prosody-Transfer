"""
awesome_cleft_pipeline.py — End-to-End Cross-Lingual Focus Transfer & Alignment Pipeline.

Implements the complete architecture:

English sentence + Malayalam sentence + emphasis marker
                    ↓
              Awesome-Aligner
                    ↓
        English → Malayalam alignment (Focus Projection)
                    ↓
             Clefting pipeline (Bhashik / SSF Pipeline)
                    ↓
       Emphasized Malayalam sentence
                    ↓
              Awesome-Aligner
             ↙              ↘
Malayalam → English     English → Malayalam
alignment               alignment
"""

import os
import sys
import re
import itertools
from typing import Dict, Any, List, Optional, Tuple

# Ensure UTF-8 console output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add awesome-align path and venv site-packages
_curr_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_curr_dir) if os.path.basename(_curr_dir) == "cleft" else _curr_dir
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
VENV_SITE_PACKAGES = os.path.join(PROJECT_ROOT, "venv", "Lib", "site-packages")
if os.path.exists(VENV_SITE_PACKAGES) and VENV_SITE_PACKAGES not in sys.path:
    sys.path.append(VENV_SITE_PACKAGES)

AWESOME_ALIGN_DIR = os.path.join(PROJECT_ROOT, "awesome-align")
if AWESOME_ALIGN_DIR not in sys.path:
    sys.path.append(AWESOME_ALIGN_DIR)

import torch
from aligner.alignment_utils import extract_alignment_and_prosody, strip_punctuation, is_word_match
from aligner.tokenizer import tokenize_malayalam
from aligner.simalign_wrapper import SimAlignerWrapper
from cleft.bhashik_focus_reorderer import BhashikFocusReorderer
from cleft.constituency_reorderer import ConstituencyReorderer
from translators.bhashaverse_translator import BhashaverseTranslator
from translators.krutrim_translator import KrutrimTranslator
from cleft.cleft_pipeline import CleftPipeline, POSTPOSITIONS


class AwesomeAlignerWrapper:
    """Wrapper around awesome-align BERT-based word alignment model."""

    def __init__(self, model_name: str = "bert-base-multilingual-cased", device: Optional[str] = None):
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._loaded = False
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        if self._loaded:
            return
        try:
            from awesome_align import modeling
            from awesome_align.modeling import BertForMaskedLM
            from awesome_align.tokenization_bert import BertTokenizer

            try:
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = BertForMaskedLM.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                self.tokenizer = BertTokenizer.from_pretrained(self.model_name, local_files_only=False)
                self.model = BertForMaskedLM.from_pretrained(self.model_name, local_files_only=False)

            modeling.PAD_ID = self.tokenizer.pad_token_id
            modeling.CLS_ID = self.tokenizer.cls_token_id
            modeling.SEP_ID = self.tokenizer.sep_token_id

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            
        except Exception as e:
            print(f">>> [Awesome-Aligner] Note: Neural awesome-align unavailable ({e}). Using cross-lingual dictionary & positional aligner.")
            self._loaded = False

    def _run_bert_align(self, sent_src: List[str], sent_tgt: List[str]) -> List[Tuple[int, int]]:
        """Run awesome-align BERT model on given token lists."""
        try:
            self._load_model()
        except Exception:
            self._loaded = False

        if self._loaded and self.model and self.tokenizer:
            try:
                token_src = [self.tokenizer.tokenize(w) for w in sent_src]
                token_tgt = [self.tokenizer.tokenize(w) for w in sent_tgt]

                wid_src = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src]
                wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

                ids_src = self.tokenizer.prepare_for_model(
                    list(itertools.chain(*wid_src)),
                    return_tensors="pt",
                    max_length=self.tokenizer.max_len,
                )["input_ids"]
                ids_tgt = self.tokenizer.prepare_for_model(
                    list(itertools.chain(*wid_tgt)),
                    return_tensors="pt",
                    max_length=self.tokenizer.max_len,
                )["input_ids"]

                bpe2word_map_src = [i for i, w in enumerate(token_src) for _ in w]
                bpe2word_map_tgt = [i for i, w in enumerate(token_tgt) for _ in w]

                with torch.no_grad():
                    word_aligns_list = self.model.get_aligned_word(
                        ids_src,
                        ids_tgt,
                        [bpe2word_map_src],
                        [bpe2word_map_tgt],
                        self.device,
                        0,
                        0,
                        align_layer=8,
                        extraction="softmax",
                        softmax_threshold=0.001,
                        test=True,
                        output_prob=False,
                    )

                return word_aligns_list[0]
            except Exception:
                return self._heuristic_align(sent_src, sent_tgt)
        else:
            return self._heuristic_align(sent_src, sent_tgt)



    def _tokenize(self, text: str) -> List[str]:
        """Universal word + punctuation tokenizer for English and Malayalam."""
        if not text:
            return []
        pattern = r"[\w\u0D00-\u0D7F\u200C\u200D]+(?:[.-][\w\u0D00-\u0D7F\u200C\u200D]+)*|[.,!?;:()\[\]\"'“”‘’]"
        tokens = re.findall(pattern, text)
        return [t for t in tokens if t.strip()]

    def align(self, src_sentence: str, tgt_sentence: str) -> Dict[str, Any]:
        """
        Align words between src_sentence and tgt_sentence using consistent
        punctuation-separated tokenization and Bidirectional Alignment Reconciliation.
        """
        sent_src = self._tokenize(src_sentence)
        sent_tgt = self._tokenize(tgt_sentence)

        # Single clean bidirectional alignment pass with separated punctuation
        forward_align = self._run_bert_align(sent_src, sent_tgt)
        reverse_align = self._run_bert_align(sent_tgt, sent_src)
        reverse_swapped = set((s_i, t_j) for (t_j, s_i) in reverse_align)

        lexical_set = set(forward_align) | reverse_swapped
        bidirectional_lexical = set(forward_align) & reverse_swapped


        reconciled_pairs = []
        for s_i, t_j in sorted(lexical_set):
            if s_i < 0 or t_j < 0 or s_i >= len(sent_src) or t_j >= len(sent_tgt):
                continue



            src_w = sent_src[s_i]
            tgt_w = sent_tgt[t_j]
            c_src = strip_punctuation(src_w).strip()
            c_tgt = strip_punctuation(tgt_w).strip()

            is_pure_punct = (not c_src) and (not c_tgt)
            is_bidirectional = (s_i, t_j) in bidirectional_lexical

            if is_pure_punct:
                confidence = "STRUCTURAL"
            elif is_bidirectional:
                confidence = "HIGH_BIDIRECTIONAL"
            else:
                confidence = "HIGH"

            reconciled_pairs.append({
                "src_index": s_i,
                "src_word": src_w,
                "tgt_index": t_j,
                "tgt_word": tgt_w,
                "is_lexical": not is_pure_punct,
                "is_bidirectional": is_bidirectional,
                "is_punctuation_only": is_pure_punct,
                "confidence": confidence
            })

        # Prune unidirectional noise when a high-confidence bidirectional alignment exists for source token s_i
        src_has_bidir = set(item["src_index"] for item in reconciled_pairs if item["is_bidirectional"] and not item["is_punctuation_only"])
        if src_has_bidir:
            pruned_pairs = []
            for item in reconciled_pairs:
                s_idx = item["src_index"]
                if s_idx in src_has_bidir and not item["is_bidirectional"] and not item["is_punctuation_only"]:
                    continue
                pruned_pairs.append(item)
            reconciled_pairs = pruned_pairs

        # Inject validated entity anchors
        for s_i, t_j in entity_anchors.items():
            if not any(item["src_index"] == s_i and item["tgt_index"] == t_j for item in reconciled_pairs):
                reconciled_pairs.append({
                    "src_index": s_i,
                    "src_word": sent_src[s_i],
                    "tgt_index": t_j,
                    "tgt_word": sent_tgt[t_j],
                    "is_lexical": True,
                    "is_bidirectional": True,
                    "is_punctuation_only": False,
                    "confidence": "ENTITY_ANCHOR"
                })

        reconciled_pairs.sort(key=lambda x: (x["src_index"], x["tgt_index"]))

        src_to_tgt = reconciled_pairs
        tgt_to_src = [
            {
                "tgt_index": item["tgt_index"],
                "tgt_word": item["tgt_word"],
                "src_index": item["src_index"],
                "src_word": item["src_word"],
                "is_lexical": item["is_lexical"],
                "is_punctuation_only": item["is_punctuation_only"],
                "confidence": item["confidence"],
            }
            for item in reconciled_pairs
        ]
        aligned_pairs = [(item["src_word"], item["tgt_word"]) for item in reconciled_pairs]

        return {
            "src_sentence": src_sentence,
            "tgt_sentence": tgt_sentence,
            "src_tokens": sent_src,
            "tgt_tokens": sent_tgt,
            "aligned_pairs": aligned_pairs,
            "src_to_tgt_alignments": src_to_tgt,
            "tgt_to_src_alignments": tgt_to_src,
        }

    def _heuristic_align(self, sent_src: List[str], sent_tgt: List[str]) -> List[Tuple[int, int]]:
        pairs = []
        used_tgt = set()
        
        for i, s_w in enumerate(sent_src):
            c_s = strip_punctuation(s_w).lower()
            matched = False
            
            # 1. Exact string / cognate match
            for j, t_w in enumerate(sent_tgt):
                c_t = strip_punctuation(t_w).lower()
                if j not in used_tgt and (c_s == c_t or (len(c_s) >= 4 and c_s in c_t)):
                    pairs.append((i, j))
                    used_tgt.add(j)
                    matched = True
                    break

            # 2. Positional ratio fallback
            if not matched and len(sent_tgt) > 0:
                pos_j = min(int(round((i / max(1, len(sent_src) - 1)) * (len(sent_tgt) - 1))), len(sent_tgt) - 1)
                if pos_j not in used_tgt:
                    pairs.append((i, pos_j))
                    used_tgt.add(pos_j)

        return pairs


from cleft.cleft_pipeline import CleftPipeline
from cleft.bhashik_focus_reorderer import BhashikFocusReorderer


class AwesomeCleftPipeline:
    """
    Complete Cross-Lingual Focus Transfer Pipeline:

    English sentence + Malayalam sentence + emphasis marker
                        ↓
                  Awesome-Aligner
                        ↓
            English → Malayalam alignment (Focus Projection)
                        ↓
             Clefting pipeline (ssf_pipeline/cleft_pipeline.py)
               * Phase 1: Eligibility check
               * Phase 2: Copula attachment (അച്ഛൻ -> അച്ഛനാണ്)
               * Phase 3: Verb normalization (വാങ്ങി -> വാങ്ങിയത്)
               * Phase 4: Clausal assembly
                        ↓
           Emphasized Malayalam sentence
                        ↓
                  Awesome-Aligner
                 ↙              ↘
    Malayalam → English     English → Malayalam
    alignment               alignment
    """

    def __init__(
        self,
        cleft_engine: str = "ssf",  # "ssf" (CleftPipeline) or "neural" (BhashikFocusReorderer)
        model_dir: Optional[str] = None,
        aligner_type: str = "awesome",  # "awesome" (default) or "simalign"
        aligner_model: Optional[str] = None,
        aligner_method: str = "itermax",  # for simalign: "itermax", "argmax", "match"
        device: Optional[str] = None,
    ):
        self.cleft_engine = cleft_engine.lower()
        self.aligner_type = (aligner_type or "awesome").lower()
        default_model = "bert-base-multilingual-cased"

        if self.aligner_type == "simalign":
            self.aligner = SimAlignerWrapper(
                model_name=aligner_model if aligner_model else default_model,
                matching_method=aligner_method,
                device=device,
            )
        else:
            self.aligner = AwesomeAlignerWrapper(
                model_name=aligner_model if aligner_model else default_model,
                device=device,
            )
        
        self.ssf_pipeline = CleftPipeline()
        self.neural_reorderer = None
        self.translator = None
        self.bhashaverse_translator = None
        self.krutrim_translator = None
        self.model_dir = model_dir
        self.device = device

    def _is_verbal_token(self, word: str) -> bool:
        """
        Check if a Malayalam token is a finite main verb or verbal predicate.
        """
        clean_w = strip_punctuation(word).strip()
        if not clean_w:
            return False

        # 1. mlmorph morphological analysis
        try:
            analyses = self.ssf_pipeline._analysis_layer.analyser.analyse(clean_w)
            for raw, _ in analyses:
                if any(tag in raw for tag in ("<v>", "<verb>", "<present>", "<past>", "<future>", "<cvb", "<imperative-mood>", "<permissive-mood>", "<conditional-mood>")):
                    if not (clean_w.endswith("ത്") and "<n><deriv>" in raw):
                        return True
        except Exception:
            pass

        # 2. Surface finite & nominalized verb suffix heuristics
        finite_verb_suffixes = (
            "ുന്നു", "ിച്ചു", "ച്ചു", "ഞ്ഞു", "ന്നു",
            "ാറുണ്ട്", "ാറില്ല", "ചെയ്യുന്നു", "ഉണ്ട്", "ആണ്", "ആയിരുന്നു",
            "ുന്നത്", "ന്നത്", "ത്തത്", "തത്", "യത്", "ിയത്", " ചെയ്തത്"
        )
        if any(clean_w.endswith(sfx) for sfx in finite_verb_suffixes):
            if clean_w not in ("ഇന്ത്യൻ", "വടക്കൻ", "തെക്കൻ", "പലതും", "എല്ലാം", "മറ്റും"):
                return True

        return False

    # ------------------------------------------------------------------
    # NP-modifier and clause-boundary helpers for focus projection
    # ------------------------------------------------------------------

    def _is_np_modifier(self, word: str) -> bool:
        """
        Check if a Malayalam token is an NP-internal modifier that naturally
        precedes a head noun: relative participles used as adjectives, genitive
        nouns, locative adjective forms, degree adverbs, numerals, etc.
        """
        clean = strip_punctuation(word).strip()
        if not clean:
            return False

        # 1. Relative participle / participial adjective endings
        rp_suffixes = (
            "ിട്ടുള്ള", "ഉള്ള", "ുന്ന", "ാത്ത", "പ്പെട്ട",
            "ായ", "ിയ", "ച്ച", "ത്ത", "ന്ന",
        )
        if any(clean.endswith(sfx) for sfx in rp_suffixes):
            return True

        # 2. Genitive case endings
        if any(clean.endswith(sfx) for sfx in ("യുടെ", "ുടെ", "ന്റെ", "ിന്റെ")):
            return True

        # 3. Adjectival locative (-ിലെ / -ലെ / -ത്തെ)
        if clean.endswith("ിലെ") or clean.endswith("ലെ") or clean.endswith("ത്തെ"):
            return True

        # 4. Adverbial modifiers (manner/quality — modify following adjective)
        if clean.endswith("മായി") or clean.endswith("ായി"):
            return True

        # 5. Postpositions and range/prepositional modifiers
        if clean in {
            "മുതൽ", "മുതല്‍", "വരെ", "വഴി", "കൊണ്ട്", "വച്ച്", "വെച്ച്",
            "ഉൾപ്പെടെ", "ഉള്‍പ്പെടെ", "അടക്കം", "പ്രകാരം", "കൂടാതെ",
        }:
            return True

        # 6. Degree adverbs
        if clean in {
            "ഏറ്റവും", "ഏറെ", "അത്യന്തം", "വളരെ", "കൂടുതൽ",
            "ഏറ്റം", "ഏറ്റവുമധികം", "ഏതാണ്ട്",
        }:
            return True

        # 7. Articles & Cardinal numerals (including transliterated titles/numbers)
        if clean in {
            "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്",
            "ഏഴ്", "എട്ട്", "ഒൻപത്", "പത്ത്", "ഇരുപത്", "മുപ്പത്",
            "നൂറ്", "ആയിരം", "വൺ", "ടു", "ത്രീ", "ഫോർ", "ഫൈവ്",
            "സിക്സ്", "സെവൻ", "എയിറ്റ്", "നൈൻ", "ടെൻ", "ഇലവൻ", "ട്വൽവ്",
            "ദി", "ദ", "ദ്", "ഒരു", "എ",
        } or clean.isdigit():
            return True

        # 7. mlmorph adjective / quantifier POS fallback
        try:
            analyses = self.ssf_pipeline._analysis_layer.analyser.analyse(clean)
            has_noun = any("<n>" in raw or "<np>" in raw or raw.endswith("<eng>") for raw, _ in analyses)
            for raw, _ in analyses:
                if ("<adj>" in raw and not has_noun) or "<quantifier>" in raw:
                    return True
        except Exception:
            pass

        return False

    def _has_clausal_verb(self, tgt_tokens: List[str], indices: List[int]) -> bool:
        """
        Check whether any token at *indices* is a finite verb or infinitive
        that would head its own clause — NOT a simple NP-internal modifier.
        Used to block cluster merging across relative-clause boundaries.
        """
        finite_suffixes = (
            "ുന്നു", "ിച്ചു", "ച്ചു", "ഞ്ഞു", "ന്നു",
            "ാറുണ്ട്", "ാറില്ല", "ഉണ്ട്", "ആണ്", "ആയിരുന്നു",
        )
        infinitive_suffixes = ("ാൻ", "ുവാൻ", "ാനായി", "ാനുള്ള")

        for idx in indices:
            if idx < 0 or idx >= len(tgt_tokens):
                continue
            clean = strip_punctuation(tgt_tokens[idx]).strip()
            if not clean:
                continue
            if any(clean.endswith(sfx) for sfx in finite_suffixes):
                return True
            if any(clean.endswith(sfx) for sfx in infinitive_suffixes):
                return True
        return False

    def _find_matrix_verb_idx(self, tgt_tokens: List[str]) -> Optional[int]:
        """Find the index of the sentence-final matrix verb (SOV order)."""
        for i in range(len(tgt_tokens) - 1, max(-1, len(tgt_tokens) - 3), -1):
            if self._is_verbal_token(tgt_tokens[i]):
                return i
        return None

    def _find_matrix_verb_for_clause(self, tgt_tokens: List[str], comp_verb_idx: Optional[int] = None) -> Optional[int]:
        """
        Find the finite matrix verb that introduces a complement clause or heads the sentence,
        skipping subordinate conditional verbs (e.g. -യാണെങ്കിൽ) and complementizer verbs (-എന്ന്).
        """
        search_start = (comp_verb_idx - 1) if comp_verb_idx is not None else (len(tgt_tokens) - 1)
        for i in range(search_start, -1, -1):
            tok = strip_punctuation(tgt_tokens[i]).strip()
            if not tok:
                continue
            if any(tok.endswith(sfx) for sfx in ("എന്ന്", "എന്നു്", "എന്നു", "ഉണ്ടെന്ന്", "ആണെന്ന്", "എങ്കിൽ", "യാണെങ്കിൽ", "കയാണെങ്കിൽ", "ച്ചാൽ", "ന്നാൽ")):
                continue
            if self._is_verbal_token(tok):
                return i
        return None

    def _expand_parenthesis_bounds(self, tgt_tokens: List[str], start: int, end: int) -> Tuple[int, int]:
        """
        Expands start and end token indices so that parenthetical expressions like '(ഡി. എൻ. ബി)'
        or '[...] ' are never split across focus tag boundaries.
        """
        span_text = " ".join(tgt_tokens[start:end + 1])
        open_parens = span_text.count("(") - span_text.count(")")
        open_brackets = span_text.count("[") - span_text.count("]")

        # Net closing parens: expand leftward to find matching '('
        while start > 0 and open_parens < 0:
            start -= 1
            if "(" in tgt_tokens[start]:
                open_parens += tgt_tokens[start].count("(")
            if ")" in tgt_tokens[start]:
                open_parens -= tgt_tokens[start].count(")")

        # Net opening parens: expand rightward to find matching ')'
        while end < len(tgt_tokens) - 1 and open_parens > 0:
            end += 1
            if ")" in tgt_tokens[end]:
                open_parens -= tgt_tokens[end].count(")")
            if "(" in tgt_tokens[end]:
                open_parens += tgt_tokens[end].count("(")

        while start > 0 and open_brackets < 0:
            start -= 1
            if "[" in tgt_tokens[start]:
                open_brackets += tgt_tokens[start].count("[")
            if "]" in tgt_tokens[start]:
                open_brackets -= tgt_tokens[start].count("]")

        while end < len(tgt_tokens) - 1 and open_brackets > 0:
            end += 1
            if "]" in tgt_tokens[end]:
                open_brackets -= tgt_tokens[end].count("]")
            if "[" in tgt_tokens[end]:
                open_brackets += tgt_tokens[end].count("[")

        return start, end

    # ------------------------------------------------------------------
    # Core focus projection: head-noun-anchored NP expansion
    # ------------------------------------------------------------------

    def _select_focus_constituent(
        self,
        clean_en_sent: str,
        clean_ml_sent: str,
        english_focus: str,
        pre_alignment: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]], str, List[int], List[Dict[str, Any]], Dict[str, Any]]:
        src_tokens = pre_alignment.get("source_tokens", pre_alignment.get("src_tokens", clean_en_sent.split()))
        tgt_tokens = pre_alignment.get("target_tokens", pre_alignment.get("tgt_tokens", clean_ml_sent.split()))
        reconciled = pre_alignment.get("reconciled_pairs", [])

        en_focus_words = [
            strip_punctuation(w).lower()
            for w in english_focus.strip().split()
            if strip_punctuation(w)
        ]
        if not en_focus_words or not tgt_tokens:
            fallback = strip_punctuation(tgt_tokens[0]) if tgt_tokens else ""
            return fallback, [], fallback, [0] if tgt_tokens else [], [], {"original_focus": english_focus}

        flen = len(en_focus_words)
        src_clean = [strip_punctuation(w).lower() for w in src_tokens]
        focus_src_indices = []
        for i in range(len(src_clean) - flen + 1):
            if src_clean[i : i + flen] == en_focus_words:
                focus_src_indices = list(range(i, i + flen))
                break

        if not focus_src_indices:
            for i, sw in enumerate(src_clean):
                if any((efw in sw or sw in efw) for efw in en_focus_words if efw):
                    focus_src_indices.append(i)

        target_indices = [
            p["tgt_index"]
            for p in reconciled
            if p.get("src_index") in focus_src_indices and p.get("is_lexical", True)
        ]

        if target_indices:
            min_t, max_t = min(target_indices), max(target_indices)
            # Jayaseelan P-stranding ban: if next token is postposition, include it in PP focus
            if max_t + 1 < len(tgt_tokens):
                from cleft.cleft_pipeline import POSTPOSITIONS
                next_w = strip_punctuation(tgt_tokens[max_t + 1]).strip()
                if next_w in POSTPOSITIONS:
                    max_t += 1
            span_indices = list(range(min_t, max_t + 1))
            selected_text = " ".join(tgt_tokens[i] for i in span_indices if strip_punctuation(tgt_tokens[i]).strip())
        else:
            avg_src = sum(focus_src_indices) / len(focus_src_indices) if focus_src_indices else 0
            approx_t = int(round(avg_src * (len(tgt_tokens) - 1) / max(len(src_tokens) - 1, 1)))
            approx_t = max(0, min(len(tgt_tokens) - 1, approx_t))
            span_indices = [approx_t]
            selected_text = strip_punctuation(tgt_tokens[approx_t]).strip()

        metadata = {
            "original_focus": english_focus,
            "selected_constituent": selected_text,
            "focus_span_indices": span_indices,
        }
        return selected_text, reconciled, selected_text, span_indices, [], metadata

    def _insert_focus_tags(
        self,
        clean_ml_sent: str,
        projected_focus: str,
        focus_span_indices: List[int],
        pre_alignment: Dict[str, Any],
    ) -> str:
        tgt_tokens = pre_alignment.get("target_tokens", pre_alignment.get("tgt_tokens", []))
        if focus_span_indices and tgt_tokens:
            min_idx = min(focus_span_indices)
            max_idx = max(focus_span_indices)
            tagged_tokens = []
            for i, tok in enumerate(tgt_tokens):
                if i == min_idx and i == max_idx:
                    tagged_tokens.append(f"<FF>{tok}</FF>")
                elif i == min_idx:
                    tagged_tokens.append(f"<FF>{tok}")
                elif i == max_idx:
                    tagged_tokens.append(f"{tok}</FF>")
                else:
                    tagged_tokens.append(tok)

            reconstructed = " ".join(tagged_tokens)
            reconstructed = re.sub(r"<FF>\s+", "<FF>", re.sub(r"\s+</FF>", "</FF>", reconstructed))
            reconstructed = re.sub(r"\s+([.,!?;:])", r"\1", reconstructed)
            return reconstructed

        clean_focus = strip_punctuation(projected_focus).strip()
        if clean_focus and clean_focus in clean_ml_sent:
            pat = re.compile(rf"(?<![\w\u0D00-\u0D7F]){re.escape(clean_focus)}(?![\w\u0D00-\u0D7F])")
            if pat.search(clean_ml_sent):
                return pat.sub(f"<FF>{clean_focus}</FF>", clean_ml_sent, count=1)
            return clean_ml_sent.replace(clean_focus, f"<FF>{clean_focus}</FF>", 1)

        return clean_ml_sent

    def process(
        self,
        english_sentence: str,
        malayalam_sentence: str = "",
        english_focus: str = "",
        malayalam_focus: str = "",
        operation: str = "CLEFT",
        focus_type: str = "INFORMATION",
        mt_model: str = "k",
    ) -> Dict[str, Any]:
        """
        Execute full cross-lingual emphasis transfer pipeline.

        Inputs:
          - english_sentence: e.g. "Father bought a book in the garden yesterday." (or with <FF>Father<FF>)
          - malayalam_sentence: e.g. "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി." (or with <FF>അച്ഛൻ<FF>)
          - english_focus: Optional focused English word/phrase (e.g. "Father")
          - malayalam_focus: Optional focused Malayalam constituent (e.g. "അച്ഛൻ")
          - mt_model: 'k' or 'krutrim' (default) vs 'b' or 'bhashaverse'
        """
        # Step 0: Extract focus markers if contained in input strings
        clean_en_sent = english_sentence
        if "<FF>" in english_sentence:
            ff_matches = re.findall(r"<FF>(.*?)(?:</FF>|<FF>)", english_sentence)
            if ff_matches:
                english_focus = ff_matches[0].strip()
            clean_en_sent = re.sub(r"</?FF>", "", english_sentence).strip()

        clean_ml_sent = malayalam_sentence
        if "<FF>" in malayalam_sentence:
            ff_ml_matches = re.findall(r"<FF>(.*?)(?:</FF>|<FF>)", malayalam_sentence)
            if ff_ml_matches:
                malayalam_focus = ff_ml_matches[0].strip()
            clean_ml_sent = re.sub(r"</?FF>", "", malayalam_sentence).strip()

        # Step 0b: Automated Neural Machine Translation (Krutrim / Bhashaverse) if Malayalam input is omitted
        if not clean_ml_sent:
            chosen_mt = (mt_model or "k").strip().lower()
            if chosen_mt.startswith("b"):
                if self.bhashaverse_translator is None:
                    self.bhashaverse_translator = BhashaverseTranslator(device=str(self.device) if self.device else None)
                clean_ml_sent = self.bhashaverse_translator.translate(clean_en_sent)
            else:
                if self.krutrim_translator is None:
                    self.krutrim_translator = KrutrimTranslator(device=str(self.device) if self.device else None)
                clean_ml_sent = self.krutrim_translator.translate(clean_en_sent)

        # Step 1: Pre-Cleft Alignment (English -> Malayalam)
        pre_alignment = self.aligner.align(clean_en_sent, clean_ml_sent)

        # Step 1b: Focus Projection & Semantic Constituent Resolution
        projected_ml_focus = malayalam_focus
        candidate_alignments = []
        reconstructed_span = ""
        focus_span_indices: List[int] = []
        focus_candidates: List[Dict[str, Any]] = []
        projection_metadata: Dict[str, Any] = {}

        if english_focus and not projected_ml_focus:
            (
                projected_ml_focus,
                candidate_alignments,
                reconstructed_span,
                focus_span_indices,
                focus_candidates,
                projection_metadata,
            ) = self._select_focus_constituent(
                clean_en_sent=clean_en_sent,
                clean_ml_sent=clean_ml_sent,
                english_focus=english_focus,
                pre_alignment=pre_alignment,
            )
        elif not projection_metadata:
            projection_metadata = {
                "original_focus": english_focus or projected_ml_focus or "",
                "selected_constituent": english_focus or projected_ml_focus or "",
                "selected_constituent_ml": projected_ml_focus or "",
                "focus_type": "WHOLE",
                "projection": "NO",
            }

        if not projected_ml_focus:
            # Fallback to first word of Malayalam sentence
            ml_words = clean_ml_sent.strip().split()
            projected_ml_focus = ml_words[0] if ml_words else ""
            reconstructed_span = f"[0] -> '{projected_ml_focus}'"
            focus_span_indices = [0] if ml_words else []
            focus_candidates = [{
                "rank": 1,
                "is_selected": True,
                "span_indices": focus_span_indices,
                "span_range": "[0..0]" if ml_words else "[0]",
                "span_text": projected_ml_focus,
                "candidate_type": "Fallback (First Word)",
                "evidence_count": 0,
                "evidence": [],
            }]

        clean_proj_focus = strip_punctuation(projected_ml_focus)


        # Step 2: SSF / Neural Clefting Pipeline Execution
        if self.cleft_engine == "neural":
            if self.neural_reorderer is None:
                self.neural_reorderer = BhashikFocusReorderer(model_dir=self.model_dir, device=self.device)
            cleft_dict = self.neural_reorderer.reorder(
                sentence=clean_ml_sent,
                focused_constituent=clean_proj_focus,
                operation=operation,
                focus_type=focus_type,
            )
            emphasized_ml_sentence = cleft_dict["reordered_sentence"]
        else:
            # SSF Rule-Based CleftPipeline (Phase 1-4)
            # Insert <FF>...<FF> around target focus using token indices
            tagged_ml_sent = self._insert_focus_tags(
                clean_ml_sent,
                clean_proj_focus,
                focus_span_indices,
                pre_alignment=pre_alignment,
            )

            cleft_res = self.ssf_pipeline.process(tagged_ml_sent)
            cleft_dict = cleft_res.to_dict()
            emphasized_ml_sentence = cleft_res.cleft_sentence if cleft_res.status == "VALID" else (cleft_res.cleft_sentence or clean_ml_sent)
            emphasized_ml_sentence = re.sub(r"</?PE>", "", emphasized_ml_sentence).strip()

        # Step 3: Post-Cleft Dual Alignment (Awesome-Aligner)
        # 3a. Malayalam -> English alignment
        post_align_ml_to_en = self.aligner.align(emphasized_ml_sentence, clean_en_sent)

        # 3b. English -> Malayalam alignment
        post_align_en_to_ml = self.aligner.align(clean_en_sent, emphasized_ml_sentence)

        # Compute English (Source) Prosody Label Sequence as a contiguous span
        en_tokens = clean_en_sent.split()
        clean_en_words = [strip_punctuation(w).lower() for w in en_tokens]
        en_focus_words = [strip_punctuation(w).lower() for w in (english_focus or "").split() if strip_punctuation(w)]
        
        en_prosody_labels = [0] * len(en_tokens)
        ef_len = len(en_focus_words)
        
        if ef_len > 0:
            en_matched = False
            # 1. Exact contiguous slice match
            for i in range(len(clean_en_words) - ef_len + 1):
                if clean_en_words[i : i + ef_len] == en_focus_words:
                    for k in range(i, i + ef_len):
                        en_prosody_labels[k] = 1
                    en_matched = True
                    break
            
            # 2. If minor tokenization differences, find best matching contiguous window
            if not en_matched:
                best_score = 0
                best_window = None
                for i in range(len(clean_en_words)):
                    for j in range(i + 1, min(len(clean_en_words) + 1, i + ef_len + 4)):
                        window = clean_en_words[i:j]
                        score = sum(1 for w in window if w in en_focus_words)
                        length_penalty = abs(len(window) - ef_len) * 0.1
                        adj_score = score - length_penalty
                        if adj_score > best_score:
                            best_score = adj_score
                            best_window = (i, j)
                if best_window and best_score > 0:
                    for k in range(best_window[0], best_window[1]):
                        en_prosody_labels[k] = 1

        # Compute / verify Target (Malayalam) Prosody Label Sequence for emphasized_ml_sentence
        ml_tokens = emphasized_ml_sentence.split()
        ml_prosody_labels = cleft_dict.get("target_prosody_label_sequence") or cleft_dict.get("prosody_label_sequence", [])
        if len(ml_prosody_labels) != len(ml_tokens):
            clean_ml_words = [strip_punctuation(t) for t in ml_tokens]
            clean_fc_words = [strip_punctuation(w) for w in (clean_proj_focus or "").split() if strip_punctuation(w)]
            fc_len = len(clean_fc_words)
            ml_pos = []
            if fc_len > 0:
                for j in range(len(clean_ml_words) - fc_len + 1):
                    if all(is_word_match(clean_ml_words[j + k], clean_fc_words[k]) for k in range(fc_len)):
                        ml_pos = list(range(j, j + fc_len))
                        break
                if not ml_pos:
                    for j, tw in enumerate(clean_ml_words):
                        if any(is_word_match(tw, fcw) for fcw in clean_fc_words):
                            ml_pos.append(j)
            ml_prosody_labels = [1 if j in ml_pos else 0 for j in range(len(ml_tokens))]

        result = {
            "english_sentence": clean_en_sent,
            "original_malayalam_sentence": clean_ml_sent,
            "english_focus": english_focus,
            "focused_malayalam_constituent": clean_proj_focus,
            "focus_candidates": focus_candidates,
            "focus_projection_metadata": projection_metadata,
            "original_focus": projection_metadata.get("original_focus", english_focus),
            "selected_constituent": projection_metadata.get("selected_constituent", clean_proj_focus),
            "focus_type": projection_metadata.get("focus_type", "WHOLE"),
            "projection": projection_metadata.get("projection", "NO"),
            "cleft_engine": self.cleft_engine,
            "english_prosody_label_sequence": en_prosody_labels,
            "malayalam_prosody_label_sequence": ml_prosody_labels,
            "pre_cleft_en_to_ml_alignment": pre_alignment,
            "cleft_pipeline_output": cleft_dict,
            "emphasized_malayalam_sentence": emphasized_ml_sentence,
            "post_cleft_ml_to_en_alignment": post_align_ml_to_en,
            "post_cleft_en_to_ml_alignment": post_align_en_to_ml,
        }

        result["cleft_reorderings"] = {}

        # Step 6: Parallel Constituency Reordering Branch
        # Operates on the ORIGINAL (non-clefted) sentence.
        # Only three rules are empirically attested; all others return NOT_APPLICABLE.
        constituency_reorderer = ConstituencyReorderer()
        effective_role = "" if (focus_type or "").upper() == "INFORMATION" else focus_type
        result["constituency_reordering"] = constituency_reorderer.reorder(
            original_sentence=clean_ml_sent,
            focused_constituent=clean_proj_focus,
            focus_role=effective_role,  # empty string enables morphological/heuristic detection
        )

        return result


def generate_pipeline_report(res: Dict[str, Any]) -> str:
    """Generate a complete, structured text report of all pipeline stages and diagnostics."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"1. English Sentence            : {res['english_sentence']}")
    lines.append(f"2. Original Malayalam Sentence : {res['original_malayalam_sentence']}")
    lines.append(f"3. English Focus Marker        : {res['english_focus'] or '—'}")
    lines.append(f"4. Selected Malayalam Focus    : {res['focused_malayalam_constituent']}")

    proj_meta = res.get("focus_projection_metadata", {})
    if proj_meta:
        lines.append(f"   * original_focus            : {proj_meta.get('original_focus', '—')}")
        lines.append(f"   * selected_constituent      : {proj_meta.get('selected_constituent', '—')}")
        lines.append(f"   * focus_type                : {proj_meta.get('focus_type', '—')}")
        lines.append(f"   * projection                : {proj_meta.get('projection', '—')}")

    # --- Focus Projection Candidates ---
    candidates = res.get("focus_candidates", [])
    lines.append("\n" + "-" * 80)
    lines.append("--- Malayalam Focus Candidates (Focus Projection) ---")
    lines.append("-" * 80)
    if proj_meta:
        lines.append(f"  Focus Projection Summary:")
        lines.append(f"    * original_focus       : {proj_meta.get('original_focus', '—')}")
        lines.append(f"    * selected_constituent : {proj_meta.get('selected_constituent', '—')}")
        lines.append(f"    * focus_type           : {proj_meta.get('focus_type', '—')}")
        lines.append(f"    * projection           : {proj_meta.get('projection', '—')}\n")
    if candidates:
        for c in candidates:
            status_tag = "[SELECTED]" if c.get("is_selected") else "[ALTERNATIVE]"
            lines.append(f"  {status_tag:<14} Rank {c.get('rank', '-')}: {c.get('span_range')} -> '{c.get('span_text')}'")
            lines.append(f"    * Type            : {c.get('candidate_type', 'Candidate')}")
            lines.append(f"    * Evidence Count  : {c.get('evidence_count', 0)} aligned token pair(s)")
            ev_list = c.get("evidence", [])
            if ev_list:
                lines.append(f"    * Alignment Evidence:")
                for ev in ev_list:
                    lines.append(f"        - [{ev['src_index']}] '{ev['src_word']}' <---> [{ev['tgt_index']}] '{ev['tgt_word']}'")
            lines.append("")
    else:
        lines.append(f"  * Selected: '{res['focused_malayalam_constituent']}'\n")

    # --- Pre-cleft English -> Malayalam Alignment ---
    lines.append("-" * 80)
    lines.append("--- Pre-Cleft Word Alignments (English -> Malayalam) ---")
    lines.append("-" * 80)
    for pair in res.get("pre_cleft_en_to_ml_alignment", {}).get("aligned_pairs", []):
        lines.append(f"  {pair[0]:<25} <---> {pair[1]:<25}")

    # --- Clefting Pipeline Output ---
    cleft = res.get("cleft_pipeline_output", {})
    lines.append("\n" + "-" * 80)
    lines.append("--- Clefting Pipeline Output & Diagnostics ---")
    lines.append("-" * 80)
    lines.append(f"  * Emphasized Malayalam Sentence             : {res.get('emphasized_malayalam_sentence', '—')}")
    lines.append(f"  * Prosody Label Sequence (Source / English)  : {res.get('english_prosody_label_sequence', [])}")
    lines.append(f"  * Prosody Label Sequence (Target / Malayalam): {res.get('malayalam_prosody_label_sequence', [])}")
    lines.append(f"  * Focus Position (Before -> After)          : {cleft.get('position_before', [])} -> {cleft.get('position_after', [])}")

    aanu = cleft.get("aanu_attachment", {})
    if aanu and aanu.get("attached_to") != "none":
        lines.append(f"  * ആണ് Attachment                : {aanu.get('attached_form')} (attached to {aanu.get('attached_to')} '{aanu.get('target_word')}')")
    lines.append(f"  * Nominalized Verb              : {cleft.get('nominalized_verb')}")
    lines.append(f"  * Strategy Route                : {cleft.get('route', 'CLEFT')}")
    lines.append(f"  * Pipeline Status               : {cleft.get('status', 'VALID')}")
    if cleft.get("error"):
        lines.append(f"  * Diagnostic / Error Note       : {cleft.get('error')}")

    # --- Post-cleft Malayalam -> English Alignment ---
    lines.append("\n" + "-" * 80)
    lines.append("--- Post-Cleft Alignment (Malayalam -> English) ---")
    lines.append("-" * 80)
    for item in res.get("post_cleft_ml_to_en_alignment", {}).get("src_to_tgt_alignments", []):
        lines.append(f"  [{item['src_index']}] {item['src_word']:<25} ---> [{item['tgt_index']}] {item['tgt_word']:<25}")

    # --- Post-cleft English -> Malayalam Alignment ---
    lines.append("\n" + "-" * 80)
    lines.append("--- Post-Cleft Alignment (English -> Malayalam) ---")
    lines.append("-" * 80)
    for item in res.get("post_cleft_en_to_ml_alignment", {}).get("src_to_tgt_alignments", []):
        lines.append(f"  [{item['src_index']}] {item['src_word']:<25} ---> [{item['tgt_index']}] {item['tgt_word']:<25}")

    # ═══════════════════════════════════════════════════════════
    # Parallel Emphasis Methods: CLEFTING vs REORDERING
    # ═══════════════════════════════════════════════════════════
    lines.append("\n" + "═" * 80)
    lines.append("EMPHASIS METHOD COMPARISON")
    lines.append("═" * 80)

    # --- Branch A: CLEFTING ---
    lines.append("\n" + "-" * 80)
    lines.append("--- BRANCH A: CLEFTING ---")
    lines.append("-" * 80)
    lines.append(f"  * Cleft Output (Emphasized)    : {res['emphasized_malayalam_sentence']}")
    reorderings = res.get("cleft_reorderings", {})
    if reorderings and reorderings.get("preverbal_focus"):
        lines.append(f"  * Preverbal  (Contrastive)     : {reorderings['preverbal_focus']}")
        lines.append(f"  * Postverbal (Informational)   : {reorderings['postverbal_focus']}")
        lines.append(f"  * Clause-Initial (Strong)      : {reorderings['clause_initial_focus']}")
        comp = reorderings.get("components", {})
        cc = comp.get("cleft_clause", {})
        if cc:
            lines.append(f"  * Cleft Components:")
            if comp.get("pre_clauses"):
                lines.append(f"      Pre-clauses      : {comp['pre_clauses']}")
            lines.append(f"      Background       : {cc.get('background', [])}")
            lines.append(f"      Focus+ആണ്         : {cc.get('focus_copula', [])}")
            lines.append(f"      Nominalized Verb : {cc.get('nominalized_verb', [])}")
            if comp.get("post_clauses"):
                lines.append(f"      Post-clauses     : {comp['post_clauses']}")
    else:
        lines.append("  * (No positional cleft reorderings produced.)")

    # --- Branch B: CONSTITUENT REORDERING ---
    lines.append("\n" + "-" * 80)
    lines.append("--- BRANCH B: CONSTITUENT REORDERING (4-Level Architecture) ---")
    lines.append("-" * 80)
    cr = res.get("constituency_reordering", {})
    if cr.get("applicable"):
        lines.append(f"  * Status                       : APPLICABLE")
        lines.append(f"  * Rule Applied                 : {cr.get('rule_applied')}")
        lines.append(f"  * Reordered Output (Full Text) : {cr.get('reordered_sentence')}")
        if cr.get("movement_vector"):
            lines.append(f"  * Movement Vector (Constituent): {cr.get('movement_vector')} (position {cr.get('position_before')} -> position {cr.get('position_after')})")
        if cr.get("isolated_sentence_before") and cr.get("isolated_sentence_after"):
            lines.append(f"  * Isolated S_focus (Before)    : {cr.get('isolated_sentence_before')}")
            lines.append(f"  * Isolated S_focus (After)     : {cr.get('isolated_sentence_after')}")
        if cr.get("constituent_sequence_before"):
            lines.append(f"  * Constituents (Before)        : {cr.get('constituent_sequence_before')}")
        if cr.get("constituent_sequence_after"):
            lines.append(f"  * Constituents (After)         : {cr.get('constituent_sequence_after')}")
        if cr.get("focus_span_tokens"):
            lines.append(f"  * Focus Span Tokens            : {cr.get('focus_span_tokens')}")
        lines.append(f"  * Reason                       : {cr.get('reason')}")
        if cr.get("diagnostic"):
            lines.append(f"  * Diagnostic                   : {cr.get('diagnostic')}")
    else:
        lines.append(f"  * Status                       : NOT APPLICABLE")
        if cr.get("isolated_sentence_before"):
            lines.append(f"  * Target Sentence Isolated     : {cr.get('isolated_sentence_before')}")
        lines.append(f"  * Reason                       : {cr.get('reason', '—')}")
        lines.append(f"  * (Only ADVERB_FRONTING, DO_FRONTING, VERB_FRONTING are attested.)")

    lines.append("\n" + "═" * 80 + "\n")
    return "\n".join(lines)


def print_pipeline_report(res: Dict[str, Any], output_path: str = "output.txt"):
    """
    Write full pipeline output and diagnostics to output_path (default: output.txt),
    while keeping the terminal output concise.
    """
    report_text = generate_pipeline_report(res)
    abs_out_path = os.path.abspath(output_path)

    with open(abs_out_path, "w", encoding="utf-8") as f:
        f.write(report_text)


    print("✓ Cross-Lingual Focus Transfer & Clefting Completed")
   


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-Lingual Focus Transfer & Dual Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--en", "-e", type=str, help="English sentence (e.g. 'Father bought a book in the garden yesterday.')")
    parser.add_argument("--ml", type=str, default="", help="Malayalam baseline sentence (e.g. 'അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി.')")
    parser.add_argument("-m", "--mt", "--model", type=str, default="", help="Malayalam sentence OR MT model ('b' for Bhashaverse, 'k' for Krutrim Translate)")
    parser.add_argument("--en_focus", "-ef", type=str, default="", help="English focus word (e.g. 'Father')")
    parser.add_argument("--ml_focus", "-mf", type=str, default="", help="Malayalam focus word (optional, derived via alignment if omitted)")
    parser.add_argument("--op", "-o", type=str, default="CLEFT", choices=["CLEFT", "FRONTING"], help="Focus operation")
    parser.add_argument("--aligner", "-a", type=str, default="awesome", choices=["awesome", "simalign"], help="Word aligner engine ('awesome' or 'simalign')")
    parser.add_argument("--aligner_model", type=str, default="", help="Model for word aligner (e.g. 'bert-base-multilingual-cased' or 'xlm-roberta-base')")
    parser.add_argument("--aligner_method", type=str, default="itermax", choices=["itermax", "argmax", "match"], help="Matching method for SimAlign ('itermax', 'argmax', 'match')")
    parser.add_argument("--output", "-out", type=str, default="output.txt", help="Output file path (default: output.txt)")
    parser.add_argument(
        "--focus_role", "-fr", type=str, default="",
        choices=["", "ADVERB", "ADVERBIAL", "TEMPORAL", "MANNER",
                 "DIRECT_OBJECT", "OBJECT", "DO", "ACCUSATIVE", "PATIENT",
                 "VERB", "FINITE_VERB", "PREDICATE", "VP", "ACTION",
                 "SUBJECT", "ADJECTIVE", "PP", "NP", "INFORMATION"],
        help=("Grammatical role of the focus constituent for reordering rule selection. "
              "E.g. 'ADVERB' enables ADVERB_FRONTING, 'DIRECT_OBJECT' enables DO_FRONTING, "
              "'VERB' enables VERB_FRONTING. Leave empty for heuristic detection."),
    )

    # Sanitize sys.argv to handle accidental '= "value"' syntax from shell
    cleaned_argv = []
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "=" and i + 1 < len(sys.argv):
            cleaned_argv.append(sys.argv[i + 1])
            i += 2
            continue
        cleaned_argv.append(arg)
        i += 1

    args, unknown = parser.parse_known_args(cleaned_argv)

    full_cmd = " ".join(sys.argv[1:])

    en_sent = args.en
    if not en_sent or "--ml" in (en_sent or ""):
        en_match = re.search(r'--(?:en|e)\s+[\'\"]?(.*?)[\'\"]?\s+--(?:ml|m|mt|model|en_focus|op|output)\b', full_cmd)
        if en_match:
            en_sent = en_match.group(1).strip("'\"")

    ml_sent = args.ml
    mt_model = args.mt

    # Support -m "b", -m "k", -m "bhashaverse", -m "krutrim"
    if not ml_sent and mt_model and mt_model.strip().lower() not in ("b", "k", "bhashaverse", "krutrim"):
        # mt was passed a Malayalam sentence
        ml_sent = mt_model
        mt_model = "k"
    elif mt_model and mt_model.strip().lower() in ("b", "k", "bhashaverse", "krutrim"):
        mt_model = mt_model.strip().lower()
    else:
        mt_model = "k"

    if not ml_sent:
        ml_match = re.search(r'--ml\s+[\'\"]?(.*?)[\'\"]?\s+--(?:en_focus|ef|op|output|mt|model)\b', full_cmd)
        if ml_match:
            ml_sent = ml_match.group(1).strip("'\"")

    en_focus = (args.en_focus or "").strip().lstrip("=").strip().strip("'\"").strip()
    if not en_focus:
        ef_match = re.search(r'--(?:en_focus|ef)\s+[\'\"]?(.*?)[\'\"]?(?:\s+--(?:op|output|mt|model)\b|$)', full_cmd)
        if ef_match:
            en_focus = ef_match.group(1).strip("'\"")

    if not en_focus and unknown:
        en_focus = " ".join(unknown).strip().lstrip("=").strip().strip("'\"").strip()

    en_sent = en_sent or "Father bought a book in the garden yesterday."
    ml_sent = ml_sent or ""
    en_focus = en_focus or "Father"

    pipeline = AwesomeCleftPipeline(
        aligner_type=args.aligner,
        aligner_model=args.aligner_model or None,
        aligner_method=args.aligner_method,
    )
    res = pipeline.process(
        english_sentence=en_sent,
        malayalam_sentence=ml_sent,
        english_focus=en_focus,
        malayalam_focus=args.ml_focus,
        operation=args.op,
        focus_type=args.focus_role or "INFORMATION",
        mt_model=mt_model,
    )

    print_pipeline_report(res, output_path=args.output)


if __name__ == "__main__":
    main()
