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
VENV_SITE_PACKAGES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Lib", "site-packages")
if os.path.exists(VENV_SITE_PACKAGES) and VENV_SITE_PACKAGES not in sys.path:
    sys.path.append(VENV_SITE_PACKAGES)

AWESOME_ALIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "awesome-align")
if AWESOME_ALIGN_DIR not in sys.path:
    sys.path.append(AWESOME_ALIGN_DIR)

import torch
from ssf_pipeline.alignment_utils import extract_alignment_and_prosody, strip_punctuation
from bhashik_focus_reorderer import BhashikFocusReorderer
from cleft_reorderer import CleftReorderer
from ssf_pipeline.bhashaverse_translator import BhashaverseTranslator
from ssf_pipeline.krutrim_translator import KrutrimTranslator
from ssf_pipeline.tokenizer import tokenize_malayalam


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

    ENTITY_DICT = {
        "india": ["ഇന്ത്യ", "ഇന്ത്യയുടെ", "ഇന്ത്യൻ", "ഇന്ത്യയിൽ"],
        "indias": ["ഇന്ത്യ", "ഇന്ത്യയുടെ", "ഇന്ത്യൻ", "ഇന്ത്യയിൽ"],
        "indian": ["ഇന്ത്യൻ", "ഇന്ത്യക്കാർ", "ഇന്ത്യക്കാരുമാണ്", "ഇന്ത്യക്കാരും"],
        "indians": ["ഇന്ത്യൻ", "ഇന്ത്യക്കാർ", "ഇന്ത്യക്കാരുമാണ്", "ഇന്ത്യക്കാരും"],
        "afghanistan": ["അഫ്ഘാനിസ്ഥാൻ", "അഫ്ഗാനിസ്ഥാൻ", "അഫ്ഘാനിസ്ഥാനിലെ", "അഫ്ഗാനിസ്ഥാൻ്റെ", "അഫ്ഘാനിസ്ഥാൻ്റെ"],
        "afghanistans": ["അഫ്ഘാനിസ്ഥാൻ", "അഫ്ഗാനിസ്ഥാൻ", "അഫ്ഘാനിസ്ഥാനിലെ", "അഫ്ഗാനിസ്ഥാൻ്റെ"],
        "britain": ["ബ്രിട്ടൻ", "ബ്രിട്ടീഷ്"],
        "british": ["ബ്രിട്ടീഷ്", "ബ്രിട്ടീഷുകാർ", "ബ്രിട്ടീഷുകാരും"],
        "france": ["ഫ്രാൻസ്", "ഫ്രാൻസിലേക്കും", "ഫ്രാൻസിൽ"],
        "spain": ["സ്പെയിൻ", "സ്പെയിനിലേക്കും"],
        "italy": ["ഇറ്റലി", "ഇറ്റലിയിലേക്കും"],
        "jamaica": ["ജമൈക്ക", "ജമൈക്കയുടെ"],
        "dehradun": ["ഡെറാഡൂൺ"],
        "asean": ["ആസിയാൻ", "ആസിയാനിലെ"],
        "commonwealth": ["കോമൺവെൽത്ത്"],
        "karate": ["കരാട്ടെ"],
        "games": ["ഗെയിംസ്"],
    }

    def _tokenize(self, text: str) -> List[str]:
        """Universal word + punctuation tokenizer for English and Malayalam."""
        tokens = re.findall(r"[\w'\u0D00-\u0D7F]+|[.,!?;]", text)
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

        # Pass 2: Named Entity / Proper Noun anchoring
        entity_anchors = {}
        for i, s_w in enumerate(sent_src):
            c_s = strip_punctuation(s_w).lower()
            if c_s in self.ENTITY_DICT:
                target_forms = self.ENTITY_DICT[c_s]
                for j, t_w in enumerate(sent_tgt):
                    c_t = strip_punctuation(t_w)
                    if any(t_stem in c_t for t_stem in target_forms):
                        entity_anchors[i] = j
                        break

        anchored_src = set(entity_anchors.keys())
        anchored_tgt = set(entity_anchors.values())

        reconciled_pairs = []
        for s_i, t_j in sorted(lexical_set):
            if s_i < 0 or t_j < 0 or s_i >= len(sent_src) or t_j >= len(sent_tgt):
                continue

            # Skip spurious cross-alignments to anchored entity positions
            if s_i in anchored_src and t_j != entity_anchors[s_i]:
                continue
            if t_j in anchored_tgt and s_i not in anchored_src:
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


from ssf_pipeline.cleft_pipeline import CleftPipeline
from bhashik_focus_reorderer import BhashikFocusReorderer


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
        aligner_model: str = "bert-base-multilingual-cased",
        device: Optional[str] = None,
    ):
        self.cleft_engine = cleft_engine.lower()
        self.aligner = AwesomeAlignerWrapper(model_name=aligner_model, device=device)
        
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
    ) -> Tuple[str, List[Dict[str, Any]], str, List[int]]:
        """
        Project English focus onto Malayalam via alignment, then resolve a
        coherent Malayalam NP / predicate-complement constituent using
        head-noun-anchored expansion with NP-modifier cluster merging.

        Returns:
            selected_focus       – focus constituent text (punctuation stripped)
            candidate_alignments – raw alignment debug info
            reconstructed_span   – debug span string
            focus_span_indices   – token indices into tgt_tokens for tag insertion
        """
        src_tokens = pre_alignment.get("src_tokens", clean_en_sent.split())
        tgt_tokens = pre_alignment.get("tgt_tokens", clean_ml_sent.split())
        src_to_tgt = pre_alignment.get("src_to_tgt_alignments", [])
        tgt_to_src = pre_alignment.get("tgt_to_src_alignments", [])

        # 1. Normalize English focus words
        en_focus_words = [
            strip_punctuation(w).lower()
            for w in english_focus.strip().split()
            if strip_punctuation(w)
        ]
        if not en_focus_words:
            fallback = strip_punctuation(tgt_tokens[0]) if tgt_tokens else ""
            return fallback, [], "", [], []

        # 2. Identify the exact contiguous focus span in English sentence (src_tokens)
        src_clean = [strip_punctuation(w).lower() for w in src_tokens]
        en_focus_clean = [strip_punctuation(w).lower() for w in english_focus.strip().split() if strip_punctuation(w)]
        focus_src_start, focus_src_end = None, None

        ef_len = len(en_focus_clean)
        if ef_len > 0:
            for i in range(len(src_clean) - ef_len + 1):
                if [src_clean[i + k] for k in range(ef_len)] == en_focus_clean:
                    focus_src_start, focus_src_end = i, i + ef_len - 1
                    break

        # Fallback: density-penalized subsegment search
        if focus_src_start is None:
            best_density = -1.0
            best_window = (0, len(src_tokens) - 1)
            for i in range(len(src_tokens)):
                for j in range(i, len(src_tokens)):
                    w_words = [src_clean[k] for k in range(i, j + 1) if src_clean[k]]
                    matches = sum(1 for w in w_words if w in en_focus_clean)
                    density = matches / (j - i + 1) if (j >= i) else 0
                    if matches > 0 and density > best_density:
                        best_density = density
                        best_window = (i, j)
            best_window = (i, j)
            focus_src_start, focus_src_end = best_window

        # ONLY source indices strictly within [focus_src_start..focus_src_end] are valid!
        matching_src_indices = set(range(focus_src_start, focus_src_end + 1))
        # If English focus starts with a participle/verb or noun preceded by a functional preposition (e.g., 'from adding'),
        # include the immediately preceding preposition index so attached Malayalam verb/postposition tokens are retained.
        if focus_src_start > 0:
            prev_en_word = src_clean[focus_src_start - 1]
            if prev_en_word in {"from", "by", "in", "on", "at", "to", "for", "with", "of", "about", "through", "after", "before", "into", "onto", "upon"}:
                matching_src_indices.add(focus_src_start - 1)

        # 3. Collect all candidate target alignments and filter using Bidirectional Consensus
        candidate_alignments: List[Dict[str, Any]] = []
        seen_pairs: set = set()
        for item in src_to_tgt:
            s_i = item["src_index"]
            t_j = item["tgt_index"]
            if s_i in matching_src_indices and item.get("is_lexical", True):
                pair_key = (s_i, t_j)
                if pair_key not in seen_pairs and t_j < len(tgt_tokens):
                    seen_pairs.add(pair_key)
                    candidate_alignments.append({
                        "src_index": s_i,
                        "src_word": item["src_word"],
                        "tgt_index": t_j,
                        "tgt_word": tgt_tokens[t_j],
                        "is_bidirectional": item.get("is_bidirectional", False),
                    })

        # Reconcile bidirectional consensus: if bidirectional pairs exist for the focus, discard unverified single-pass links
        bidirectional_cands = [c for c in candidate_alignments if c.get("is_bidirectional", False)]
        if bidirectional_cands:
            candidate_alignments = bidirectional_cands
        else:
            # If no bidirectional consensus links exist for focus, discard noisy single-pass links that conflict with reverse alignment
            candidate_alignments = [
                c for c in candidate_alignments
                if not any(
                    rev_item.get("tgt_index") == c["tgt_index"]
                    and rev_item.get("is_lexical", True)
                    and rev_item.get("src_index") not in matching_src_indices
                    for rev_item in tgt_to_src
                )
            ]

        if not candidate_alignments:
            if matching_src_indices and len(src_tokens) > 0 and len(tgt_tokens) > 0:
                sorted_src_indices = sorted(matching_src_indices)
                src_mid_ratio = (sorted_src_indices[0] + sorted_src_indices[-1]) / (2.0 * max(1, len(src_tokens) - 1))
                approx_tgt_idx = min(int(round(src_mid_ratio * (len(tgt_tokens) - 1))), len(tgt_tokens) - 1)
                candidate_alignments.append({
                    "src_index": sorted_src_indices[0],
                    "src_word": src_tokens[sorted_src_indices[0]],
                    "tgt_index": approx_tgt_idx,
                    "tgt_word": tgt_tokens[approx_tgt_idx],
                    "is_fallback": True,
                })
            else:
                fallback = strip_punctuation(tgt_tokens[0]) if tgt_tokens else ""
                return fallback, [], "", [], []

        # 4. Unique candidate target indices, excluding un-focused matrix verb and pure punctuation tokens (e.g. ':', ',', '.')
        cand_indices_set = set(
            c["tgt_index"] for c in candidate_alignments
            if c["tgt_index"] < len(tgt_tokens) and strip_punctuation(tgt_tokens[c["tgt_index"]]).strip()
        )
        filtered_cand_indices = set()
        for t_j in cand_indices_set:
            has_direct_focus_align = any(
                item.get("tgt_index") == t_j and item.get("is_lexical", True) and item.get("src_index") in matching_src_indices
                for item in src_to_tgt
            )
            aligns_outside = any(
                item.get("tgt_index") == t_j and item.get("is_lexical", True) and item.get("src_index") not in matching_src_indices
                for item in tgt_to_src
            )
            if aligns_outside and not has_direct_focus_align and self._is_verbal_token(tgt_tokens[t_j]):
                continue
            filtered_cand_indices.add(t_j)
        cand_indices_set = filtered_cand_indices
        matrix_verb_idx = self._find_matrix_verb_idx(tgt_tokens)
        if matrix_verb_idx is not None:
            verb_aligned_to_focus = any(c["tgt_index"] == matrix_verb_idx for c in candidate_alignments)
            clean_verb = strip_punctuation(tgt_tokens[matrix_verb_idx])
            is_complement_verb = any(clean_verb.endswith(sfx) for sfx in ("എന്ന്", "എന്നു്", "എന്നു", "ഉണ്ടെന്ന്", "ആണെന്ന്", "എന്നതിന്", "എന്നതിനോട്", "എന്നതോടെ", "എന്നതുപോലെ", "എന്നതിനെ", "എന്നതിൽ"))
            earlier_matrix_verb = any(
                i < matrix_verb_idx and self._is_verbal_token(tgt_tokens[i])
                for i in range(len(tgt_tokens))
            )
            has_non_verbal_cand = any(idx != matrix_verb_idx and not self._is_verbal_token(tgt_tokens[idx]) for idx in cand_indices_set)
            
            if has_non_verbal_cand:
                # Prioritize non-verbal NP focus constituent over matrix verb swallowing
                cand_indices_set.discard(matrix_verb_idx)
            elif not verb_aligned_to_focus and not is_complement_verb and not earlier_matrix_verb:
                cand_indices_set.discard(matrix_verb_idx)

        # Filter candidate indices separated by a main matrix verb if the primary cluster is after it
        comp_verb_idx = None
        if matrix_verb_idx is not None:
            clean_mv = strip_punctuation(tgt_tokens[matrix_verb_idx])
            if any(clean_mv.endswith(sfx) for sfx in ("എന്ന്", "എന്നു്", "ഉണ്ടെന്ന്", "ആണെന്ന്")):
                comp_verb_idx = matrix_verb_idx

        main_mv_idx = self._find_matrix_verb_for_clause(tgt_tokens, comp_verb_idx=comp_verb_idx)

        if main_mv_idx is not None:
            after_mv_cand = [idx for idx in cand_indices_set if idx > main_mv_idx]
            before_mv_cand = [idx for idx in cand_indices_set if idx < main_mv_idx]
            has_comp_postp = any(
                strip_punctuation(tgt_tokens[b_i]) in ("എന്ന", "എന്നുള്ള", "എന്നതിന്", "എന്നതിനോട്", "എന്നതോടെ", "എന്നതുപോലെ", "എന്നതിനെ", "എന്നതിൽ", "ആനുപാതികമായി")
                for b_i in before_mv_cand
            )
            if not has_comp_postp and len(after_mv_cand) >= len(before_mv_cand) and len(after_mv_cand) > 0:
                for b_idx in before_mv_cand:
                    cand_indices_set.discard(b_idx)

        # Shift candidate anchor leftward if candidate is immediately preceded by an adverbial postposition (e.g. ആനുപാതികമായി / പ്രകാരം / അനുസരിച്ച്)
        shifted_cand_indices = set()
        for idx in cand_indices_set:
            if idx > 0:
                prev_clean = strip_punctuation(tgt_tokens[idx - 1])
                is_adv_postp = any(prev_clean.endswith(sfx) for sfx in ("മായി", "ായി", "കൊണ്ട്", "പ്രകാരം", "അനുസരിച്ച്", "പകരമായി")) and not any(prev_clean.endswith(sfx) for sfx in ("യുമായി", "ഇയുമായി", "നുമായി", "വോടുമായി", "യോടുമായി"))
                if is_adv_postp:
                    shifted_cand_indices.add(idx - 1)
                else:
                    shifted_cand_indices.add(idx)
            else:
                shifted_cand_indices.add(idx)
        cand_indices_set = shifted_cand_indices

        cand_indices = sorted(cand_indices_set)
        if not cand_indices:
            if matching_src_indices and len(src_tokens) > 0 and len(tgt_tokens) > 0:
                src_mid_ratio = (matching_src_indices[0] + matching_src_indices[-1]) / (2.0 * max(1, len(src_tokens) - 1))
                approx_tgt_idx = min(int(round(src_mid_ratio * (len(tgt_tokens) - 1))), len(tgt_tokens) - 1)
                cand_indices = [approx_tgt_idx]
            else:
                fallback = strip_punctuation(tgt_tokens[0]) if tgt_tokens else ""
                return fallback, candidate_alignments, "", [], []

        # 5. Group candidates into strictly contiguous clusters
        clusters: List[List[int]] = []
        cur: List[int] = [cand_indices[0]]
        for idx in cand_indices[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                clusters.append(cur)
                cur = [idx]
        clusters.append(cur)

        # If a copula 'ആണ്'-bearing token exists within aligned focus candidates, use it as anchor
        aanu_cand_idx = None
        for idx in cand_indices:
            tok = strip_punctuation(tgt_tokens[idx])
            if tok.endswith("ആണ്") or tok.endswith("ാണ്"):
                if any(c["tgt_index"] == idx and c["src_index"] in matching_src_indices for c in candidate_alignments):
                    aanu_cand_idx = idx
                    break

        # 6. Head anchor — copula-bearing token if aligned, else candidate with highest focus alignment coverage & positional proximity
        if aanu_cand_idx is not None:
            head_anchor = aanu_cand_idx
        else:
            sorted_src_indices = sorted(matching_src_indices) if matching_src_indices else [0]
            src_mid_ratio = (sorted_src_indices[0] + sorted_src_indices[-1]) / (2.0 * max(1, len(src_tokens) - 1))
            approx_tgt_idx = min(int(round(src_mid_ratio * (len(tgt_tokens) - 1))), len(tgt_tokens) - 1)

            def _cand_score(idx: int) -> Tuple[int, int, int]:
                focus_cnt = sum(
                    1 for c in candidate_alignments
                    if c.get("tgt_index") == idx and c.get("src_index") in matching_src_indices
                )
                dist_penalty = -abs(idx - approx_tgt_idx)
                return (focus_cnt, dist_penalty, -idx)

            head_anchor = max(cand_indices, key=_cand_score)

        head_ci = len(clusters) - 1
        for ci, cl in enumerate(clusters):
            if head_anchor in cl:
                head_ci = ci
                break

        # 7. Merge with adjacent left clusters using NP-modifier bridging.
        merged_start = clusters[head_ci][0]
        merged_end = head_anchor if aanu_cand_idx is not None else clusters[head_ci][-1]
        # Strip numeric formatting commas (e.g. 1,000 -> 1000) when checking for structural text commas
        ef_text_no_num_commas = re.sub(r'(\d),(\d)', r'\1\2', english_focus)
        has_internal_commas = (
            "," in ef_text_no_num_commas
            or " or " in f" {english_focus.lower()} "
            or " and " in f" {english_focus.lower()} "
        )

        # Identify any preceding temporal/participial/comparative postposition or clausal boundary
        # (e.g. ശേഷം, മുമ്പ്, നേതൃത്വത്തിൽ, സംബന്ധിച്ച്, പോലെ, എന്നാൽ, എങ്കിലും, പ്രകാരം, കാരണം, അനുസരിച്ച്)
        postp_boundary_idx = None
        for i in range(head_anchor):
            tok_clean = strip_punctuation(tgt_tokens[i])
            if tok_clean in (
                "ശേഷം", "മുമ്പ്", "നേതൃത്വത്തിൽ", "കുറിച്ച്", "സംബന്ധിച്ച്", "കൊണ്ട്",
                "പോലെ", "എന്നാൽ", "എങ്കിലും", "പ്രകാരം", "കാരണം", "തുടങ്ങി", "അനുസരിച്ച്"
            ):
                postp_boundary_idx = i

        def _is_gap_connective(tok: str) -> bool:
            cln = strip_punctuation(tok).strip()
            if not cln:
                return True
            if self._is_np_modifier(cln):
                return True
            if cln.endswith("ഉം") or cln.endswith("ും") or cln.endswith("വും") or cln.endswith("യും") or cln.endswith("ഓ") or cln.endswith("യോ"):
                return True
            if cln in ("കൂടാതെ", "അഥവാ", "അല്ലെങ്കിൽ", "അതായത്", "എന്നിവ", "എന്നീ", "മുതലായ", "എന്ന", "എന്നുള്ള", "എന്നതിന്", "എന്നതിനോട്", "എന്നതോടെ", "എന്നതുപോലെ", "എന്നതിനെ", "എന്നതിൽ"):
                return True
            # General Syntactic Rule: Any non-clausal, non-verbal token inside NP gap is an NP modifier
            if not self._is_verbal_token(cln):
                return True
            return False

        for ci in range(head_ci - 1, -1, -1):
            left_cl = clusters[ci]
            gap_start = left_cl[-1] + 1
            gap_end = merged_start

            # Stop leftward merge if left cluster crosses a preceding postposition boundary
            if postp_boundary_idx is not None and left_cl[0] <= postp_boundary_idx:
                valid_left = [idx for idx in left_cl if idx > postp_boundary_idx]
                if valid_left:
                    merged_start = valid_left[0]
                break

            # (a) gap size cap
            max_gap = 6 if has_internal_commas else 3
            if gap_end - gap_start > max_gap:
                break

            # (b) gap token NP modifier / connective check
            if not all(_is_gap_connective(tgt_tokens[g]) for g in range(gap_start, gap_end)):
                break

            # (c) left cluster or gap must not contain clausal verbs
            if self._has_clausal_verb(tgt_tokens, left_cl):
                break

            # (e) gap tokens must not have lexical alignment with English tokens outside focus
            has_outside_alignment = False
            for g in range(gap_start, gap_end):
                gap_tok_clean = strip_punctuation(tgt_tokens[g]).strip()
                if gap_tok_clean in ("എന്ന", "എന്നുള്ള", "ഒരു", "എ", "ദ", "ദി", "ദ്"):
                    continue
                for item in tgt_to_src:
                    if item.get("tgt_index") == g and item.get("is_lexical", True):
                        if item.get("src_index") not in matching_src_indices:
                            has_outside_alignment = True
                            break
                if has_outside_alignment:
                    break
            if has_outside_alignment:
                break

            # All checks passed — extend span leftward
            merged_start = left_cl[0]

        if postp_boundary_idx is not None and merged_start <= postp_boundary_idx:
            if postp_boundary_idx + 1 <= merged_end:
                merged_start = postp_boundary_idx + 1

        # 7a. Disjunctive series extension for multi-clause focus
        if has_internal_commas:
            min_allowed = (postp_boundary_idx + 1) if postp_boundary_idx is not None else 0
            while merged_start > min_allowed:
                prev_tok = tgt_tokens[merged_start - 1]
                if (
                    prev_tok.endswith(",")
                    or prev_tok.endswith("ഓ")
                    or prev_tok.endswith("യോ")
                    or prev_tok.endswith("കൂടാതെ")
                ):
                    merged_start -= 1
                else:
                    break

        # 7b. Parenthesis & Bracket Balance Protection
        merged_start, merged_end = self._expand_parenthesis_bounds(tgt_tokens, merged_start, merged_end)

        # 7c. Focus particle extension (e.g. മാത്രം / മാത്രമേ / മാത്രമായി) when 'only' or 'just' is focused
        if merged_end + 1 < len(tgt_tokens):
            next_tok_clean = strip_punctuation(tgt_tokens[merged_end + 1]).lower()
            if next_tok_clean in ("മാത്രം", "മാത്രമേ", "മാത്രമാണ്", "മാത്രമായി", "മാത്രമായിട്ട്", "അല്ലാതെ", "ഒഴികെ") or next_tok_clean.startswith("മാത്ര"):
                if "only" in english_focus.lower() or "just" in english_focus.lower() or "merely" in english_focus.lower():
                    merged_end += 1

        # 7d. Rightward verbal participle & modal verb extension (e.g. വിൽക്കാം എന്ന് / തുകൊണ്ട് / ക്കുന്നത്)
        if merged_end + 1 < len(tgt_tokens):
            next_clean = strip_punctuation(tgt_tokens[merged_end + 1])
            if matrix_verb_idx is None or (merged_end + 1) != matrix_verb_idx:
                if any(next_clean.endswith(sfx) for sfx in ("തുകൊണ്ട്", "ക്കുന്നത്", "ക്കുന്നതുകൊണ്ട്", "ന്നത്", "ത്തത്", " ചെയ്തത്", " ചെയ്യുന്നത്", "ാം", "പറ്റും", "സാധിക്കും", "കഴിയും")):
                    merged_end += 1

        # 7e. Expand leftward to complement clause boundary (after main matrix verb) if applicable
        if main_mv_idx is not None and merged_start > main_mv_idx + 1:
            can_expand_clause = True
            for g_i in range(main_mv_idx + 1, merged_start):
                tok_clean = strip_punctuation(tgt_tokens[g_i])
                if not (self._is_np_modifier(tok_clean) or any(tok_clean.endswith(sfx) for sfx in ("നു", "ന്", "ക്കും", "ക്ക്", "യിൽ", "ൽ", "ത്തിന്റെ", "ന്റെ", "തും", "ത്")) or tok_clean in ("അതിനു", "ഇതിനു", "അതിന്", "ഇതിന്", "അവന്", "അവൾക്ക്", "അവർക്ക്", "അത്", "ഇത്")):
                    can_expand_clause = False
                    break
            if can_expand_clause:
                merged_start = main_mv_idx + 1

        # 7f. Rightward conjoined nominal extension (e.g. ആവശ്യങ്ങളും + മുൻഗണനകളും)
        if merged_end + 1 < len(tgt_tokens):
            curr_clean = strip_punctuation(tgt_tokens[merged_end])
            next_clean = strip_punctuation(tgt_tokens[merged_end + 1])
            conjoined_suffixes = ("ും", "യും", "വും", "തും", "കയും", "ഉം", "ം")
            if any(curr_clean.endswith(sfx) for sfx in conjoined_suffixes) and any(next_clean.endswith(sfx) for sfx in conjoined_suffixes):
                merged_end += 1

        # 7g. Participial & Postpositional Appositive Extension (e.g. എന്ന ചിത്രത്തോടു കൂടി / എന്ന പേരിൽ)
        if merged_end + 1 < len(tgt_tokens):
            tok_next1 = strip_punctuation(tgt_tokens[merged_end + 1])
            if tok_next1 in ("എന്ന", "എന്നുള്ള"):
                if merged_end + 3 < len(tgt_tokens) and strip_punctuation(tgt_tokens[merged_end + 3]) in ("കൂടി", "കൂടെ"):
                    merged_end += 3
                elif merged_end + 2 < len(tgt_tokens) and any(strip_punctuation(tgt_tokens[merged_end + 2]).endswith(sfx) for sfx in ("ൽ", "യിൽ", "ത്തിൽ", "ത്ത്", "കൊണ്ട്")):
                    merged_end += 2
                elif merged_end + 1 < len(tgt_tokens):
                    merged_end += 1

        # 7h. Associative case postposition extension (-ത്തോടു + കൂടി / -ഓടു + കൂടി)
        if merged_end + 1 < len(tgt_tokens):
            curr_clean = strip_punctuation(tgt_tokens[merged_end])
            next_clean = strip_punctuation(tgt_tokens[merged_end + 1])
            if any(curr_clean.endswith(sfx) for sfx in ("ത്തോടു", "ഓടു", "ത്തോട്", "ോട്", "നോട്", "നോടു")) or next_clean in ("കൂടി", "കൂടെ", "ഒപ്പം"):
                if next_clean in ("കൂടി", "കൂടെ", "ഒപ്പം"):
                    merged_end += 1

        # 7i. Complement Clause / Dative Postposition Leftward Expansion (e.g. എന്നതിന് ആനുപാതികമായി / എന്ന നിബന്ധനയിൽ)
        # Note: Zero direct alignment evidence must not trigger unrestricted constituent expansion over neighboring clauses
        has_direct_focus_evidence = any(
            c.get("src_index") in matching_src_indices and c.get("is_lexical", True) and not c.get("is_fallback", False)
            for c in candidate_alignments
        )
        if has_direct_focus_evidence:
            start_check_idx = merged_start if strip_punctuation(tgt_tokens[merged_start]) in ("എന്ന", "എന്നുള്ള", "എന്നതിന്", "എന്നതിനോട്", "എന്നതോടെ", "എന്നതുപോലെ", "എന്നതിനെ", "എന്നതിൽ") else (merged_start - 1)
            if start_check_idx >= 0:
                check_tok_clean = strip_punctuation(tgt_tokens[start_check_idx])
                if check_tok_clean in ("എന്ന", "എന്നുള്ള", "എന്നതിന്", "എന്നതിനോട്", "എന്നതോടെ", "എന്നതുപോലെ", "എന്നതിനെ", "എന്നതിൽ"):
                    curr_i = start_check_idx
                    while curr_i - 1 >= 0:
                        cand_tok = strip_punctuation(tgt_tokens[curr_i - 1])
                        if self._is_verbal_token(cand_tok):
                            if not any(cand_tok.endswith(sfx) for sfx in ("ുക", "ക്കുക", "ഇക്കൽ", "ക്കൽ", "ഉള്ള", "ുള്ള", "ുന്ന", "ാത്ത", "പ്പെട്ട", "ായ", "ിയ", "ച്ച", "ത്ത", "ന്ന", "ഉക")):
                                break
                        if cand_tok.endswith(",") or cand_tok.endswith("."):
                            break
                        curr_i -= 1
                    if curr_i < merged_start:
                        merged_start = curr_i
            if merged_end + 1 < len(tgt_tokens):
                next_clean = strip_punctuation(tgt_tokens[merged_end + 1])
                curr_clean = strip_punctuation(tgt_tokens[merged_end])
                conjoined_suffixes = ("ും", "യും", "വും", "തും", "കയും", "ഉം", "ം")
                if any(curr_clean.endswith(sfx) for sfx in conjoined_suffixes) and any(next_clean.endswith(sfx) for sfx in conjoined_suffixes):
                    merged_end += 1

        # 7g. Participial & Postpositional Appositive Extension (e.g. എന്ന ചിത്രത്തോടു കൂടി / എന്ന പേരിൽ)
        if merged_end + 1 < len(tgt_tokens):
            tok_next1 = strip_punctuation(tgt_tokens[merged_end + 1])
            if tok_next1 in ("എന്ന", "എന്നുള്ള"):
                if merged_end + 3 < len(tgt_tokens) and strip_punctuation(tgt_tokens[merged_end + 3]) in ("കൂടി", "കൂടെ"):
                    merged_end += 3
                elif merged_end + 2 < len(tgt_tokens) and any(strip_punctuation(tgt_tokens[merged_end + 2]).endswith(sfx) for sfx in ("ൽ", "യിൽ", "ത്തിൽ", "ത്ത്", "കൊണ്ട്")):
                    merged_end += 2
                elif merged_end + 1 < len(tgt_tokens):
                    merged_end += 1

        # 7h. Associative case postposition extension (-ത്തോടു + കൂടി / -ഓടു + കൂടി)
        if merged_end + 1 < len(tgt_tokens):
            curr_clean = strip_punctuation(tgt_tokens[merged_end])
            next_clean = strip_punctuation(tgt_tokens[merged_end + 1])
            if any(curr_clean.endswith(sfx) for sfx in ("ത്തോടു", "ഓടു", "ത്തോട്", "ോട്", "നോട്", "നോടു")) or next_clean in ("കൂടി", "കൂടെ", "ഒപ്പം"):
                if next_clean in ("കൂടി", "കൂടെ", "ഒപ്പം"):
                    merged_end += 1

        # 7j. Dative Complement Adverbial Postposition Rightward Extension (e.g. എന്നതിന് + ആനുപാതികമായി / തുല്യമായി / വിരുദ്ധമായി)
        if merged_end + 1 < len(tgt_tokens):
            curr_clean = strip_punctuation(tgt_tokens[merged_end])
            next_clean = strip_punctuation(tgt_tokens[merged_end + 1])
            if any(curr_clean.endswith(sfx) for sfx in ("തിന്", "തിനോട്", "തോടു", "തോട്", "ന്", "ിന്")):
                if any(next_clean.endswith(sfx) for sfx in ("മായി", "ായി", "ഇച്ച്", "ഉമായി", "മായിട്ട്")):
                    merged_end += 1

        # 7k. Trailing NP-Modifier Rightward Head Extension (e.g. വിവിധ -> അലങ്കാരമുദ്രകളില്നിന്ന്)
        if merged_end + 1 < len(tgt_tokens):
            curr_clean = strip_punctuation(tgt_tokens[merged_end])
            if self._is_np_modifier(curr_clean) and not any(curr_clean.endswith(sfx) for sfx in ("നിന്ന്", "നിന്നാണ്", "യിൽ", "ത്തിൽ", "ൽ", "ത്തേക്ക്", "ലേക്ക്")) and matrix_verb_idx != (merged_end + 1):
                next_clean = strip_punctuation(tgt_tokens[merged_end + 1])
                if next_clean and not self._is_verbal_token(tgt_tokens[merged_end + 1]) and not self._is_np_modifier(next_clean):
                    merged_end += 1
        if merged_start > merged_end:
            merged_start = merged_end
        focus_span_indices = list(range(merged_start, merged_end + 1))
        if not focus_span_indices:
            focus_span_indices = [head_anchor] if 'head_anchor' in locals() else [0]
        constituent_tokens = [
            strip_punctuation(tgt_tokens[i]) for i in focus_span_indices
        ]
        selected_focus = " ".join(t for t in constituent_tokens if t)
        reconstructed_span = (
            f"[{focus_span_indices[0]}..{focus_span_indices[-1]}]"
            f" -> '{selected_focus}'"
        )

        # Collect all candidate spans (merged constituent + individual clusters)
        candidate_spans = []
        candidate_spans.append({
            "indices": focus_span_indices,
            "is_selected": True,
            "type": "Head-noun anchored constituent (Merged NP)" if len(focus_span_indices) > len(clusters[head_ci]) else "Head-noun constituent cluster"
        })

        for cl in clusters:
            # Add other clusters if not identical to the selected span
            if cl != focus_span_indices and not (cl[0] >= merged_start and cl[-1] <= merged_end and len(focus_span_indices) > len(cl)):
                candidate_spans.append({
                    "indices": cl,
                    "is_selected": False,
                    "type": "Alternative aligned cluster"
                })

        # Build full candidate objects with alignment evidence
        focus_candidates = []
        seen_spans = set()
        for cand in candidate_spans:
            c_indices = cand["indices"]
            span_key = (c_indices[0], c_indices[-1])
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)

            c_tokens = [strip_punctuation(tgt_tokens[i]) for i in c_indices]
            c_text = " ".join(t for t in c_tokens if t)

            # Collect evidence alignments for this candidate
            c_evidence = [
                item for item in candidate_alignments
                if item["tgt_index"] in c_indices
            ]

            focus_candidates.append({
                "span_indices": c_indices,
                "span_range": f"[{c_indices[0]}..{c_indices[-1]}]",
                "span_text": c_text,
                "is_selected": cand["is_selected"],
                "candidate_type": cand["type"],
                "evidence_count": len(c_evidence),
                "evidence": c_evidence,
            })

        # Sort: Selected candidate first (Rank 1), then by evidence count descending
        focus_candidates.sort(key=lambda c: (0 if c["is_selected"] else 1, -c["evidence_count"], -len(c["span_indices"])))
        for rank, c in enumerate(focus_candidates, 1):
            c["rank"] = rank

        return selected_focus, candidate_alignments, reconstructed_span, focus_span_indices, focus_candidates

    # ------------------------------------------------------------------
    # Token-index-based <FF> tag insertion (replaces string matching)
    # ------------------------------------------------------------------

    def _insert_focus_tags(
        self,
        clean_ml_sent: str,
        clean_proj_focus: str,
        focus_span_indices: List[int],
    ) -> str:
        """
        Insert <FF>…<FF> tags around the focus constituent using token
        indices.  Falls back to token-level matching with punctuation
        stripping.  **Never** prepends focus text before the sentence.
        """
        ml_tokens = clean_ml_sent.split()

        # Strategy 1: Index-based insertion (primary — from _select_focus_constituent)
        if focus_span_indices:
            start = focus_span_indices[0]
            end = focus_span_indices[-1]
            if 0 <= start < len(ml_tokens) and 0 <= end < len(ml_tokens):
                ml_tokens_copy = list(ml_tokens)
                end_tok = ml_tokens_copy[end]
                punct = ""
                while end_tok and end_tok[-1] in (",", ".", ";", "!", "?"):
                    punct = end_tok[-1] + punct
                    end_tok = end_tok[:-1]
                ml_tokens_copy[end] = end_tok

                before = " ".join(ml_tokens_copy[:start])
                focus_part = " ".join(ml_tokens_copy[start : end + 1])
                after = " ".join(ml_tokens_copy[end + 1 :])
                focus_tagged = f"<FF>{focus_part}<FF>{punct}"
                parts = [p for p in [before, focus_tagged, after] if p]
                return " ".join(parts)

        # Strategy 2: Token-level matching (handles punctuation differences)
        if clean_proj_focus:
            focus_words = clean_proj_focus.split()
            fc_len = len(focus_words)

            # Contiguous match with punctuation stripping
            for i in range(len(ml_tokens) - fc_len + 1):
                window = [strip_punctuation(ml_tokens[i + k]) for k in range(fc_len)]
                if window == focus_words:
                    before = " ".join(ml_tokens[:i])
                    focus_part = " ".join(ml_tokens[i : i + fc_len])
                    after = " ".join(ml_tokens[i + fc_len :])
                    parts = [p for p in [before, f"<FF>{focus_part}<FF>", after] if p]
                    return " ".join(parts)

            # Single-word fallback
            if fc_len == 1:
                for i, tok in enumerate(ml_tokens):
                    if strip_punctuation(tok) == focus_words[0]:
                        ml_tokens[i] = f"<FF>{tok}<FF>"
                        return " ".join(ml_tokens)

        # Strategy 3: Legacy exact-string match
        if clean_proj_focus and clean_proj_focus in clean_ml_sent:
            return clean_ml_sent.replace(
                clean_proj_focus, f"<FF>{clean_proj_focus}<FF>", 1
            )

        # NEVER prepend — return untagged; CleftPipeline will report the error
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

        if english_focus and not projected_ml_focus:
            projected_ml_focus, candidate_alignments, reconstructed_span, focus_span_indices, focus_candidates = self._select_focus_constituent(
                clean_en_sent=clean_en_sent,
                clean_ml_sent=clean_ml_sent,
                english_focus=english_focus,
                pre_alignment=pre_alignment,
            )

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
                clean_ml_sent, clean_proj_focus, focus_span_indices
            )

            cleft_res = self.ssf_pipeline.process(tagged_ml_sent)
            cleft_dict = cleft_res.to_dict()
            emphasized_ml_sentence = cleft_res.cleft_sentence or clean_ml_sent
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

        result = {
            "english_sentence": clean_en_sent,
            "original_malayalam_sentence": clean_ml_sent,
            "english_focus": english_focus,
            "focused_malayalam_constituent": clean_proj_focus,
            "focus_candidates": focus_candidates,
            "cleft_engine": self.cleft_engine,
            "english_prosody_label_sequence": en_prosody_labels,
            "malayalam_prosody_label_sequence": cleft_dict.get("prosody_label_sequence", []),
            "pre_cleft_en_to_ml_alignment": pre_alignment,
            "cleft_pipeline_output": cleft_dict,
            "emphasized_malayalam_sentence": emphasized_ml_sentence,
            "post_cleft_ml_to_en_alignment": post_align_ml_to_en,
            "post_cleft_en_to_ml_alignment": post_align_en_to_ml,
        }

        # Step 5: Clause-Aware Cleft Positional Reordering
        reorderer = CleftReorderer()
        result["cleft_reorderings"] = reorderer.reorder(result)

        return result


def generate_pipeline_report(res: Dict[str, Any]) -> str:
    """Generate a complete, structured text report of all pipeline stages and diagnostics."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"1. English Sentence            : {res['english_sentence']}")
    lines.append(f"2. Original Malayalam Sentence : {res['original_malayalam_sentence']}")
    lines.append(f"3. English Focus Marker        : {res['english_focus'] or '—'}")
    lines.append(f"4. Selected Malayalam Focus    : {res['focused_malayalam_constituent']}")

    # --- Focus Projection Candidates ---
    candidates = res.get("focus_candidates", [])
    lines.append("\n" + "-" * 80)
    lines.append("--- Malayalam Focus Candidates (Focus Projection) ---")
    lines.append("-" * 80)
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
    for pair in res["pre_cleft_en_to_ml_alignment"].get("aligned_pairs", []):
        lines.append(f"  {pair[0]:<25} <---> {pair[1]:<25}")

    # --- Clefting Pipeline Output ---
    cleft = res["cleft_pipeline_output"]
    lines.append("\n" + "-" * 80)
    lines.append("--- Clefting Pipeline Output & Diagnostics ---")
    lines.append("-" * 80)
    lines.append(f"  * Emphasized Malayalam Sentence             : {res['emphasized_malayalam_sentence']}")
    lines.append(f"  * Prosody Label Sequence (Source / English)  : {res['english_prosody_label_sequence']}")
    lines.append(f"  * Prosody Label Sequence (Target / Malayalam): {res['malayalam_prosody_label_sequence']}")
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
    for item in res["post_cleft_ml_to_en_alignment"].get("src_to_tgt_alignments", []):
        lines.append(f"  [{item['src_index']}] {item['src_word']:<25} ---> [{item['tgt_index']}] {item['tgt_word']:<25}")

    # --- Post-cleft English -> Malayalam Alignment ---
    lines.append("\n" + "-" * 80)
    lines.append("--- Post-Cleft Alignment (English -> Malayalam) ---")
    lines.append("-" * 80)
    for item in res["post_cleft_en_to_ml_alignment"].get("src_to_tgt_alignments", []):
        lines.append(f"  [{item['src_index']}] {item['src_word']:<25} ---> [{item['tgt_index']}] {item['tgt_word']:<25}")

    # --- Cleft Positional Reorderings ---
    reorderings = res.get("cleft_reorderings", {})
    if reorderings and reorderings.get("preverbal_focus"):
        lines.append("\n" + "-" * 80)
        lines.append("--- Cleft Positional Reorderings ---")
        lines.append("-" * 80)
        lines.append(f"  * Preverbal (Contrastive)      : {reorderings['preverbal_focus']}")
        lines.append(f"  * Postverbal (Information)      : {reorderings['postverbal_focus']}")
       # lines.append(f"  * Clause-Initial (Strong)       : {reorderings['clause_initial_focus']}")
        comp = reorderings.get("components", {})
        cc = comp.get("cleft_clause", {})
        if cc:
            lines.append(f"  * Components:")
            if comp.get("pre_clauses"):
                lines.append(f"      Pre-clauses      : {comp['pre_clauses']}")
            lines.append(f"      Background       : {cc.get('background', [])}")
            lines.append(f"      Focus+ആണ്         : {cc.get('focus_copula', [])}")
            lines.append(f"      Nominalized Verb : {cc.get('nominalized_verb', [])}")
            if comp.get("post_clauses"):
                lines.append(f"      Post-clauses     : {comp['post_clauses']}")

    lines.append("\n" + "=" * 80 + "\n")
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

    # Minimal terminal output
    print("=" * 68)
    print("✓ Cross-Lingual Focus Transfer & Clefting Completed")
    print("=" * 68)
    print(f"• English Focus               : {res['english_focus'] or '—'}")
    print(f"• Selected Malayalam Focus   : {res['focused_malayalam_constituent']}")
    print(f"• Emphasized Malayalam Output: {res['emphasized_malayalam_sentence']}")
    reorderings = res.get("cleft_reorderings", {})
    if reorderings and reorderings.get("preverbal_focus"):
        print(f"• Preverbal  (Contrastive)   : {reorderings['preverbal_focus']}")
        print(f"• Postverbal (Information)   : {reorderings['postverbal_focus']}")
        print(f"• Clause-Initial (Strong)    : {reorderings['clause_initial_focus']}")
    print(f"• Focus Candidates Evaluated : {len(res.get('focus_candidates', []))}")
    print(f"• Full report & diagnostics   : {abs_out_path}")
    print("=" * 68)


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
    parser.add_argument("--output", "-out", type=str, default="output.txt", help="Output file path (default: output.txt)")

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

    pipeline = AwesomeCleftPipeline()
    res = pipeline.process(
        english_sentence=en_sent,
        malayalam_sentence=ml_sent,
        english_focus=en_focus,
        malayalam_focus=args.ml_focus,
        operation=args.op,
        mt_model=mt_model,
    )

    print_pipeline_report(res, output_path=args.output)


if __name__ == "__main__":
    main()
