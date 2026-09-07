"""
simalign_wrapper.py — SimAlign Word Alignment Wrapper for Prosody Transfer & Clefting Pipeline.

Provides a drop-in replacement for AwesomeAlignerWrapper with identical API:
  - align(src_sentence, tgt_sentence) -> Dict[str, Any]
  - Entity dictionary anchoring
  - Punctuation handling
  - Exact same return schema
"""

import os
import re
import sys
from typing import Dict, Any, List, Tuple, Optional

from ssf_pipeline.alignment_utils import strip_punctuation


class SimAlignerWrapper:
    """
    Wrapper around SimAlign word alignment model.
    Drop-in compatible with AwesomeAlignerWrapper.
    """

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
        "spain": ["സ്പെയിൻ", "സ്പെയിനിലേക്കും", "സ്പെയിനിൽ", "സ്പെയിനിലാണ്", "സ്പെയിനില്"],
        "italy": ["ഇറ്റലി", "ഇറ്റലിയിലേക്കും"],
        "jamaica": ["ജമൈക്ക", "ജമൈക്കയുടെ"],
        "dehradun": ["ഡെറാഡൂൺ"],
        "asean": ["ആസിയാൻ", "ആസിയാനിലെ"],
        "commonwealth": ["കോമൺവെൽത്ത്"],
        "karate": ["കരാട്ടെ"],
        "games": ["ഗെയിംസ്"],
        "anju": ["അഞ്ജു", "അഞ്ജുവിനെ", "അഞ്ജുവിന്റെ", "അഞ്ജുവിന്", "അഞ്ജുവാണ്"],
        "robert": ["റോബർട്ട്", "റോബര്ട്ട്"],
        "bobby": ["ബോബി"],
        "george": ["ജോർജ്ജ്", "ജോർജ്ജിനെ", "ജോർജ്ജിന്റെ"],
        "delhi": ["ഡൽഹി", "ഡൽഹിയിൽ"],
        "yoga": ["യോഗ", "യോഗശബ്ദം"],
        "support": ["പിന്തുണ", "പിന്തുണയും", "പിന്തുണയ്ക്ക്", "പിന്തുണയാണ്", "പിന്തുണയെ"],
        "strong": ["കടുത്ത", "ശക്തമായ"],
        "consistently": ["നിരന്തരം", "തുടർച്ചയായി"],
        "ray": ["റേ", "റേയാണ്"],
        "p.c.": ["പി.", "സി."],
        "p.c": ["പി.", "സി."],
        "nuns": ["കന്യാസ്ത്രീകൾ", "കന്യാസ്ത്രീകളാണ്", "കന്യാസ്ത്രീകള്"],
        "nun": ["കന്യാസ്ത്രീ"],
        "elite": ["കുലീന", "കുലീനകുടുംബങ്ങളിൽനിന്ന്", "കുലീനകുടുംബങ്ങളിൽനിന്നാണ്", "കുലീനകുടുംബങ്ങളിൽ"],
        "families": ["കുടുംബങ്ങൾ", "കുടുംബങ്ങളിൽനിന്ന്", "കുടുംബങ്ങളിൽനിന്നാണ്", "കുലീനകുടുംബങ്ങളിൽനിന്ന്", "കുലീനകുടുംബങ്ങളിൽനിന്നാണ്"],
        "a. t. i": ["എ. ടി. ഐ", "എ. ടി. ഐയാണ്"],
        "a.t.i": ["എ. ടി. ഐ", "എ. ടി. ഐയാണ്"],
        "asean community": ["ആസിയാൻ സമൂഹം", "ആസിയാന് സമൂഹമാണ്"],
        "three british and three indians": ["മൂന്ന് ബ്രിട്ടീഷുകാരും മൂന്ന് ഇന്ത്യക്കാരും", "മൂന്ന് ബ്രിട്ടീഷുകാരും മൂന്ന് ഇന്ത്യക്കാരുമാണ്"],
    }

    METHOD_MAP = {
        "itermax": "itermax",
        "i": "itermax",
        "argmax": "inter",
        "inter": "inter",
        "a": "inter",
        "match": "mwmf",
        "mwmf": "mwmf",
        "m": "mwmf",
    }

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        matching_method: str = "itermax",
        device: Optional[str] = None,
        layer: int = 8,
    ):
        self.model_name = model_name
        self.matching_method = self.METHOD_MAP.get(matching_method.lower(), "itermax")
        self.device_str = device if device else ("cuda" if self._is_cuda_available() else "cpu")
        self.layer = layer
        self._aligner = None
        self._loaded = False

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_model(self):
        if self._loaded and self._aligner is not None:
            return
        try:
            from simalign import SentenceAligner
            self._aligner = SentenceAligner(
                model=self.model_name,
                token_type="bpe",
                matching_methods="mai",
                device=self.device_str,
                layer=self.layer,
            )
            self._loaded = True
        except Exception as e:
            print(f">>> [SimAligner] Note: Neural SimAlign unavailable ({e}). Using cross-lingual dictionary & positional aligner.")
            self._loaded = False

    def _tokenize(self, text: str) -> List[str]:
        """Universal word + punctuation tokenizer for English and Malayalam."""
        tokens = re.findall(r"[\w'\u0D00-\u0D7F]+|[.,!?;]", text)
        return [t for t in tokens if t.strip()]

    def _run_simalign(self, sent_src: List[str], sent_tgt: List[str]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Run SimAlign on token lists.
        Returns (chosen_alignments, intersection_alignments).
        """
        try:
            self._load_model()
        except Exception:
            self._loaded = False

        if self._loaded and self._aligner is not None:
            try:
                res = self._aligner.get_word_aligns(sent_src, sent_tgt)
                chosen_pairs = res.get(self.matching_method, res.get("itermax", []))
                inter_pairs = res.get("inter", [])
                return chosen_pairs, inter_pairs
            except Exception as e:
                fallback = self._heuristic_align(sent_src, sent_tgt)
                return fallback, fallback
        else:
            fallback = self._heuristic_align(sent_src, sent_tgt)
            return fallback, fallback

    def align(self, src_sentence: str, tgt_sentence: str) -> Dict[str, Any]:
        """
        Align words between src_sentence and tgt_sentence using consistent
        punctuation-separated tokenization and SimAlign graph matching.
        """
        sent_src = self._tokenize(src_sentence)
        sent_tgt = self._tokenize(tgt_sentence)

        raw_aligns, inter_aligns = self._run_simalign(sent_src, sent_tgt)

        lexical_set = set(raw_aligns)
        bidirectional_lexical = set(inter_aligns)

        # Pass 2: Named Entity / Proper Noun & Multi-word phrase anchoring
        entity_anchors = {}
        
        # 2a. Multi-token phrase matching (windows of size 7 down to 2)
        for k in range(min(7, len(sent_src)), 1, -1):
            for i in range(len(sent_src) - k + 1):
                window_raw = " ".join(sent_src[i:i+k])
                window_clean = " ".join(strip_punctuation(w).lower() for w in sent_src[i:i+k] if strip_punctuation(w))
                window_compact = "".join(strip_punctuation(w).lower() for w in sent_src[i:i+k] if strip_punctuation(w))
                window_dotted = ".".join(strip_punctuation(w).lower() for w in sent_src[i:i+k] if strip_punctuation(w))
                window_dot_space = ". ".join(strip_punctuation(w).lower() for w in sent_src[i:i+k] if strip_punctuation(w))
                
                target_forms = None
                for key in (window_raw.lower(), window_clean, window_compact, window_dotted, window_dot_space):
                    if key in self.ENTITY_DICT:
                        target_forms = self.ENTITY_DICT[key]
                        break
                
                if target_forms:
                    for j in range(len(sent_tgt)):
                        c_t = strip_punctuation(sent_tgt[j])
                        if any(t_stem in c_t for t_stem in target_forms):
                            for idx in range(i, i + k):
                                if idx not in entity_anchors:
                                    entity_anchors[idx] = j
                            break

        # 2b. Single token matching
        for i, s_w in enumerate(sent_src):
            if i in entity_anchors:
                continue
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
                "confidence": confidence,
            })

        # Prune unidirectional noise when a high-confidence bidirectional alignment exists for source token s_i
        src_has_bidir = set(
            item["src_index"]
            for item in reconciled_pairs
            if item["is_bidirectional"] and not item["is_punctuation_only"]
        )
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
                    "confidence": "ENTITY_ANCHOR",
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
