"""
morpho_simalign.py — Standalone Morphologically Segmented SimAlign Pipeline.

Addresses morphological bottlenecks in cross-lingual neural aligners (SimAlign / mBERT):
  1. Compound Word Splitting: Agglutinative Malayalam compounds (e.g., 'ശിക്ഷാനിയമം',
     'സ്വവർഗ്ഗരതി') are decomposed into constituent morphemes ('ശിക്ഷ', 'നിയമം') so multi-word
     English expressions ('Penal Code') can align directly to their respective components.
  2. Nominal Case Stripping: Inflectional case suffixes (e.g., locative '-യിൽ' in 'ഇന്ത്യയിൽ',
     dative '-ിന്' in 'ജോണിന്', instrumental '-കൊണ്ട്' in 'കത്തികൊണ്ട്') are stripped to yield
     base nominal roots ('ഇന്ത്യ', 'ജോൺ', 'കത്തി') preventing subword fragmentation and hubness.
  3. Reverse Index Projection: Alignments produced on decomposed tokens are projected
     back to the original Malayalam surface sentence tokens, preserving 100% fidelity for downstream
     prosody transfer, focus marking, and clefting.
  4. Closed-Class Adposition Binding: Extracted case markers (locative, dative, instrumental,
     genitive) are optionally bound to governing English prepositions ('in', 'to', 'with', 'of').
"""

import os
import re
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Tuple, Optional, Set

from aligner.alignment_utils import strip_punctuation


class MorphoSimAligner:
    """
    Standalone Word Aligner combining Malayalam morphological decomposition
    (via mlmorph) with neural subword representations (via SimAlign / mBERT).
    """

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

    # Closed-class English preposition to Malayalam grammatical case mapping
    ADPOSITION_CASE_MAP = {
        "in": "locative",
        "at": "locative",
        "on": "locative",
        "inside": "locative",
        "to": "dative",
        "for": "dative",
        "with": "instrumental",
        "by": "instrumental",
        "of": "genitive",
        "from": "ablative",
    }

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        matching_method: str = "itermax",
        device: Optional[str] = None,
        layer: int = 8,
        bind_adpositions: bool = True,
    ):
        self.model_name = model_name
        self.matching_method = self.METHOD_MAP.get(matching_method.lower(), "itermax")
        self.device_str = device if device else ("cuda" if self._is_cuda_available() else "cpu")
        self.layer = layer
        self.bind_adpositions = bind_adpositions

        self._simalign_model = None
        self._simalign_loaded = False

        self._mlmorph_analyser = None
        self._mlmorph_loaded = False

    @staticmethod
    def _is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_simalign(self):
        if self._simalign_loaded and self._simalign_model is not None:
            return
        try:
            from simalign import SentenceAligner
            self._simalign_model = SentenceAligner(
                model=self.model_name,
                token_type="bpe",
                matching_methods="mai",
                device=self.device_str,
                layer=self.layer,
            )
            self._simalign_loaded = True
        except Exception as e:
            print(f">>> [MorphoSimAligner] Warning: Neural SimAlign unavailable ({e}). Using heuristic fallback.")
            self._simalign_loaded = False

    def _load_mlmorph(self):
        if self._mlmorph_loaded and self._mlmorph_analyser is not None:
            return
        try:
            import mlmorph
            self._mlmorph_analyser = mlmorph.Analyser()
            self._mlmorph_loaded = True
        except Exception as e:
            print(f">>> [MorphoSimAligner] Warning: mlmorph unavailable ({e}). Decomposition will fallback to raw tokens.")
            self._mlmorph_loaded = False

    def tokenize(self, text: str) -> List[str]:
        """Universal word + punctuation tokenizer for English and Malayalam."""
        if not text:
            return []
        pattern = r"[\w\u0D00-\u0D7F\u200C\u200D]+(?:[.-][\w\u0D00-\u0D7F\u200C\u200D]+)*|[.,!?;:()\[\]\"'“”‘’]"
        tokens = re.findall(pattern, text)
        return [t for t in tokens if t.strip()]

    def decompose_token(self, token: str) -> List[Dict[str, Any]]:
        """
        Decompose a single Malayalam surface token using mlmorph:
          - Splits compounds into morphemes
          - Strips case markers to base citation roots
          - Returns list of decomposed morpheme dictionaries
        """
        clean_word = strip_punctuation(token).strip()
        if not clean_word or not re.search(r"[\u0D00-\u0D7F]", clean_word):
            return [{
                "stem": token,
                "case": None,
                "is_compound": False,
                "is_punctuation": not bool(clean_word),
                "analysis_str": None,
            }]

        self._load_mlmorph()
        if not self._mlmorph_loaded or self._mlmorph_analyser is None:
            return [{
                "stem": token,
                "case": None,
                "is_compound": False,
                "is_punctuation": False,
                "analysis_str": None,
            }]

        analyses = self._mlmorph_analyser.analyse(clean_word)
        if not analyses:
            return [{
                "stem": token,
                "case": None,
                "is_compound": False,
                "is_punctuation": False,
                "analysis_str": None,
            }]

        top_analysis_str = analyses[0][0]

        # Detect case tags
        case_set = {
            "locative", "dative", "accusative", "instrumental",
            "genitive", "sociative", "vocative", "ablative", "nominative"
        }
        all_tags = re.findall(r"<([^>]+)>", top_analysis_str)
        detected_case = None
        for tag in all_tags:
            if tag in case_set:
                detected_case = tag
                break

        # Extract morpheme stems by splitting XML tags
        raw_morphemes = [p for p in re.split(r"<[^>]+>", top_analysis_str) if p]
        if not raw_morphemes:
            raw_morphemes = [token]

        is_compound = len(raw_morphemes) > 1

        res = []
        for m in raw_morphemes:
            res.append({
                "stem": m,
                "case": detected_case,
                "is_compound": is_compound,
                "is_punctuation": False,
                "analysis_str": top_analysis_str,
            })
        return res

    def decompose_target_sentence(
        self, target_tokens: List[str]
    ) -> Tuple[List[str], Dict[int, List[int]], List[int], List[Dict[str, Any]]]:
        """
        Decompose a list of Malayalam surface tokens into stems and compound parts.
        Returns:
          - decomp_tokens: list of decomposed string tokens to feed into SimAlign
          - orig_to_decomp: map from original token index -> list of decomposed token indices
          - decomp_to_orig: list mapping each decomposed token index -> original token index
          - morpho_metadata: list of morphological metadata for each decomposed token
        """
        decomp_tokens: List[str] = []
        orig_to_decomp: Dict[int, List[int]] = {i: [] for i in range(len(target_tokens))}
        decomp_to_orig: List[int] = []
        morpho_metadata: List[Dict[str, Any]] = []

        for orig_idx, token in enumerate(target_tokens):
            sub_units = self.decompose_token(token)
            for unit in sub_units:
                decomp_idx = len(decomp_tokens)
                stem = unit["stem"]
                decomp_tokens.append(stem)
                decomp_to_orig.append(orig_idx)
                orig_to_decomp[orig_idx].append(decomp_idx)
                unit["orig_index"] = orig_idx
                unit["decomp_index"] = decomp_idx
                unit["surface_token"] = token
                morpho_metadata.append(unit)

        return decomp_tokens, orig_to_decomp, decomp_to_orig, morpho_metadata

    def align(self, src_sentence: str, tgt_sentence: str) -> Dict[str, Any]:
        """
        Align an English sentence and a Malayalam sentence using Morphologically
        Segmented SimAlign with full reverse index projection to original tokens.
        """
        src_tokens = self.tokenize(src_sentence)
        tgt_tokens = self.tokenize(tgt_sentence)

        if not src_tokens or not tgt_tokens:
            return {
                "source_tokens": src_tokens,
                "target_tokens": tgt_tokens,
                "reconciled_pairs": [],
                "unaligned_source_words": [],
                "unaligned_target_words": [],
                "alignment_map_forward": {},
                "alignment_map_reverse": {},
                "coverage": 0.0,
                "aligner_type": "morpho_simalign",
            }

        # Step 1: Morphological decomposition of target Malayalam tokens
        (
            decomp_tokens,
            orig_to_decomp,
            decomp_to_orig,
            morpho_meta,
        ) = self.decompose_target_sentence(tgt_tokens)

        # Step 2: Run Neural SimAlign on (src_tokens, decomp_tokens)
        raw_pairs, inter_pairs = self._run_neural_simalign(src_tokens, decomp_tokens)

        bidirectional_set = set(inter_pairs)
        lexical_pairs_set = set(raw_pairs)

        # Step 3: Project alignments back to original surface tokens
        projected_pairs_dict: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for s_i, d_j in sorted(lexical_pairs_set):
            if s_i < 0 or s_i >= len(src_tokens) or d_j < 0 or d_j >= len(decomp_tokens):
                continue

            orig_t_j = decomp_to_orig[d_j]
            src_w = src_tokens[s_i]
            tgt_w = tgt_tokens[orig_t_j]
            meta = morpho_meta[d_j]

            c_src = strip_punctuation(src_w).strip()
            c_tgt = strip_punctuation(tgt_w).strip()
            is_pure_punct = (not c_src) and (not c_tgt)

            is_bidir = (s_i, d_j) in bidirectional_set

            if is_pure_punct:
                conf = "STRUCTURAL"
            elif is_bidir:
                conf = "HIGH_BIDIRECTIONAL"
            else:
                conf = "HIGH"

            pair_key = (s_i, orig_t_j)
            if pair_key not in projected_pairs_dict:
                projected_pairs_dict[pair_key] = {
                    "src_index": s_i,
                    "src_word": src_w,
                    "tgt_index": orig_t_j,
                    "tgt_word": tgt_w,
                    "matched_stem": meta["stem"],
                    "is_compound": meta["is_compound"],
                    "case": meta["case"],
                    "is_lexical": not is_pure_punct,
                    "is_bidirectional": is_bidir,
                    "is_punctuation_only": is_pure_punct,
                    "confidence": conf,
                }
            else:
                # If already present, update confidence if bidir
                if is_bidir:
                    projected_pairs_dict[pair_key]["is_bidirectional"] = True
                    projected_pairs_dict[pair_key]["confidence"] = "HIGH_BIDIRECTIONAL"

        # Filter out spurious punctuation alignments
        final_projected: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for (s_i, orig_t_j), p_data in projected_pairs_dict.items():
            src_is_punct = not strip_punctuation(p_data["src_word"]).strip()
            tgt_is_punct = not strip_punctuation(p_data["tgt_word"]).strip()

            # Punctuation must only align to punctuation
            if src_is_punct != tgt_is_punct:
                continue

            final_projected[(s_i, orig_t_j)] = p_data

        projected_pairs_dict = final_projected

        # Step 4: Closed-class adposition binding
        if self.bind_adpositions:
            self._bind_adpositions_to_cases(
                src_tokens, tgt_tokens, projected_pairs_dict, morpho_meta, decomp_to_orig
            )

        # Build list of final reconciled pairs
        final_pairs: List[Dict[str, Any]] = list(projected_pairs_dict.values())

        final_pairs.sort(key=lambda x: (x["src_index"], x["tgt_index"]))

        # Build forward and reverse maps
        aligned_src_indices = set(p["src_index"] for p in final_pairs)
        aligned_tgt_indices = set(p["tgt_index"] for p in final_pairs)

        unaligned_src = [
            {"index": i, "word": w}
            for i, w in enumerate(src_tokens)
            if i not in aligned_src_indices and strip_punctuation(w).strip()
        ]
        unaligned_tgt = [
            {"index": j, "word": w}
            for j, w in enumerate(tgt_tokens)
            if j not in aligned_tgt_indices and strip_punctuation(w).strip()
        ]

        alignment_map_forward: Dict[int, List[int]] = {}
        alignment_map_reverse: Dict[int, List[int]] = {}

        for p in final_pairs:
            s = p["src_index"]
            t = p["tgt_index"]
            alignment_map_forward.setdefault(s, []).append(t)
            alignment_map_reverse.setdefault(t, []).append(s)

        # Calculate coverage
        src_lexical_count = sum(1 for w in src_tokens if strip_punctuation(w).strip())
        aligned_src_lexical = sum(1 for p in final_pairs if p["is_lexical"])
        coverage = aligned_src_lexical / max(src_lexical_count, 1)

        return {
            "source_tokens": src_tokens,
            "target_tokens": tgt_tokens,
            "decomposed_target_tokens": decomp_tokens,
            "decomp_to_orig_map": decomp_to_orig,
            "orig_to_decomp_map": orig_to_decomp,
            "reconciled_pairs": final_pairs,
            "unaligned_source_words": unaligned_src,
            "unaligned_target_words": unaligned_tgt,
            "alignment_map_forward": alignment_map_forward,
            "alignment_map_reverse": alignment_map_reverse,
            "coverage": coverage,
            "morpho_metadata": morpho_meta,
            "aligner_type": "morpho_simalign",
            "model_name": self.model_name,
            "matching_method": self.matching_method,
        }

    def _bind_adpositions_to_cases(
        self,
        src_tokens: List[str],
        tgt_tokens: List[str],
        projected_pairs: Dict[Tuple[int, int], Dict[str, Any]],
        morpho_meta: List[Dict[str, Any]],
        decomp_to_orig: List[int],
    ):
        """
        Bind unaligned or weak English prepositions ('in', 'to', 'with', 'of')
        to Malayalam tokens carrying corresponding morphological case suffixes.
        """
        aligned_src = set(k[0] for k in projected_pairs.keys())

        for s_i, word in enumerate(src_tokens):
            w_lower = word.lower()
            if w_lower not in self.ADPOSITION_CASE_MAP:
                continue

            # If preposition is directly before a noun that aligned to target token t_j
            expected_case = self.ADPOSITION_CASE_MAP[w_lower]
            next_s_i = s_i + 1

            # Find target token aligned to the object of the preposition
            target_t_j = None
            # Check if next word or subsequent word is aligned
            for lookahead in range(1, 4):
                cand_idx = s_i + lookahead
                if cand_idx >= len(src_tokens):
                    break
                for (s_k, t_k) in projected_pairs.keys():
                    if s_k == cand_idx:
                        target_t_j = t_k
                        break
                if target_t_j is not None:
                    break

            if target_t_j is not None:
                # Check if this target token carried the expected case
                has_matching_case = any(
                    m["case"] == expected_case and m["orig_index"] == target_t_j
                    for m in morpho_meta
                )
                if has_matching_case:
                    pair_key = (s_i, target_t_j)
                    if pair_key not in projected_pairs:
                        projected_pairs[pair_key] = {
                            "src_index": s_i,
                            "src_word": word,
                            "tgt_index": target_t_j,
                            "tgt_word": tgt_tokens[target_t_j],
                            "matched_stem": f"[CASE_BIND:{expected_case}]",
                            "is_compound": False,
                            "case": expected_case,
                            "is_lexical": True,
                            "is_bidirectional": True,
                            "is_punctuation_only": False,
                            "confidence": "HIGH_BIDIRECTIONAL",
                        }

    def _run_neural_simalign(
        self, sent_src: List[str], sent_tgt: List[str]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Run SimAlign on token lists.
        Returns (chosen_alignments, intersection_alignments).
        """
        try:
            self._load_simalign()
        except Exception:
            self._simalign_loaded = False

        if self._simalign_loaded and self._simalign_model is not None:
            try:
                res = self._simalign_model.get_word_aligns(sent_src, sent_tgt)
                chosen_pairs = res.get(self.matching_method, res.get("itermax", []))
                inter_pairs = res.get("inter", [])
                return chosen_pairs, inter_pairs
            except Exception as e:
                print(f">>> [MorphoSimAligner] Error during neural inference: {e}")
                fallback = self._heuristic_align(sent_src, sent_tgt)
                return fallback, fallback
        else:
            fallback = self._heuristic_align(sent_src, sent_tgt)
            return fallback, fallback

    def _heuristic_align(self, sent_src: List[str], sent_tgt: List[str]) -> List[Tuple[int, int]]:
        """Lightweight positional fallback."""
        pairs = []
        n_s = len(sent_src)
        n_t = len(sent_tgt)
        if n_s == 0 or n_t == 0:
            return pairs

        for i in range(n_s):
            approx_j = int(round(i * (n_t - 1) / max(n_s - 1, 1)))
            approx_j = max(0, min(n_t - 1, approx_j))
            pairs.append((i, approx_j))

        return sorted(list(set(pairs)))

    def align_and_compare(self, src_sentence: str, tgt_sentence: str) -> Dict[str, Any]:
        """
        Run BOTH raw SimAlign and Morpho-Segmented SimAlign on the same sentence pair
        and return a structured comparison dictionary highlighting improvements.
        """
        from aligner.simalign_wrapper import SimAlignerWrapper

        raw_aligner = SimAlignerWrapper(model_name=self.model_name, matching_method=self.matching_method)
        raw_result = raw_aligner.align(src_sentence, tgt_sentence)

        morpho_result = self.align(src_sentence, tgt_sentence)

        return {
            "src_sentence": src_sentence,
            "tgt_sentence": tgt_sentence,
            "raw_simalign": raw_result,
            "morpho_simalign": morpho_result,
            "raw_coverage": raw_result["coverage"],
            "morpho_coverage": morpho_result["coverage"],
            "raw_pairs_count": len(raw_result["reconciled_pairs"]),
            "morpho_pairs_count": len(morpho_result["reconciled_pairs"]),
            "decomposed_tokens": morpho_result["decomposed_target_tokens"],
        }


def print_comparison_report(comp: Dict[str, Any]):
    """Print an aligned side-by-side terminal report comparing raw vs morpho-segmented."""
    print("=" * 80)
    print("MORPHO-SEGMENTED SIMALIGN vs. RAW PRETRAINED SIMALIGN")
    print("=" * 80)
    print(f"Source (EN): {comp['src_sentence']}")
    print(f"Target (ML): {comp['tgt_sentence']}")
    print(f"Decomposed (ML stems): {' '.join(comp['decomposed_tokens'])}")
    print("-" * 80)
    print(f"Coverage: Raw={comp['raw_coverage']:.1%} | Morpho-Segmented={comp['morpho_coverage']:.1%}")
    print(f"Aligned Pairs Count: Raw={comp['raw_pairs_count']} | Morpho-Segmented={comp['morpho_pairs_count']}")
    print("-" * 80)
    print(f"{'English Token':<18} | {'Raw SimAlign Target':<22} | {'Morpho-SimAlign Target':<25}")
    print("-" * 80)

    src_tokens = comp["morpho_simalign"]["source_tokens"]
    raw_fwd = comp["raw_simalign"]["alignment_map_forward"]
    morpho_fwd = comp["morpho_simalign"]["alignment_map_forward"]
    tgt_tokens = comp["morpho_simalign"]["target_tokens"]

    for s_i, src_w in enumerate(src_tokens):
        raw_targets = [tgt_tokens[t] for t in raw_fwd.get(s_i, []) if t < len(tgt_tokens)]
        raw_str = ", ".join(raw_targets) if raw_targets else "[UNALIGNED]"

        morpho_targets = [tgt_tokens[t] for t in morpho_fwd.get(s_i, []) if t < len(tgt_tokens)]
        morpho_str = ", ".join(morpho_targets) if morpho_targets else "[UNALIGNED]"

        diff_marker = "  "
        if raw_str != morpho_str:
            diff_marker = "* " if raw_str == "[UNALIGNED]" or len(morpho_targets) > len(raw_targets) else "~ "

        print(f"{diff_marker}{src_w:<16} | {raw_str:<22} | {morpho_str:<25}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Morpho-Segmented SimAligner")
    parser.add_argument("--src", type=str, default="Under Section 377 of the Indian Penal Code, homosexuality is a crime in India.")
    parser.add_argument("--tgt", type=str, default="ഇന്ത്യൻ ശിക്ഷാനിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതി ഇന്ത്യയിൽ കുറ്റകരം.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark suite of Malayalam compound & case sentences")
    args = parser.parse_args()

    aligner = MorphoSimAligner()

    if args.benchmark:
        test_suites = [
            (
                "Under Section 377 of the Indian Penal Code, homosexuality is a crime in India.",
                "ഇന്ത്യൻ ശിക്ഷാനിയമം 377 പ്രകാരം സ്വവർഗ്ഗരതി ഇന്ത്യയിൽ കുറ്റകരം."
            ),
            (
                "Mary gave a book to John.",
                "മേരി ജോണിന് ഒരു പുസ്തകം കൊടുത്തു."
            ),
            (
                "The boy cut the apple with a knife.",
                "ആൺകുട്ടി കത്തികൊണ്ട് ആപ്പിൾ മുറിച്ചു."
            ),
            (
                "Father bought a book in the garden yesterday.",
                "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ ഒരു പുസ്തകം വാങ്ങി."
            )
        ]
        for src, tgt in test_suites:
            comp = aligner.align_and_compare(src, tgt)
            print_comparison_report(comp)
            print()
    else:
        comp = aligner.align_and_compare(args.src, args.tgt)
        print_comparison_report(comp)
