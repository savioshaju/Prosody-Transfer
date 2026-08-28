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

# Add awesome-align path
AWESOME_ALIGN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "awesome-align")
if AWESOME_ALIGN_DIR not in sys.path:
    sys.path.append(AWESOME_ALIGN_DIR)

import torch
from ssf_pipeline.alignment_utils import extract_alignment_and_prosody, strip_punctuation
from bhashik_focus_reorderer import BhashikFocusReorderer


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

    def align(self, src_sentence: str, tgt_sentence: str) -> Dict[str, Any]:
        """
        Align words between src_sentence and tgt_sentence.
        Returns bidirectional alignment mappings and alignment pair lists.
        """
        sent_src = src_sentence.strip().split()
        sent_tgt = tgt_sentence.strip().split()

        try:
            self._load_model()
        except Exception:
            self._loaded = False

        alignments = []
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

                alignments = word_aligns_list[0]
            except Exception as e:
                alignments = self._heuristic_align(sent_src, sent_tgt)
        else:
            alignments = self._heuristic_align(sent_src, sent_tgt)

        src_to_tgt = []
        tgt_to_src = []
        aligned_pairs = []

        for align_pair in alignments:
            s_i, t_j = align_pair[0], align_pair[1]
            if s_i != -1 and t_j != -1 and s_i < len(sent_src) and t_j < len(sent_tgt):
                src_word = sent_src[s_i]
                tgt_word = sent_tgt[t_j]
                aligned_pairs.append((src_word, tgt_word))
                src_to_tgt.append({
                    "src_index": s_i,
                    "src_word": src_word,
                    "tgt_index": t_j,
                    "tgt_word": tgt_word
                })
                tgt_to_src.append({
                    "tgt_index": t_j,
                    "tgt_word": tgt_word,
                    "src_index": s_i,
                    "src_word": src_word
                })

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

        # 2. Surface finite verb suffix heuristics
        finite_verb_suffixes = (
            "ുന്നു", "ിച്ചു", "ച്ചു", "ഞ്ഞു", "ന്നു",
            "ാറുണ്ട്", "ാറില്ല", "ചെയ്യുന്നു", "ഉണ്ട്", "ആണ്", "ആയിരുന്നു"
        )
        if any(clean_w.endswith(sfx) for sfx in finite_verb_suffixes):
            if clean_w not in ("ഇന്ത്യൻ", "വടക്കൻ", "തെക്കൻ", "പലതും", "എല്ലാം", "മറ്റും"):
                return True

        return False

    def _select_focus_constituent(
        self,
        clean_en_sent: str,
        clean_ml_sent: str,
        english_focus: str,
        pre_alignment: Dict[str, Any],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Collect all candidate Malayalam tokens aligned to the English focus,
        determine which token or contiguous Malayalam constituent corresponds
        to the semantic focus (never selecting finite/main verbs over nominals),
        and return the selected constituent along with candidate alignments for debugging.
        """
        src_tokens = pre_alignment.get("src_tokens", clean_en_sent.split())
        tgt_tokens = pre_alignment.get("tgt_tokens", clean_ml_sent.split())
        src_to_tgt = pre_alignment.get("src_to_tgt_alignments", [])

        # 1. Normalize English focus words
        en_focus_words = [strip_punctuation(w).lower() for w in english_focus.strip().split() if strip_punctuation(w)]
        if not en_focus_words:
            fallback = strip_punctuation(tgt_tokens[0]) if tgt_tokens else ""
            return fallback, []

        # 2. Identify matching source indices in English sentence
        matching_src_indices = set()
        for i, src_w in enumerate(src_tokens):
            c_w = strip_punctuation(src_w).lower()
            if c_w in en_focus_words or any(fw == c_w or (len(fw) >= 3 and (fw in c_w or c_w in fw)) for fw in en_focus_words):
                matching_src_indices.add(i)

        # 3. Collect all candidate alignments (tgt_index, tgt_word, src_index, src_word)
        candidate_alignments = []
        seen_pairs = set()
        for item in src_to_tgt:
            s_i = item["src_index"]
            t_j = item["tgt_index"]
            src_w = strip_punctuation(item["src_word"]).lower()
            if s_i in matching_src_indices or src_w in en_focus_words or any(fw in src_w for fw in en_focus_words):
                pair_key = (s_i, t_j)
                if pair_key not in seen_pairs and t_j < len(tgt_tokens):
                    seen_pairs.add(pair_key)
                    candidate_alignments.append({
                        "src_index": s_i,
                        "src_word": item["src_word"],
                        "tgt_index": t_j,
                        "tgt_word": tgt_tokens[t_j],
                    })

        if not candidate_alignments:
            fallback = strip_punctuation(tgt_tokens[0]) if tgt_tokens else ""
            return fallback, []

        # 4. Separate candidate Malayalam indices into nominal/argument and verbal
        cand_indices = sorted(list(set(c["tgt_index"] for c in candidate_alignments)))
        non_verb_indices = [idx for idx in cand_indices if not self._is_verbal_token(tgt_tokens[idx])]
        verb_indices = [idx for idx in cand_indices if self._is_verbal_token(tgt_tokens[idx])]

        # Use non-verb indices if present to prevent spurious verb focus
        selected_indices = non_verb_indices if non_verb_indices else verb_indices

        if not selected_indices:
            selected_indices = cand_indices

        # 5. Form contiguous constituent span for multi-word focus phrases
        min_idx = min(selected_indices)
        max_idx = max(selected_indices)
        span_indices = list(range(min_idx, max_idx + 1))
        span_length = len(span_indices)
        max_allowed_span = len(en_focus_words) + 2

        # If all tokens in the span from min_idx to max_idx are non-verbal,
        # bridge the span into a single complete multi-word constituent (e.g. 'റോബര്‍ട്ട് ബോബി ജോര്‍ജ്ജിനെ')
        all_non_verbal = all(not self._is_verbal_token(tgt_tokens[i]) for i in span_indices)

        if all_non_verbal and (span_length <= max_allowed_span or len(en_focus_words) > 1):
            constituent_tokens = [strip_punctuation(tgt_tokens[i]) for i in span_indices]
            selected_focus = " ".join(t for t in constituent_tokens if t)
            reconstructed_span = f"[{min_idx}..{max_idx}] -> '{selected_focus}'"
            return selected_focus, candidate_alignments, reconstructed_span

        # Otherwise, group contiguous clusters and select the longest cluster
        clusters = []
        current_cluster = [selected_indices[0]]
        for idx in selected_indices[1:]:
            if idx == current_cluster[-1] + 1:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)

        # Select the longest contiguous cluster
        best_cluster = max(clusters, key=len)

        # Extract complete contiguous constituent words
        constituent_tokens = [strip_punctuation(tgt_tokens[i]) for i in range(best_cluster[0], best_cluster[-1] + 1)]
        selected_focus = " ".join(t for t in constituent_tokens if t)
        reconstructed_span = f"[{best_cluster[0]}..{best_cluster[-1]}] -> '{selected_focus}'"

        return selected_focus, candidate_alignments, reconstructed_span

    def process(
        self,
        english_sentence: str,
        malayalam_sentence: str,
        english_focus: str = "",
        malayalam_focus: str = "",
        operation: str = "CLEFT",
        focus_type: str = "INFORMATION",
    ) -> Dict[str, Any]:
        """
        Execute full cross-lingual emphasis transfer pipeline.

        Inputs:
          - english_sentence: e.g. "Father bought a book in the garden yesterday." (or with <FF>Father<FF>)
          - malayalam_sentence: e.g. "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി." (or with <FF>അച്ഛൻ<FF>)
          - english_focus: Optional focused English word/phrase (e.g. "Father")
          - malayalam_focus: Optional focused Malayalam constituent (e.g. "അച്ഛൻ")
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

        # Step 1: Pre-Cleft Alignment (English -> Malayalam)
        pre_alignment = self.aligner.align(clean_en_sent, clean_ml_sent)

        # Step 1b: Focus Projection & Semantic Constituent Resolution
        projected_ml_focus = malayalam_focus
        candidate_alignments = []
        reconstructed_span = ""

        if english_focus and not projected_ml_focus:
            projected_ml_focus, candidate_alignments, reconstructed_span = self._select_focus_constituent(
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
            # Insert <FF>...<FF> around target focus constituent
            if clean_proj_focus and clean_proj_focus in clean_ml_sent:
                tagged_ml_sent = clean_ml_sent.replace(clean_proj_focus, f"<FF>{clean_proj_focus}<FF>", 1)
            else:
                tagged_ml_sent = f"<FF>{clean_proj_focus}<FF> {clean_ml_sent}"

            cleft_res = self.ssf_pipeline.process(tagged_ml_sent)
            cleft_dict = cleft_res.to_dict()
            emphasized_ml_sentence = cleft_res.cleft_sentence or clean_ml_sent

        # Step 3: Post-Cleft Dual Alignment (Awesome-Aligner)
        # 3a. Malayalam -> English alignment
        post_align_ml_to_en = self.aligner.align(emphasized_ml_sentence, clean_en_sent)

        # 3b. English -> Malayalam alignment
        post_align_en_to_ml = self.aligner.align(clean_en_sent, emphasized_ml_sentence)

        # Compute English (Source) Prosody Label Sequence
        en_tokens = clean_en_sent.split()
        en_focus_words = [strip_punctuation(w).lower() for w in (english_focus or "").split() if strip_punctuation(w)]
        en_prosody_labels = []
        for w in en_tokens:
            cw = strip_punctuation(w).lower()
            if cw in en_focus_words or any(fw == cw for fw in en_focus_words):
                en_prosody_labels.append(1)
            else:
                en_prosody_labels.append(0)

        return {
            "english_sentence": clean_en_sent,
            "original_malayalam_sentence": clean_ml_sent,
            "english_focus": english_focus,
            "focused_malayalam_constituent": clean_proj_focus,
            "cleft_engine": self.cleft_engine,
            "english_prosody_label_sequence": en_prosody_labels,
            "malayalam_prosody_label_sequence": cleft_dict.get("prosody_label_sequence", []),
            "pre_cleft_en_to_ml_alignment": pre_alignment,
            "cleft_pipeline_output": cleft_dict,
            "emphasized_malayalam_sentence": emphasized_ml_sentence,
            "post_cleft_ml_to_en_alignment": post_align_ml_to_en,
            "post_cleft_en_to_ml_alignment": post_align_en_to_ml,
        }


def print_pipeline_report(res: Dict[str, Any]):
    print(f"1. English Sentence            : {res['english_sentence']}")
    print(f"2. Original Malayalam Sentence : {res['original_malayalam_sentence']}")
    print(f"3. English Focus Marker        : {res['english_focus'] or '—'}")
    print(f"4. Projected Malayalam Focus   : {res['focused_malayalam_constituent']}")

    print("\n--- English -> Malayalam ---")
    for pair in res["pre_cleft_en_to_ml_alignment"]["aligned_pairs"]:
        print(f"  {pair[0]:<20} <---> {pair[1]:<20}")

    cleft = res["cleft_pipeline_output"]
    print("\n--- Clefting Pipeline Output ---")
    print(f"  * Emphasized Malayalam Sentence             : {res['emphasized_malayalam_sentence']}")
    print(f"  * Prosody Label Sequence (Source / English)  : {res['english_prosody_label_sequence']}")
    print(f"  * Prosody Label Sequence (Target / Malayalam): {res['malayalam_prosody_label_sequence']}")
    print(f"  * Focus Position (Before -> After)          : {cleft['position_before']} -> {cleft['position_after']}")

    aanu = cleft.get("aanu_attachment", {})
    if aanu and aanu.get("attached_to") != "none":
        print(f"  * ആണ് Attachment                : {aanu.get('attached_form')} (attached to {aanu.get('attached_to')} '{aanu.get('target_word')}')")
    print(f"  * Nominalized Verb              : {cleft['nominalized_verb']}")

    print("\n--- Malayalam -> English ---")
    for item in res["post_cleft_ml_to_en_alignment"]["src_to_tgt_alignments"]:
        print(f"  [{item['src_index']}] {item['src_word']:<20} ---> [{item['tgt_index']}] {item['tgt_word']:<20}")

    print("\n--- English -> Malayalam ---")
    for item in res["post_cleft_en_to_ml_alignment"]["src_to_tgt_alignments"]:
        print(f"  [{item['src_index']}] {item['src_word']:<20} ---> [{item['tgt_index']}] {item['tgt_word']:<20}")

    print("=" * 70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-Lingual Focus Transfer & Dual Alignment Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--en", "-e", type=str, help="English sentence (e.g. 'Father bought a book in the garden yesterday.')")
    parser.add_argument("--ml", "-m", type=str, help="Malayalam baseline sentence (e.g. 'അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി.')")
    parser.add_argument("--en_focus", "-ef", type=str, default="", help="English focus word (e.g. 'Father')")
    parser.add_argument("--ml_focus", "-mf", type=str, default="", help="Malayalam focus word (optional, derived via alignment if omitted)")
    parser.add_argument("--op", "-o", type=str, default="CLEFT", choices=["CLEFT", "FRONTING"], help="Focus operation")

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

    en_sent = args.en or "Father bought a book in the garden yesterday."
    ml_sent = args.ml or "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി."
    en_focus = args.en_focus or ""

    # If extra positional arguments were provided (e.g. from shell quoting), append to focus
    if unknown and not en_focus:
        en_focus = " ".join(unknown)
    elif unknown and en_focus:
        en_focus = f"{en_focus} {' '.join(unknown)}"

    en_focus = en_focus.strip().lstrip("=").strip().strip("'\"").strip()
    if not en_focus:
        en_focus = "Father"

    pipeline = AwesomeCleftPipeline()
    res = pipeline.process(
        english_sentence=en_sent,
        malayalam_sentence=ml_sent,
        english_focus=en_focus,
        malayalam_focus=args.ml_focus,
        operation=args.op,
    )

    print_pipeline_report(res)


if __name__ == "__main__":
    main()
