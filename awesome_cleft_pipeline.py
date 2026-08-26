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
        print(f">>> [Awesome-Aligner] Initializing model '{self.model_name}' (Device: {self.device})...")
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
            print(">>> [Awesome-Aligner] Model loaded successfully.")
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

        # Step 1b: Focus Projection (If English focus provided but Malayalam focus missing)
        projected_ml_focus = malayalam_focus
        if english_focus and not projected_ml_focus:
            en_clean_focus = strip_punctuation(english_focus).lower()
            for align_item in pre_alignment["src_to_tgt_alignments"]:
                src_w = strip_punctuation(align_item["src_word"]).lower()
                if src_w == en_clean_focus or en_clean_focus in src_w:
                    projected_ml_focus = align_item["tgt_word"]
                    break

        if not projected_ml_focus:
            # Fallback to first word of Malayalam sentence
            ml_words = clean_ml_sent.strip().split()
            projected_ml_focus = ml_words[0] if ml_words else ""

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

        return {
            "english_sentence": clean_en_sent,
            "original_malayalam_sentence": clean_ml_sent,
            "english_focus": english_focus,
            "focused_malayalam_constituent": clean_proj_focus,
            "cleft_engine": self.cleft_engine,
            "pre_cleft_en_to_ml_alignment": pre_alignment,
            "cleft_pipeline_output": cleft_dict,
            "emphasized_malayalam_sentence": emphasized_ml_sentence,
            "post_cleft_ml_to_en_alignment": post_align_ml_to_en,
            "post_cleft_en_to_ml_alignment": post_align_en_to_ml,
        }


def print_pipeline_report(res: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(" CROSS-LINGUAL FOCUS TRANSFER & DUAL ALIGNMENT REPORT")
    print("=" * 70)
    print(f"1. English Sentence            : {res['english_sentence']}")
    print(f"2. Original Malayalam Sentence : {res['original_malayalam_sentence']}")
    print(f"3. English Focus Marker        : {res['english_focus'] or '—'}")
    print(f"4. Projected Malayalam Focus   : {res['focused_malayalam_constituent']}")

    print("\n--- [Step 1] Pre-Cleft Alignment (English -> Malayalam) ---")
    for pair in res["pre_cleft_en_to_ml_alignment"]["aligned_pairs"]:
        print(f"  {pair[0]:<20} <---> {pair[1]:<20}")

    cleft = res["cleft_pipeline_output"]
    print("\n--- [Step 2] Clefting Pipeline Output ---")
    print(f"  * Emphasized Malayalam Sentence : {res['emphasized_malayalam_sentence']}")
    print(f"  * Prosody Label Sequence        : {cleft['prosody_label_sequence']}")
    print(f"  * Focus Position (Before -> After): {cleft['position_before']} -> {cleft['position_after']}")

    aanu = cleft.get("aanu_attachment", {})
    if aanu and aanu.get("attached_to") != "none":
        print(f"  * ആണ് Attachment                : {aanu.get('attached_form')} (attached to {aanu.get('attached_to')} '{aanu.get('target_word')}')")
    print(f"  * Nominalized Verb              : {cleft['nominalized_verb']}")

    print("\n--- [Step 3a] Post-Cleft Alignment (Malayalam -> English) ---")
    for item in res["post_cleft_ml_to_en_alignment"]["src_to_tgt_alignments"]:
        print(f"  [{item['src_index']}] {item['src_word']:<20} ---> [{item['tgt_index']}] {item['tgt_word']:<20}")

    print("\n--- [Step 3b] Post-Cleft Alignment (English -> Malayalam) ---")
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

    args = parser.parse_args()

    en_sent = args.en or "Father bought a book in the garden yesterday."
    ml_sent = args.ml or "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി."
    en_focus = args.en_focus or "Father"

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
