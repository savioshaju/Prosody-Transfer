import os
import sys
import unicodedata
from typing import Optional, Dict, Any

# Ensure UTF-8 output encoding for Malayalam on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator


from ssf_pipeline.alignment_utils import extract_alignment_and_prosody


def normalize_malayalam(text: str) -> str:
    """Normalize Malayalam unicode characters and clean whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.strip().split())


class BhashikFocusReorderer:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the Bhashik Focus Reorderer.
        
        Args:
            model_dir: Path to the fine-tuned model directory (defaults to 'models/bhashik_focus_reorderer').
            device: 'cuda', 'cpu', or None (auto-detect).
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), "models", "bhashik_focus_reorderer")

        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found at: {model_dir}")

        self.model_dir = model_dir

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f">>> [Bhashik] Loading model from: {self.model_dir} (Device: {self.device})")

        # Load Tokenizer & Model directly from fine-tuned directory
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            do_lower_case=False,
            use_fast=False,
            keep_accents=True,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Cache special token IDs for IndicBART
        self.ml_id = self.tokenizer._convert_token_to_id_with_added_voc("<2ml>")
        self.bos_id = self.tokenizer._convert_token_to_id_with_added_voc("<s>")
        self.eos_id = self.tokenizer._convert_token_to_id_with_added_voc("</s>")
        self.pad_id = self.tokenizer._convert_token_to_id_with_added_voc("<pad>")

        print(">>> [Bhashik] Model loaded successfully and ready for inference.")

    def reorder(
        self,
        sentence: str,
        focused_constituent: str,
        operation: str = "CLEFT",
        focus_type: str = "INFORMATION",
        num_beams: int = 4,
        max_length: int = 64,
    ) -> Dict[str, Any]:
        """
        Reorder or cleft a Malayalam sentence based on focus conditioning.

        Args:
            sentence: Baseline Malayalam canonical sentence (e.g. "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി.")
            focused_constituent: Target word or phrase with prosodic focus (e.g. "അച്ഛൻ", "പുസ്തകം", "തോട്ടത്തിൽ")
            operation: 'CLEFT' or 'FRONTING'
            focus_type: 'INFORMATION' or 'CONTRASTIVE'
            num_beams: Beam search size (default: 4)
            max_length: Max generation length (default: 64)

        Returns:
            Dict containing:
                - 'input_sentence' / 'original_sentence': original sentence
                - 'clefted_sentence' / 'reordered_sentence': output sentence
                - 'focused_constituent': targeted constituent
                - 'operation': focus operation used
                - 'focus_type': focus type used
                - 'prosody_label_sequence': binary label sequence for source words (e.g. [1, 0, 0, 0, 0, 0])
                - 'position_before': word index/indices of focus before movement
                - 'position_after': word index/indices of focus after movement
                - 'aanu_attachment': details on copula attachment
                - 'nominalized_verb': nominalized main verb form
                - 'word_alignments': word-level alignment list before & after clefting
                - 'constituent_alignments': constituent-level alignment list
        """
        sentence = normalize_malayalam(sentence)
        fc = normalize_malayalam(focused_constituent)
        op = operation.upper()
        ftype = focus_type.upper()

        # 1. Transliterate to unified Devanagari representation for IndicBART
        orig_hi = UnicodeIndicTransliterator.transliterate(sentence, "ml", "hi")
        fc_hi = UnicodeIndicTransliterator.transliterate(fc, "ml", "hi")

        # 2. Build controlled prompt
        # Format: <focus:FC> <op:OP> <type:TYPE> Sentence </s> <2ml>
        prompt_dev = f"<focus:{fc_hi}> <op:{op}> <type:{ftype}> {orig_hi} </s> <2ml>"

        # 3. Tokenize
        inputs = self.tokenizer(prompt_dev, add_special_tokens=False, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # 4. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length,
                min_length=2,
                early_stopping=True,
                decoder_start_token_id=self.ml_id,
                bos_token_id=self.bos_id,
                eos_token_id=self.eos_id,
                pad_token_id=self.pad_id,
            )

        # 5. Decode & Transliterate back to Malayalam
        gen_raw = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        for sp in ["<2ml>", "<s>", "</s>", "<pad>", "<unk>"]:
            gen_raw = gen_raw.replace(sp, "")
        gen_hi = gen_raw.strip()

        reordered_ml = UnicodeIndicTransliterator.transliterate(gen_hi, "hi", "ml")
        reordered_ml = normalize_malayalam(reordered_ml)

        # 6. Extract full alignment, prosody label sequence, position tracking, aanu attachment & nominalized verb
        align_info = extract_alignment_and_prosody(
            original_sentence=sentence,
            clefted_sentence=reordered_ml,
            focused_constituent=fc,
        )

        res = {
            "input_sentence": sentence,
            "original_sentence": sentence,
            "focused_constituent": fc,
            "operation": op,
            "focus_type": ftype,
            "reordered_sentence": reordered_ml,
            "clefted_sentence": reordered_ml,
            "prosody_label_sequence": align_info["prosody_label_sequence"],
            "position_before": align_info["position_before"],
            "position_after": align_info["position_after"],
            "aanu_attachment": align_info["aanu_attachment"],
            "nominalized_verb": align_info["nominalized_verb"],
            "word_alignments": align_info["word_alignments"],
            "constituent_alignments": align_info["constituent_alignments"],
        }
        return res


def print_reorder_results(res: Dict[str, Any]):
    print("\n" + "=" * 65)
    print(" Bhashik Focus Reorderer — Output & Alignment Details")
    print("=" * 65)
    print(f"  * Original Sentence      : {res['original_sentence']}")
    print(f"  * Clefted Sentence       : {res['clefted_sentence']}")
    print(f"  * Focused Constituent    : {res['focused_constituent']} (Op: {res.get('operation', 'CLEFT')}, Type: {res.get('focus_type', 'INFORMATION')})")
    print(f"  * Prosody Label Sequence : {res['prosody_label_sequence']}")
    print(f"  * Position Before        : {res['position_before']}")
    print(f"  * Position After         : {res['position_after']}")

    aanu = res.get("aanu_attachment", {})
    if aanu and aanu.get("attached_to") != "none":
        print(f"  * ആണ് Attachment         : {aanu.get('attached_form')} (attached to {aanu.get('attached_to')} '{aanu.get('target_word')}' at index {aanu.get('position')})")
    else:
        print("  * ആണ് Attachment         : None")

    print(f"  * Nominalized Verb       : {res['nominalized_verb'] or '—'}")

    print("\n  --- Word Alignments (Source -> Target) ---")
    for wa in res.get("word_alignments", []):
        src_str = f"[{wa['src_index']}] {wa['src_word']}"
        tgt_str = f"[{wa['tgt_index']}] {wa['tgt_word']}" if wa["tgt_index"] is not None else "[—] UNMATCHED"
        print(f"    {src_str:<22} ---> {tgt_str:<25} ({wa['alignment_type']})")

    print("\n  --- Constituent Alignments ---")
    for ca in res.get("constituent_alignments", []):
        print(f"    * {ca['role']:<22}: Src {ca['src_position']} '{ca['src_form']}' ---> Tgt {ca['tgt_position']} '{ca['tgt_form']}'")
    print("=" * 65 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Bhashik Focus Reorderer — Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Direct CLI Reordering (Information Cleft)
  python bhashik_focus_reorderer.py --sentence "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി." --focus "അച്ഛൻ" --op CLEFT --type INFORMATION

  # 2. Constituent Fronting
  python bhashik_focus_reorderer.py --sentence "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി." --focus "പുസ്തകം" --op FRONTING

  # 3. Contrastive Cleft
  python bhashik_focus_reorderer.py --sentence "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി." --focus "അച്ഛൻ" --op CLEFT --type CONTRASTIVE

  # 4. Interactive Terminal Mode
  python bhashik_focus_reorderer.py --interactive
        """
    )
    parser.add_argument("--sentence", "-s", type=str, help="Baseline Malayalam canonical sentence")
    parser.add_argument("--focus", "-f", type=str, help="Focused word/phrase carrying prosodic accent")
    parser.add_argument("--op", "-o", type=str, default="CLEFT", choices=["CLEFT", "FRONTING", "cleft", "fronting"], help="Focus operation: CLEFT or FRONTING (default: CLEFT)")
    parser.add_argument("--type", "-t", type=str, default="INFORMATION", choices=["INFORMATION", "CONTRASTIVE", "information", "contrastive"], help="Focus type: INFORMATION or CONTRASTIVE (default: INFORMATION)")
    parser.add_argument("--model_dir", "-m", type=str, default=None, help="Path to fine-tuned model directory (optional)")
    parser.add_argument("--device", "-d", type=str, default=None, choices=["cuda", "cpu", "CUDA", "CPU"], help="Execution device: cuda or cpu (default: auto-detect GPU/CPU)")
    parser.add_argument("--num_beams", "-b", type=int, default=4, help="Beam search size (default: 4)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive testing console")

    args = parser.parse_args()

    reorderer = BhashikFocusReorderer(model_dir=args.model_dir, device=args.device)

    # 1. Interactive Mode
    if args.interactive:
        print("\n" + "=" * 65)
        print(" Bhashik Interactive Focus Reordering Console")
        print(" (Type 'exit' or 'q' at any prompt to quit)")
        print("=" * 65)

        while True:
            try:
                print("\n" + "-" * 50)
                sent_input = input("Enter Malayalam Sentence: ").strip()
                if sent_input.lower() in ("exit", "q", "quit"):
                    print("Exiting interactive console.")
                    break
                if not sent_input:
                    continue

                focus_input = input("Enter Focused Constituent: ").strip()
                if focus_input.lower() in ("exit", "q", "quit"):
                    break
                if not focus_input:
                    print("Error: Focused constituent cannot be empty.")
                    continue

                op_input = input("Enter Operation [CLEFT/FRONTING] (default: CLEFT): ").strip().upper()
                if not op_input:
                    op_input = "CLEFT"

                type_input = input("Enter Focus Type [INFORMATION/CONTRASTIVE] (default: INFORMATION): ").strip().upper()
                if not type_input:
                    type_input = "INFORMATION"

                res = reorderer.reorder(
                    sentence=sent_input,
                    focused_constituent=focus_input,
                    operation=op_input,
                    focus_type=type_input,
                    num_beams=args.num_beams,
                )
                print_reorder_results(res)

            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive console.")
                break
        return

    # 2. Single Sentence CLI Mode
    if args.sentence and args.focus:
        res = reorderer.reorder(
            sentence=args.sentence,
            focused_constituent=args.focus,
            operation=args.op,
            focus_type=args.type,
            num_beams=args.num_beams,
        )
        print_reorder_results(res)
        return

    # 3. Default Demo Mode
    print("\n[No sentence provided via CLI. Running standard verification examples...]")
    sample_sentence = "അച്ഛൻ ഇന്നലെ തോട്ടത്തിൽ വെച്ച് പുസ്തകം വാങ്ങി."

    print("\n--- Example 1: Information Cleft (Focus on Subject) ---")
    res1 = reorderer.reorder(sample_sentence, "അച്ഛൻ", operation="CLEFT", focus_type="INFORMATION")
    print_reorder_results(res1)

    print("\n--- Example 2: Information Fronting (Focus on Object) ---")
    res2 = reorderer.reorder(sample_sentence, "പുസ്തകം", operation="FRONTING", focus_type="INFORMATION")
    print_reorder_results(res2)

    print("\n--- Example 3: Contrastive Cleft (Focus on Subject) ---")
    res3 = reorderer.reorder(sample_sentence, "അച്ഛൻ", operation="CLEFT", focus_type="CONTRASTIVE")
    print_reorder_results(res3)

    print("\nTip: Run with --help or --interactive to test your own custom sentences:")
    print('  python bhashik_focus_reorderer.py --interactive')
    print('  python bhashik_focus_reorderer.py -s "..." -f "..." -o CLEFT')


if __name__ == "__main__":
    main()

