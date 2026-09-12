"""
Bhashaverse Machine Translation Module for SSF Pipeline.

Provides an offline-first GPU-accelerated wrapper around the LTRC IIIT-Hyderabad
Bhashaverse translation model (OneNMT v3b) for English-to-Malayalam.
"""

import os
import sys
import json
import logging
from typing import List, Optional
import torch

logger = logging.getLogger(__name__)

# FLORES-200 Language Codes
LANGUAGE_MAPPING = {
    "eng": "eng_Latn",
    "mal": "mal_Mlym",
}


class BhashaverseTranslator:
    """
    Offline-first Neural Machine Translation using Bhashaverse (IIIT-H).
    Translates English to Malayalam using OneNMT v3b.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        beam_size: int = 5,
        max_new_tokens: int = 256,
        lazy_load: bool = True,
    ):
        # 1. Device selection — prioritize GPU
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Local model assets and cached weights directory
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.asset_dir = model_dir if (model_dir and os.path.exists(model_dir)) else os.path.join(repo_root, "test", "bhashaverse")
        self.hf_snapshot_dir = r"D:\hf_cache\hub\models--ltrciiith--bhashaverse\snapshots\02b536f29b9ba211f5d3f25485679bffb23abca3"
        self.weight_path = os.path.join(self.hf_snapshot_dir, "model.safetensors")

        self.beam_size = beam_size
        self.max_new_tokens = max_new_tokens

        self._loaded = False
        self.model = None
        self.sp = None
        self.src_sym2id = {}
        self.tgt_id2sym = {}
        self.eos_id = 2
        self.pad_id = 1
        self.bos_id = 0
        self.unk_id = 3

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load SentencePiece tokenizer, fairseq dictionary, and mBART weights on GPU."""
        if self._loaded:
            return

        if not os.path.exists(self.asset_dir):
            raise FileNotFoundError(f"Bhashaverse asset directory not found at: {self.asset_dir}")

        import sentencepiece as spm
        from transformers import MBartForConditionalGeneration

        # 1. Load SentencePiece Processor
        spm_path = os.path.join(self.asset_dir, "onemtv3b_spm.model")
        self.sp = spm.SentencePieceProcessor(model_file=spm_path)

        # 2. Load Fairseq Dictionary
        dict_path = os.path.join(self.asset_dir, "fairseq_dict.json")
        with open(dict_path, "r", encoding="utf-8") as f:
            fs_dict = json.load(f)

        self.src_sym2id = fs_dict["src"]
        self.tgt_id2sym = {v: k for k, v in fs_dict["tgt"].items()}
        sp_specials = fs_dict.get("special", {})
        self.eos_id = sp_specials.get("eos", 2)
        self.pad_id = sp_specials.get("pad", 1)
        self.bos_id = sp_specials.get("bos", 0)
        self.unk_id = sp_specials.get("unk", 3)

        # 3. Load Model from local HuggingFace cache snapshot
        model_load_dir = self.hf_snapshot_dir if os.path.exists(self.hf_snapshot_dir) else self.asset_dir
        self.model = MBartForConditionalGeneration.from_pretrained(model_load_dir, local_files_only=True)

        # 4. Move to GPU / CPU
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(dtype=torch.float32, device=self.device)
            else:
                self.device = torch.device("cpu")
                self.model = self.model.to(device=self.device)
        except Exception:
            self.device = torch.device("cpu")
            self.model = self.model.to(device=self.device)

        self.model.eval()
        self._loaded = True
        logger.info(f"BhashaverseTranslator ready on {self.device}.")

    def _encode(self, text: str, src_flores: str, tgt_flores: str) -> List[int]:
        """Encode text with task prefix token into dictionary IDs."""
        tagged = f"###{src_flores}-to-{tgt_flores}### {text}"
        pieces = self.sp.encode(tagged, out_type=str)
        return [self.src_sym2id.get(p, self.unk_id) for p in pieces] + [self.eos_id]

    def _decode(self, token_ids: List[int]) -> str:
        """Decode output IDs back to clean target string."""
        clean = [t for t in token_ids if t not in (self.bos_id, self.pad_id, self.eos_id)]
        pieces = [self.tgt_id2sym.get(t, "<unk>") for t in clean]
        return self.sp.decode(pieces).strip()

    @torch.inference_mode()
    def _translate_batch_encoded(self, encoded_batch: List[List[int]]) -> List[str]:
        """Run batch inference on encoded token sequences."""
        if not encoded_batch:
            return []

        max_len = max(len(ids) for ids in encoded_batch)
        input_ids = torch.full(
            (len(encoded_batch), max_len), self.pad_id, dtype=torch.long, device=self.device
        )
        attn_mask = torch.zeros_like(input_ids)

        for i, ids in enumerate(encoded_batch):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attn_mask[i, : len(ids)] = 1

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_start_token_id=self.eos_id,
            num_beams=self.beam_size,
            max_new_tokens=self.max_new_tokens,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.3,
        )

        return [self._decode(out.tolist()) for out in generated]

    def translate(
        self,
        text: str,
        src_lang: str = "eng",
        tgt_lang: str = "mal",
    ) -> str:
        """
        Translate a single sentence from English to Malayalam.
        """
        self._load_model()
        clean_text = text.strip()
        if not clean_text:
            return ""

        src_flores = LANGUAGE_MAPPING.get(src_lang, src_lang)
        tgt_flores = LANGUAGE_MAPPING.get(tgt_lang, tgt_lang)

        encoded = [self._encode(clean_text, src_flores, tgt_flores)]
        results = self._translate_batch_encoded(encoded)
        return results[0] if results else ""

    def translate_batch(
        self,
        texts: List[str],
        src_lang: str = "eng",
        tgt_lang: str = "mal",
        batch_size: int = 8,
    ) -> List[str]:
        """
        Translate a list of sentences in batches.
        """
        self._load_model()
        if not texts:
            return []

        src_flores = LANGUAGE_MAPPING.get(src_lang, src_lang)
        tgt_flores = LANGUAGE_MAPPING.get(tgt_lang, tgt_lang)

        all_results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encoded = [self._encode(t.strip(), src_flores, tgt_flores) for t in chunk]
            preds = self._translate_batch_encoded(encoded)
            all_results.extend(preds)

        return all_results
