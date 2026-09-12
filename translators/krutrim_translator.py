"""
krutrim_translator.py — Standalone Neural MT module for Krutrim Translate (IndicTrans2-based).

Translates English to Malayalam using CTranslate2 with GPU prioritized and IndicNLP transliteration.
"""

import os
import sys
import glob
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _setup_cuda_dlls():
    """Add NVIDIA CUDA DLL directories to Windows DLL search path."""
    if sys.platform == "win32":
        site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
        nvidia_dir = os.path.join(site_packages, "nvidia")
        if os.path.exists(nvidia_dir):
            bin_dirs = glob.glob(os.path.join(nvidia_dir, "*", "bin"))
            lib_dirs = glob.glob(os.path.join(nvidia_dir, "*", "lib"))
            for p in bin_dirs + lib_dirs:
                if os.path.exists(p):
                    try:
                        os.add_dll_directory(p)
                    except Exception:
                        pass
            if bin_dirs:
                os.environ["PATH"] = ";".join(bin_dirs) + ";" + os.environ.get("PATH", "")


_setup_cuda_dlls()


class KrutrimTranslator:
    """
    Krutrim Translate (IndicTrans2) English-to-Malayalam translator using CTranslate2.
    Prioritizes CUDA GPU acceleration.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        beam_size: int = 3,
        lazy_load: bool = True,
    ):
        self.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. Locate Model Directory
        candidates = [
            model_dir,
            os.path.join(self.repo_root, "test", "KrutrimTranslate", "static_data", "ct_model_english_indic"),
            os.path.join(self.repo_root, "models", "KrutrimTranslate", "static_data", "ct_model_english_indic"),
        ]
        self.model_dir = None
        for cand in candidates:
            if cand and os.path.exists(os.path.join(cand, "model.bin")):
                self.model_dir = cand
                break

        self.beam_size = beam_size
        self.requested_device = device
        self._loaded = False
        self.translator = None
        self.sp_src = None
        self.sp_tgt = None
        self.device = "cpu"

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        if self._loaded:
            return

        if not self.model_dir or not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Krutrim model directory not found. Candidates checked: {self.model_dir}")

        import ctranslate2
        import sentencepiece as spm

        _setup_cuda_dlls()

        # 1. Load SentencePiece Processors
        sp_src_path = os.path.join(self.model_dir, "vocab", "model.SRC")
        sp_tgt_path = os.path.join(self.model_dir, "vocab", "model.TGT")
        
        if not os.path.exists(sp_src_path) or not os.path.exists(sp_tgt_path):
            raise FileNotFoundError(f"SentencePiece vocabulary models missing in {os.path.join(self.model_dir, 'vocab')}")

        self.sp_src = spm.SentencePieceProcessor(model_file=sp_src_path)
        self.sp_tgt = spm.SentencePieceProcessor(model_file=sp_tgt_path)

        # 2. Select Device (GPU Priority)
        target_device = self.requested_device
        if not target_device:
            target_device = "cuda" if (ctranslate2.get_cuda_device_count() > 0) else "cpu"

        # 3. Initialize CTranslate2 Translator with GPU fallback
        try:
            if target_device == "cuda":
                self.translator = ctranslate2.Translator(
                    self.model_dir,
                    device="cuda",
                    device_index=0,
                    compute_type="float16" if ctranslate2.get_cuda_device_count() > 0 else "default",
                )
                self.device = "cuda"
            else:
                self.translator = ctranslate2.Translator(self.model_dir, device="cpu")
                self.device = "cpu"
        except Exception as e:
            logger.warning(f"KrutrimTranslator CUDA init failed ({e}), falling back to CPU.")
            self.translator = ctranslate2.Translator(self.model_dir, device="cpu")
            self.device = "cpu"

        self._loaded = True
        logger.info(f"KrutrimTranslator ready on {self.device}.")

    def _postprocess_malayalam(self, devanagari_text: str) -> str:
        """Convert IndicTrans2 Devanagari representation to Malayalam script."""
        try:
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            mal = UnicodeIndicTransliterator.transliterate(devanagari_text, "hi", "ml")
        except Exception:
            mal = devanagari_text

        # Clean punctuation spacing
        import re
        mal = re.sub(r'\s+([.,!?;:])', r'\1', mal)
        mal = re.sub(r'\s+', ' ', mal).strip()
        return mal

    def translate(
        self,
        text: str,
        src_lang: str = "eng_Latn",
        tgt_lang: str = "mal_Mlym",
    ) -> str:
        """Translate a single sentence from English to Malayalam."""
        self._load_model()
        clean_text = text.strip()
        if not clean_text:
            return ""

        tokens = [src_lang, tgt_lang] + self.sp_src.encode(clean_text, out_type=str) + ["</s>"]
        res = self.translator.translate_batch([tokens], beam_size=self.beam_size)
        decoded = self.sp_tgt.decode(res[0].hypotheses[0])
        return self._postprocess_malayalam(decoded)

    def batch_translate(
        self,
        texts: List[str],
        src_lang: str = "eng_Latn",
        tgt_lang: str = "mal_Mlym",
        batch_size: int = 16,
    ) -> List[str]:
        """Translate a batch of sentences from English to Malayalam."""
        self._load_model()
        if not texts:
            return []

        all_results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            token_batches = []
            for t in chunk:
                clean_t = t.strip()
                if clean_t:
                    toks = [src_lang, tgt_lang] + self.sp_src.encode(clean_t, out_type=str) + ["</s>"]
                else:
                    toks = []
                token_batches.append(toks)

            non_empty_indices = [idx for idx, toks in enumerate(token_batches) if toks]
            non_empty_tokens = [token_batches[idx] for idx in non_empty_indices]

            chunk_outputs = [""] * len(chunk)
            if non_empty_tokens:
                res = self.translator.translate_batch(non_empty_tokens, beam_size=self.beam_size)
                for orig_idx, translation_result in zip(non_empty_indices, res):
                    decoded = self.sp_tgt.decode(translation_result.hypotheses[0])
                    chunk_outputs[orig_idx] = self._postprocess_malayalam(decoded)

            all_results.extend(chunk_outputs)

        return all_results

    def translate_batch(self, texts: List[str], **kwargs) -> List[str]:
        """Alias for batch_translate."""
        return self.batch_translate(texts, **kwargs)
