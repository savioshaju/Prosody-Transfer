"""
syntax_guided_aligner.py — Phrase & Clause-Level Syntactic Alignment for English-Malayalam.

Implements two complementary syntactic alignment modalities:
  1. StandalonePhraseAligner: Direct phrase-level & clause-level matching using
     mBERT contextualized phrase embeddings and syntactic category compatibility.
  2. SyntaxGuidedSimAligner (Hybrid): Combines fine-grained SimAlign token similarity
     matrices with English and Malayalam phrase chunkers (NP, PP, AdvP, VP) and
     clause boundaries, enforcing Dravidian island constraints (Jayaseelan 2001).
"""

import os
import re
import sys
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn.functional as F

from aligner.alignment_utils import strip_punctuation
from aligner.english_phrase_chunker import EnglishPhraseChunker, EnglishChunk
from cleft.phrase_chunker import MalayalamPhraseChunker, PhraseChunk
from aligner.simalign_wrapper import SimAlignerWrapper


class StandalonePhraseAligner:
    """
    Modality A: Standalone Phrase & Clause Aligner.
    Matches English focus constituent to Malayalam syntactic chunks using
    mBERT contextual phrase embeddings within matching clausal scopes.
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased", shared_model=None, shared_tokenizer=None, device: Optional[str] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        self.en_chunker = EnglishPhraseChunker()
        self.ml_chunker = MalayalamPhraseChunker()
        self.model = shared_model
        self.tokenizer = shared_tokenizer
        self._loaded = bool(shared_model is not None and shared_tokenizer is not None)

    def _load_model(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
            except Exception:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
        except Exception as e:
            print(f">>> [StandalonePhraseAligner] Model load warning: {e}")
            self._loaded = False

    def _encode_sentence(self, sentence: str):
        """Encode sentence once and return tokenized inputs and layer 8 hidden states."""
        self._load_model()
        if not self._loaded or not self.model or not self.tokenizer:
            return None, None
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[8]
        return inputs, hidden

    def _extract_span_vec(self, inputs, hidden, phrase_text: str) -> Optional[torch.Tensor]:
        """Slice and mean-pool phrase representation from precomputed sentence hidden states."""
        p_clean = strip_punctuation(phrase_text).strip()
        if not p_clean or inputs is None or hidden is None:
            return None

        p_inputs = self.tokenizer(p_clean, add_special_tokens=False)
        p_ids = p_inputs["input_ids"]
        sent_ids = inputs["input_ids"][0].tolist()

        # Try contiguous subword sequence
        for i in range(len(sent_ids) - len(p_ids) + 1):
            if sent_ids[i : i + len(p_ids)] == p_ids:
                return hidden[0, i : i + len(p_ids), :].mean(dim=0, keepdim=True)

        # Fallback to token id set overlap
        p_set = set(p_ids)
        match_idx = [i for i, tid in enumerate(sent_ids) if tid in p_set]
        if match_idx:
            return hidden[0, match_idx, :].mean(dim=0, keepdim=True)

        return hidden[:, 0, :]

    def align_focus(
        self,
        en_sentence: str,
        ml_sentence: str,
        en_focus: str,
    ) -> Dict[str, Any]:
        """
        Align English focus expression to the best Malayalam syntactic chunk.
        """
        en_chunk = self.en_chunker.find_focus_phrase(en_sentence, en_focus)
        en_text = en_chunk.text if en_chunk else en_focus
        en_type = en_chunk.chunk_type if en_chunk else "NP"

        # Tokenize Malayalam consistently
        tokens = re.findall(r"[\w\u0D00-\u0D7F\u200C\u200D]+(?:[.-][\w\u0D00-\u0D7F\u200C\u200D]+)*|[.,!?;:()\[\]\"'“”‘’]", ml_sentence)
        ml_tokens = [t for t in tokens if t.strip()]

        ml_chunks = self.ml_chunker.chunk_sentence(ml_tokens, sentence_text=ml_sentence)
        if not ml_chunks:
            return {
                "selected_constituent": ml_tokens[0] if ml_tokens else "",
                "start_idx": 0,
                "end_idx": 0,
                "confidence": "LOW_FALLBACK"
            }

        # Single forward pass for English and single forward pass for Malayalam
        en_inputs, en_hidden = self._encode_sentence(en_sentence)
        en_vec = self._extract_span_vec(en_inputs, en_hidden, en_text)
        if en_vec is None:
            return {
                "selected_constituent": ml_chunks[0].text,
                "start_idx": ml_chunks[0].start_idx,
                "end_idx": ml_chunks[0].end_idx,
                "confidence": "LEXICAL_FALLBACK"
            }

        ml_inputs, ml_hidden = self._encode_sentence(ml_sentence)

        best_score = -1.0
        best_chunk = ml_chunks[0]

        for chunk in ml_chunks:
            ml_vec = self._extract_span_vec(ml_inputs, ml_hidden, chunk.text)
            if ml_vec is None:
                continue
            sim = F.cosine_similarity(en_vec, ml_vec).item()
            if en_type == chunk.chunk_type:
                sim += 0.05

            if sim > best_score:
                best_score = sim
                best_chunk = chunk

        return {
            "selected_constituent": best_chunk.text,
            "start_idx": best_chunk.start_idx,
            "end_idx": best_chunk.end_idx,
            "score": best_score,
            "chunk_type": best_chunk.chunk_type,
            "confidence": "PHRASE_STANDALONE",
            "target_tokens": ml_tokens
        }


class SyntaxGuidedSimAligner:
    """
    Modality C: Syntax-Guided / Chunk-Constrained SimAligner (Hybrid).
    Uses SimAlign bipartite matching weights aggregated over candidate
    syntactic chunks (NP, PP, AdvP) within clausal domains.
    """

    def __init__(
        self,
        simalign_instance: Optional[SimAlignerWrapper] = None,
        model_name: str = "bert-base-multilingual-cased",
        matching_method: str = "itermax",
        device: Optional[str] = None
    ):
        if simalign_instance is not None:
            self.simalign = simalign_instance
        else:
            self.simalign = SimAlignerWrapper(model_name=model_name, matching_method=matching_method, device=device)
        self.en_chunker = EnglishPhraseChunker()
        self.ml_chunker = MalayalamPhraseChunker()

    def align_and_project_focus(
        self,
        en_sentence: str,
        ml_sentence: str,
        en_focus: str,
    ) -> Dict[str, Any]:
        """
        Project English focus to Malayalam constituent using Syntax-Guided SimAlign:
        1. Identifies English focus maximal phrase XP_EN via EnglishPhraseChunker.
        2. Computes neural SimAlign token correspondences.
        3. Identifies Malayalam candidate syntactic chunks XP_ML via MalayalamPhraseChunker.
        4. Constrains the selection to the syntactic chunk that maximizes aligned token mass.
        """
        # 1. Parse English focus constituent
        en_chunk = self.en_chunker.find_focus_phrase(en_sentence, en_focus)
        en_focus_text = en_chunk.text if en_chunk else en_focus
        en_focus_type = en_chunk.chunk_type if en_chunk else "NP"

        # 2. Run SimAlign
        pre_alignment = self.simalign.align(en_sentence, ml_sentence)
        src_tokens = pre_alignment["source_tokens"]
        tgt_tokens = pre_alignment["target_tokens"]
        reconciled = pre_alignment.get("reconciled_pairs", [])

        # Find English focus indices in src_tokens
        en_words = [strip_punctuation(w).lower() for w in self.simalign._tokenize(en_focus_text) if strip_punctuation(w)]
        src_clean = [strip_punctuation(w).lower() for w in src_tokens]
        flen = len(en_words)

        focus_src_indices = []
        for i in range(len(src_clean) - flen + 1):
            if src_clean[i : i + flen] == en_words:
                focus_src_indices = list(range(i, i + flen))
                break

        if not focus_src_indices:
            for i, sw in enumerate(src_clean):
                if any(ew in sw or sw in ew for ew in en_words if ew):
                    focus_src_indices.append(i)

        # Fallback to original en_focus if en_chunk words did not match
        if not focus_src_indices:
            orig_words = [strip_punctuation(w).lower() for w in self.simalign._tokenize(en_focus) if strip_punctuation(w)]
            olen = len(orig_words)
            for i in range(len(src_clean) - olen + 1):
                if src_clean[i : i + olen] == orig_words:
                    focus_src_indices = list(range(i, i + olen))
                    break

        # 3. Collect aligned target token indices
        target_indices = [
            p["tgt_index"]
            for p in reconciled
            if p.get("src_index") in focus_src_indices and p.get("is_lexical", True)
        ]

        # 4. Chunk Malayalam sentence into maximal syntactic projections
        ml_chunks = self.ml_chunker.chunk_sentence(tgt_tokens, sentence_text=ml_sentence)

        if not target_indices:
            # Fallback to relative positional approximation
            avg_src = sum(focus_src_indices) / len(focus_src_indices) if focus_src_indices else 0
            approx_t = int(round(avg_src * (len(tgt_tokens) - 1) / max(len(src_tokens) - 1, 1)))
            target_indices = [max(0, min(len(tgt_tokens) - 1, approx_t))]

        # 5. Score candidate Malayalam chunks by alignment mass & syntactic type
        best_chunk = None
        best_score = -1.0
        target_set = set(target_indices)

        for chunk in ml_chunks:
            chunk_tokens = set(range(chunk.start_idx, chunk.end_idx + 1))
            overlap_count = len(chunk_tokens.intersection(target_set))
            if overlap_count == 0:
                continue

            # Base score: number of aligned tokens captured
            score = float(overlap_count)

            # Precision bonus: penalize excessively large chunks that capture non-focus tokens
            chunk_len = (chunk.end_idx - chunk.start_idx + 1)
            score += (overlap_count / max(chunk_len, 1)) * 0.5

            # Syntactic type compatibility bonus
            if en_focus_type == chunk.chunk_type:
                score += 1.0
            elif en_focus_type == "PP" and chunk.chunk_type == "NP":
                # If Malayalam expresses English PP as suffixed NP or case-marked noun
                score += 0.3

            if score > best_score:
                best_score = score
                best_chunk = chunk

        # If no chunk had overlap, expand enclosing phrase
        if best_chunk is None:
            min_t, max_t = min(target_indices), max(target_indices)
            enclosing = self.ml_chunker.find_enclosing_phrase(tgt_tokens, min_t, max_t, ml_sentence)
            if enclosing:
                best_chunk = enclosing
            else:
                sel_tokens = [tgt_tokens[i] for i in range(min_t, max_t + 1) if strip_punctuation(tgt_tokens[i]).strip()]
                return {
                    "selected_constituent": " ".join(sel_tokens),
                    "start_idx": min_t,
                    "end_idx": max_t,
                    "chunk_type": "UNKNOWN",
                    "score": 0.0,
                    "confidence": "RAW_SIMALIGN_SPAN",
                    "target_tokens": tgt_tokens,
                    "pre_alignment": pre_alignment
                }

        sel_tokens = [tgt_tokens[i] for i in range(best_chunk.start_idx, best_chunk.end_idx + 1) if strip_punctuation(tgt_tokens[i]).strip()]
        selected_text = " ".join(sel_tokens)

        return {
            "selected_constituent": selected_text,
            "start_idx": best_chunk.start_idx,
            "end_idx": best_chunk.end_idx,
            "chunk_type": best_chunk.chunk_type,
            "score": best_score,
            "confidence": "SYNTAX_GUIDED_HYBRID",
            "target_tokens": tgt_tokens,
            "pre_alignment": pre_alignment
        }
