"""
alignment_utils.py — Alignment, Prosody Labeling, Position Tracking, and Copula Analysis for Malayalam Clefting.

Provides comprehensive extraction of:
  1. Original sentence & Clefted sentence
  2. Focused constituent
  3. Prosody label sequence for original sentence (e.g. [1, 1, 0, 0, 0])
  4. Position before and position after movement
  5. ആണ് (aanu) attachment analysis
  6. Nominalized verb extraction
  7. Word-level and Constituent-level alignments before and after clefting
"""

import re
from typing import Dict, Any, List, Optional


_ANALYZER = None


def get_analyzer():
    global _ANALYZER
    if _ANALYZER is None:
        try:
            from mlmorph import Analyser
            _ANALYZER = Analyser()
        except Exception:
            _ANALYZER = False
    return _ANALYZER if _ANALYZER is not False else None


def strip_punctuation(word: str) -> str:
    if not isinstance(word, str):
        return ""
    return re.sub(r"[\.,!\?;:\"'“”‘’\(\)\[\]\{\}\-\–\—]", "", word).strip()


def get_word_lemmas(word: str) -> set:
    clean_w = strip_punctuation(word)
    if not clean_w:
        return set()
    lemmas = {clean_w}
    if len(clean_w) > 2:
        lemmas.add(clean_w[:-1])
    analyzer = get_analyzer()
    if analyzer:
        try:
            analyses = analyzer.analyse(clean_w)
            for raw, _ in analyses:
                m = re.match(r"^([^<]+)", raw)
                if m:
                    lemmas.add(m.group(1))
        except Exception:
            pass
    return lemmas


def is_word_match(w1: str, w2: str) -> bool:
    c1 = strip_punctuation(w1)
    c2 = strip_punctuation(w2)
    if not c1 or not c2:
        return False
    if c1 == c2:
        return True
    lemmas1 = get_word_lemmas(c1)
    lemmas2 = get_word_lemmas(c2)
    if lemmas1.intersection(lemmas2):
        return True
    min_len = min(len(c1), len(c2))
    if min_len >= 3 and c1[: min_len - 1] == c2[: min_len - 1]:
        return True
    for sfx in ("ആണ്", "ാണ്", "താണ്", "യാണ്", "നാണ്", "ാണു്", "ാണു"):
        if c1.endswith(sfx) and (c1[:-len(sfx)] == c2 or is_word_match(c1[:-len(sfx)], c2)):
            return True
        if c2.endswith(sfx) and (c2[:-len(sfx)] == c1 or is_word_match(c1, c2[:-len(sfx)])):
            return True
    return False


def extract_alignment_and_prosody(
    original_sentence: str,
    clefted_sentence: str,
    focused_constituent: str,
    main_verb: str = "",
    normalized_verb: str = "",
) -> Dict[str, Any]:
    """
    Compute full alignment, prosody label sequence, position tracking,
    copula attachment, and nominalized verb details.
    """
    orig_clean_sent = re.sub(r"</?[A-Za-z]+>", "", original_sentence).strip()
    cleft_clean_sent = re.sub(r"</?[A-Za-z]+>", "", clefted_sentence).strip()
    fc_clean_str = re.sub(r"</?[A-Za-z]+>", "", focused_constituent).strip()

    src_tokens = orig_clean_sent.split()
    clean_src = [strip_punctuation(t) for t in src_tokens]

    tgt_tokens = cleft_clean_sent.split()
    clean_tgt = [strip_punctuation(t) for t in tgt_tokens]

    fc_tokens = fc_clean_str.split()
    clean_fc = [strip_punctuation(w) for w in fc_tokens]

    # 1. Position Before (in original source sentence)
    pos_before = []
    fc_len = len(clean_fc)
    if fc_len > 0:
        # A. Exact contiguous slice match
        for i in range(len(clean_src) - fc_len + 1):
            if clean_src[i : i + fc_len] == clean_fc:
                pos_before = list(range(i, i + fc_len))
                break
        
        # B. Fallback: best contiguous window matching clean_fc
        if not pos_before:
            best_score = 0
            best_window = None
            for i in range(len(clean_src)):
                for j in range(i + 1, min(len(clean_src) + 1, i + fc_len + 4)):
                    window = clean_src[i:j]
                    score = sum(1 for w in window if any(is_word_match(w, fcw) for fcw in clean_fc))
                    length_penalty = abs(len(window) - fc_len) * 0.1
                    adj_score = score - length_penalty
                    if adj_score > best_score:
                        best_score = adj_score
                        best_window = list(range(i, j))
            if best_window and best_score > 0:
                pos_before = best_window

    source_prosody_label_sequence = [1 if i in pos_before else 0 for i in range(len(src_tokens))]

    # 2. Main Verb & Nominalized Verb
    if not main_verb:
        main_verb = src_tokens[-1] if src_tokens else ""
    clean_mv = strip_punctuation(main_verb)

    if not normalized_verb and clean_mv:
        try:
            from .verb_normalizer import VerbNormalizer
            vn = VerbNormalizer()
            norm_res = vn.normalize(clean_mv)
            if norm_res.get("status") == "VALID" and norm_res.get("normalized"):
                normalized_verb = norm_res["normalized"]
            else:
                normalized_verb = norm_res.get("normalized", clean_mv)
        except Exception:
            normalized_verb = clean_mv

    clean_norm_verb = strip_punctuation(normalized_verb)

    # 3. Position After Movement (in final target/clefted sentence)
    pos_after = []
    if fc_len > 0:
        # A. Exact / inflected contiguous slice match
        for j in range(len(clean_tgt) - fc_len + 1):
            if all(is_word_match(clean_tgt[j + k], clean_fc[k]) for k in range(fc_len)):
                pos_after = list(range(j, j + fc_len))
                break
        
        # B. Fallback: best contiguous window matching clean_fc
        if not pos_after:
            best_score = 0
            best_window = None
            for i in range(len(clean_tgt)):
                for j in range(i + 1, min(len(clean_tgt) + 1, i + fc_len + 4)):
                    window = clean_tgt[i:j]
                    score = sum(1 for w in window if any(is_word_match(w, fcw) for fcw in clean_fc))
                    length_penalty = abs(len(window) - fc_len) * 0.1
                    adj_score = score - length_penalty
                    if adj_score > best_score:
                        best_score = adj_score
                        best_window = list(range(i, j))
            if best_window and best_score > 0:
                pos_after = best_window

    # C. Fallback: match individual tokens if contiguous slice was interrupted
    if not pos_after:
        matched_indices = []
        for j, tw in enumerate(clean_tgt):
            if any(is_word_match(tw, fcw) for fcw in clean_fc):
                matched_indices.append(j)
        if matched_indices:
            pos_after = matched_indices

    # Target (Malayalam) prosody label sequence: exactly one label per token in final tgt_tokens
    target_prosody_label_sequence = [1 if j in pos_after else 0 for j in range(len(tgt_tokens))]

    # 4. ആണ് (aanu) Attachment Analysis
    aanu_info = {
        "attached_to": "none",
        "target_word": "",
        "attached_form": "",
        "position": None,
    }

    for j, tw in enumerate(clean_tgt):
        clean_token = strip_punctuation(tgt_tokens[j])
        has_copula = (
            "ആണ്" in clean_token
            or clean_token.endswith("ാണ്")
            or clean_token.endswith("താണ്")
            or clean_token.endswith("യാണ്")
            or clean_token.endswith("നാണ്")
        )
        if has_copula:
            if any(is_word_match(tw, fcw) for fcw in clean_fc):
                aanu_info = {
                    "attached_to": "focused_constituent",
                    "target_word": focused_constituent,
                    "attached_form": tgt_tokens[j],
                    "position": j,
                }
                break
            elif clean_norm_verb and (
                clean_norm_verb[:3] in tw or is_word_match(tw, clean_norm_verb)
            ):
                aanu_info = {
                    "attached_to": "nominalized_verb",
                    "target_word": normalized_verb,
                    "attached_form": tgt_tokens[j],
                    "position": j,
                }
                break
            elif clean_token == "ആണ്":
                aanu_info = {
                    "attached_to": "standalone",
                    "target_word": "ആണ്",
                    "attached_form": tgt_tokens[j],
                    "position": j,
                }
                break

    # 5. Word Alignments
    word_alignments = []
    used_tgt = set()

    mv_idx = None
    for i, sw in enumerate(clean_src):
        if is_word_match(sw, clean_mv):
            mv_idx = i
            break
    if mv_idx is None and src_tokens:
        mv_idx = len(src_tokens) - 1

    for i, s_tok in enumerate(src_tokens):
        s_clean = clean_src[i]

        if i in pos_before:
            t_idx = pos_after[0] if pos_after else None
            t_tok = tgt_tokens[t_idx] if t_idx is not None and t_idx < len(tgt_tokens) else ""
            word_alignments.append(
                {
                    "src_index": i,
                    "src_word": s_tok,
                    "tgt_index": t_idx,
                    "tgt_word": t_tok,
                    "alignment_type": "FOCUSED_CONSTITUENT_MOVEMENT",
                }
            )
            if t_idx is not None:
                used_tgt.add(t_idx)

        elif i == mv_idx or is_word_match(s_clean, clean_mv):
            t_idx = None
            for j, t_tok in enumerate(tgt_tokens):
                t_clean = clean_tgt[j]
                if (
                    is_word_match(t_clean, clean_norm_verb)
                    or clean_norm_verb[:3] in t_clean
                    or "താണ്" in t_tok
                ):
                    t_idx = j
                    break
            t_tok = tgt_tokens[t_idx] if t_idx is not None else ""
            word_alignments.append(
                {
                    "src_index": i,
                    "src_word": s_tok,
                    "tgt_index": t_idx,
                    "tgt_word": t_tok,
                    "alignment_type": "VERB_NOMINALIZATION",
                }
            )
            if t_idx is not None:
                used_tgt.add(t_idx)

        else:
            t_idx = None
            for j, t_tok in enumerate(tgt_tokens):
                t_clean = clean_tgt[j]
                if j not in used_tgt and is_word_match(s_clean, t_clean):
                    t_idx = j
                    break
            t_tok = tgt_tokens[t_idx] if t_idx is not None else ""
            align_type = (
                "EXACT_MATCH"
                if (t_idx is not None and s_clean == clean_tgt[t_idx])
                else ("INFLECTED_MATCH" if t_idx is not None else "UNMATCHED")
            )
            word_alignments.append(
                {
                    "src_index": i,
                    "src_word": s_tok,
                    "tgt_index": t_idx,
                    "tgt_word": t_tok,
                    "alignment_type": align_type,
                }
            )
            if t_idx is not None:
                used_tgt.add(t_idx)

    # 6. Constituent Alignments
    constituent_alignments = []
    if pos_before and pos_after:
        constituent_alignments.append(
            {
                "constituent": focused_constituent,
                "role": "FOCUSED_CONSTITUENT",
                "src_position": pos_before,
                "tgt_position": pos_after,
                "src_form": " ".join([src_tokens[i] for i in pos_before]),
                "tgt_form": " ".join([tgt_tokens[j] for j in pos_after]),
            }
        )

    if mv_idx is not None:
        verb_tgt_idx = [
            wa["tgt_index"]
            for wa in word_alignments
            if wa["src_index"] == mv_idx and wa["tgt_index"] is not None
        ]
        constituent_alignments.append(
            {
                "constituent": src_tokens[mv_idx],
                "role": "NOMINALIZED_VERB",
                "src_position": [mv_idx],
                "tgt_position": verb_tgt_idx,
                "src_form": src_tokens[mv_idx],
                "tgt_form": " ".join([tgt_tokens[j] for j in verb_tgt_idx])
                if verb_tgt_idx
                else normalized_verb,
            }
        )

    return {
        "original_sentence": orig_clean_sent,
        "clefted_sentence": cleft_clean_sent,
        "focused_constituent": fc_clean_str,
        "prosody_label_sequence": target_prosody_label_sequence,
        "target_prosody_label_sequence": target_prosody_label_sequence,
        "source_prosody_label_sequence": source_prosody_label_sequence,
        "position_before": pos_before,
        "position_after": pos_after,
        "aanu_attachment": aanu_info,
        "nominalized_verb": normalized_verb,
        "word_alignments": word_alignments,
        "constituent_alignments": constituent_alignments,
    }
