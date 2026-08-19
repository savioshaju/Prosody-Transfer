from .ir import Token, Sentence
from .tokenizer import tokenize_malayalam
from .mlmorph_utils import MlmorphAnalyzer
from .bert_pos_utils import BertPosTagger
from .verb_normalizer import VerbNormalizer

class MalayalamPipeline:
    def __init__(self):
        self.analyzer = MlmorphAnalyzer()
        self.bert_tagger = BertPosTagger()
        self.verb_normalizer = VerbNormalizer()

    def _score_verb_candidate(self, idx, form, form_pos, lemma_pos, analysis):
        score = 0
        feats = analysis.get('feats', '')
        
        # 1. Base POS tag scoring from BERT
        if form_pos.endswith("_VF"):
            score += 100
        elif form_pos.startswith("V_VM") or form_pos == "V_VAUX":
            score += 50
        elif form_pos.endswith("_VNF") or form_pos.endswith("_VINF"):
            score -= 50
            
        # 2. mlmorph morphological feature scoring
        feat_tags = set(feats.split('|')) if feats else set()
        non_finite_markers = {
            'cvb-adv-part-past', 'cvb-adv-part-simul', 'cvb-adv-part-future',
            'adv-clause-rp-past', 'adv-clause-rp-present', 'infinitive', 'purposive-mood'
        }
        finite_markers = {
            'past', 'present', 'future', 'imperative-mood', 'promissive-mood',
            'permissive-mood', 'conditional-mood', 'habitual-aspect', 'aff'
        }
        
        if feat_tags & non_finite_markers:
            score -= 40
        if feat_tags & finite_markers:
            score += 40
            
        # 3. Position bias (rightmost matrix finite verb in clause structure)
        score += idx
        
        return score

    def identify_main_verb_index(self, tokens, form_pos_tags, lemma_pos_tags, analyses):
        candidates = []
        for i, (form, form_pos, analysis, lemma_pos) in enumerate(
            zip(tokens, form_pos_tags, analyses, lemma_pos_tags)
        ):
            feats = analysis.get('feats', '')
            has_verb_tag = (
                form_pos.startswith("V_") or
                lemma_pos.startswith("V_") or
                any(t in feats for t in ('past', 'present', 'future', 'mood', 'aspect', 'aff'))
            )
            if has_verb_tag:
                s = self._score_verb_candidate(i, form, form_pos, lemma_pos, analysis)
                candidates.append((i, s))
                
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
        
    def process(self, text):
        # 1. Canonical Tokenization
        tokens = tokenize_malayalam(text)
        if not tokens:
            return Sentence(text=text, tokens=[])
        
        # 2. Form POS — BERT on surface tokens
        form_pos_tags = self.bert_tagger.get_pos_tags(tokens)
        
        # 3. Lemma + Features — mlmorph on surface tokens
        analyses = [self.analyzer.analyze_token(form) for form in tokens]
        lemmas = [a['lemma'] for a in analyses]
        
        # 4. Lemma POS — BERT on lemmas
        lemma_pos_tags = self.bert_tagger.get_pos_tags(lemmas)
        
        # 5. Identify Main Finite Verb
        main_verb_idx = self.identify_main_verb_index(tokens, form_pos_tags, lemma_pos_tags, analyses)
        main_verb_form = None
        normalized_verb_form = None
        
        if main_verb_idx is not None:
            main_verb_form = tokens[main_verb_idx]
            # Send ONLY the identified main verb to VerbNormalizer
            norm_res = self.verb_normalizer.normalize(main_verb_form)
            if norm_res.get("status") == "VALID" and norm_res.get("normalized"):
                normalized_verb_form = norm_res["normalized"]
            else:
                # If UNRESOLVED, preserve original verb without crashing
                normalized_verb_form = main_verb_form

        # 6. Build IR Tokens
        ir_tokens = []
        for i, (form, form_pos, analysis, lemma_pos) in enumerate(
            zip(tokens, form_pos_tags, analyses, lemma_pos_tags), 1
        ):
            is_mv = (i - 1 == main_verb_idx) if main_verb_idx is not None else False
            norm_form = normalized_verb_form if is_mv else None
            token = Token(
                id=i,
                form=form,
                lemma=analysis['lemma'],
                form_pos=form_pos,
                lemma_pos=lemma_pos,
                feats=analysis['feats'],
                normalized_form=norm_form,
                is_main_verb=is_mv
            )
            ir_tokens.append(token)
            
        return Sentence(
            text=text,
            tokens=ir_tokens,
            main_verb=main_verb_form,
            normalized_verb=normalized_verb_form
        )
