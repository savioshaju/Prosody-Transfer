import logging
import string
from mlmorph import Analyser

logger = logging.getLogger(__name__)

class MlmorphAnalyzer:
    def __init__(self):
        self.analyzer = Analyser()

    def _is_punctuation(self, token):
        extra_punctuation = "“”’‘—–"
        return all(c in string.punctuation or c in extra_punctuation for c in token)
        
    def analyze_token(self, form):
        
        if self._is_punctuation(form):
            return {'lemma': form, 'feats': ''}

        analyses = self.analyzer.analyse(form)
        if not analyses:
            # logger.warning("mlmorph: no analysis for surface form '%s'", form)
            return {'lemma': form, 'feats': ''}
            
        # Get the first (most probable) analysis
        # Format is typically 'lemma<tag1><tag2>...'
        best_analysis = analyses[0][0]
        
        if '<' not in best_analysis:
            return {'lemma': best_analysis, 'feats': ''}
            
        parts = best_analysis.split('<', 1)
        lemma = parts[0]
        tags_str = '<' + parts[1]
        
        # Extract features (skip first tag which is the surface POS)
        tags = [t for t in tags_str.replace('>', '').split('<') if t]
        feats = '|'.join(tags[1:]) if len(tags) > 1 else ''
        
        return {'lemma': lemma, 'feats': feats}

    def get_lemma_pos(self, lemma):
        """
        Analyze the lemma itself to get its true POS.
        This runs mlmorph on the lemma form (e.g. 'വരുക') rather than
        the surface form (e.g. 'വന്നു'), so the POS reflects the
        dictionary entry of the word.
        """
        if self._is_punctuation(lemma):
            return 'RD_PUNC'

        analyses = self.analyzer.analyse(lemma)
        if not analyses:
            logger.warning("mlmorph: no analysis for lemma '%s'", lemma)
            return 'UNK'
            
        best_analysis = analyses[0][0]
        
        if '<' not in best_analysis:
            return 'UNK'
            
        parts = best_analysis.split('<', 1)
        tags_str = '<' + parts[1]
        tags = [t for t in tags_str.replace('>', '').split('<') if t]
        
        return tags[0] if tags else 'UNK'


