import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import string

class BertPosTagger:
    def __init__(self, model_name="akhisreelibra/bert-malayalam-pos-tagger"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _is_punctuation(self, token):
        # Returns True if the token consists entirely of punctuation characters
        extra_punctuation = "“”’‘—–"
        return all(c in string.punctuation or c in extra_punctuation for c in token)

    def get_pos_tags(self, tokens):
        """
        Takes a list of canonical tokens (words/punctuation).
        Returns a list of POS tags exactly matching the length of the input tokens.
        """
        if not tokens:
            return []

        # We tell the tokenizer that the input is already split into words.
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
        inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs_on_device)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        
        # predictions[0] corresponds to the sequence in the batch
        pred_labels = predictions[0].tolist()
        
        # word_ids() returns a list mapping each subword to its original word index
        word_ids = inputs.word_ids(batch_index=0)
        
        id2label = self.model.config.id2label
        
        # Align predictions to original tokens
        final_tags = [None] * len(tokens)
        for word_idx, label_id in zip(word_ids, pred_labels):
            if word_idx is None:
                continue
            if word_idx < len(final_tags) and final_tags[word_idx] is None:
                final_tags[word_idx] = id2label[label_id]

        # Fill any missing and override punctuation
        for i, token in enumerate(tokens):
            if final_tags[i] is None:
                final_tags[i] = "RD_UNK"
            if self._is_punctuation(token):
                final_tags[i] = "RD_PUNC"
            
        return final_tags

