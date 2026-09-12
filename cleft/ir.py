import json

class Token:
    def __init__(self, id, form, lemma, form_pos, lemma_pos, feats, normalized_form=None, is_main_verb=False):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.form_pos = form_pos
        self.lemma_pos = lemma_pos
        self.features = feats
        self.normalized_form = normalized_form
        self.is_main_verb = is_main_verb

    def to_dict(self):
        d = {
            "id": self.id,
            "form": self.form,
            "lemma": self.lemma,
            "form_pos": self.form_pos,
            "lemma_pos": self.lemma_pos,
            "features": self.features
        }
        if self.is_main_verb:
            d["is_main_verb"] = self.is_main_verb
            d["normalized_form"] = self.normalized_form
        return d

class Sentence:
    def __init__(self, text, tokens=None, main_verb=None, normalized_verb=None):
        self.text = text
        self.tokens = tokens or []
        self.main_verb = main_verb
        self.normalized_verb = normalized_verb
        
    def to_dict(self):
        d = {
            "sentence": self.text,
            "tokens": [t.to_dict() for t in self.tokens]
        }
        if self.main_verb is not None:
            d["main_verb"] = self.main_verb
        if self.normalized_verb is not None:
            d["normalized_verb"] = self.normalized_verb
        return d
        
    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
