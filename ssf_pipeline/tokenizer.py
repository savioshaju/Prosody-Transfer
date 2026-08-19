import re

def tokenize_malayalam(text):
    
    tokens = re.findall(r"[\w'\u0D00-\u0D7F]+|[.,!?;]", text)
    return [t for t in tokens if t.strip()]
