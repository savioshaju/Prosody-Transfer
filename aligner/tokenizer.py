import re

def tokenize_malayalam(text):
    if not text:
        return []
    pattern = r"[\w\u0D00-\u0D7F\u200C\u200D]+(?:[.-][\w\u0D00-\u0D7F\u200C\u200D]+)*|[.,!?;:()\[\]\"'“”‘’]"
    tokens = re.findall(pattern, text)
    return [t for t in tokens if t.strip()]
