import re


def clean_text(text):
    """Normalize text by lowercasing, replacing punctuation/symbols with spaces,
    and collapsing whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
