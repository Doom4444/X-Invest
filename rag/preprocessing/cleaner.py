import re

def clean_text(text: str) -> str:
    """
    Clean text before chunking.
    """

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text