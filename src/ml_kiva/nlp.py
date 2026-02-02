# src/ml_kiva/nlp.py
from functools import lru_cache
import spacy


@lru_cache(maxsize=1)
def get_nlp():
    """Lazy-load and cache spaCy model (CI-safe)."""
    return spacy.load("en_core_web_sm", disable=["parser", "ner"])
