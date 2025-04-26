
import re
from .numbers import normalize_numbers
from unidecode import unidecode

_whitespace_re = re.compile(r'\s+')

def mongolian_cleaners(text):
    """Pipeline for Mongolian text cleaning."""
    text = text.lower() # Or upper, be consistent
    # TODO: Implement Mongolian abbreviation expansion
    # text = _expand_abbreviations(text)
    text = normalize_numbers(text) # Implement this!
    # TODO: Implement specific punctuation handling
    # Keep basic punctuation? Remove special symbols?
    # text = _basic_cleanup(text)
    text = re.sub(_whitespace_re, ' ', text).strip()
    return text

def _basic_cleanup(text):
    # Example: keep only Cyrillic, basic punctuation, space
    # THIS IS A VERY BASIC EXAMPLE - NEEDS REFINEMENT FOR MONGOLIAN
    text = re.sub(r'[^\u0400-\u04FF .,!?\-]', '', text) # Basic Cyrillic range + punctuation
    return text