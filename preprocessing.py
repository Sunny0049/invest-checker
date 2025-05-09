import re
import sys
import os
from cleantext import clean
from unidecode import unidecode

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def substitute_numeric_entities(text):
    """
    Replace patterns like 'user200' or 'hall5' with 'user<SUB>' or 'hall<SUB>'
    Replace standalone numbers like '5', '50.5' with '<SUB>'
    """
    # Replace alphanumeric words with trailing digits (user200 -> user<SUB>)
    text = re.sub(r'\b([a-zA-Z]+)\d+\b', r'\1<SUB>', text)
    # Replace pure numeric values (including decimal numbers)
    text = re.sub(r'\b\d+(\.\d+)?\b', '<SUB>', text)
    return text

def clean_suffix_if_irrelevant(text):
    """
    Remove irrelevant suffixes or noise at the end (e.g., 'Note.', '✓', 'M.', numbers).
    This now skips if it's part of a useful sentence.
    """
    suffix_pattern = r'\s+(?<!\w)(?:(?:Now|Z|✓|M|Note|X|Check|Done|End|Okay|Yes)[\.]?)$'
    return re.sub(suffix_pattern, '', text.strip(), flags=re.IGNORECASE)

def standard_clean_text(text: str) -> str:
    """
    Clean generic noise: line breaks, URLs, currency, emojis, etc.
    """
    return clean(
        text,
        fix_unicode=True,            # Normalize unicode
        to_ascii=True,               # Convert unicode to closest ASCII
        lower=False,                 # Preserve case (BERT is case-sensitive if uncased model is not used)
        no_line_breaks=True,         # Flatten multi-line text
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_currency_symbols=True,
        no_punct=False,              # Keep punctuation for syntactic structure
        replace_with_url="<SUB>",
        replace_with_email="<SUB>",
        replace_with_phone_number="<SUB>",
        replace_with_currency_symbol="<SUB>",
        replace_with_number="<SUB>", # E.g. "five" to "<SUB>"
        replace_with_digit="<SUB>"   # E.g. "5" to "<SUB>"
    )

def normalize_unicode(text: str) -> str:
    return unidecode(text)

def substitute_acceptance(text: str) -> str:
    return re.sub(r'\b(?:Acceptance Criteria|AC|Acceptance|acceptance_criteria)[:]?(\s|$)', '<AC> ', text, flags=re.IGNORECASE)

def mask_actors(text: str) -> str:
    """Replace actor role with [ACTOR] for patterns like 'As a user,' or 'As an admin,'."""
    return re.sub(r"\bAs\s+(?:a|an)\s+[^,]+,", "As a [ACTOR],", text, flags=re.IGNORECASE)

def clean_user_story(text: str) -> str:
    text = normalize_unicode(text)
    text = mask_actors(text)
    text = substitute_acceptance(text)
    text = substitute_numeric_entities(text)
    text = clean_suffix_if_irrelevant(text)
    text = standard_clean_text(text)
    return text.strip()
