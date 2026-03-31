"""
modules/preprocessing.py
========================
Cleans and prepares raw text for the embedding pipeline.

Why this exists:
  Transformer models have a 512-token limit. Long resumes must be split into
  overlapping chunks so no information is lost. This module handles that plus
  all normalisation (encoding artefacts, bullet chars, etc.).
"""
import re
from dataclasses import dataclass, field
from typing import Optional

# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class ParsedDoc:
    full_text:   str        = ""          # full cleaned text
    chunks:      list[str]  = field(default_factory=list)  # 512-token-safe chunks
    sections:    dict       = field(default_factory=dict)  # detected section content
    word_count:  int        = 0
    char_count:  int        = 0


# ── Text cleaning ─────────────────────────────────────────────────────────────

_BULLET_CHARS = re.compile(r'[•►▪▸‣⦾⦿◆◇■□●○◉]')
_MULTI_SPACE  = re.compile(r'[ \t]{2,}')
_MULTI_NL     = re.compile(r'\n{3,}')
_CTRL_CHARS   = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def clean_text(raw: str) -> str:
    """
    Normalise raw extracted text:
      - Strip control characters
      - Replace fancy bullets with hyphens
      - Collapse whitespace and blank lines
      - Normalise unicode quotes/dashes
    """
    text = raw
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = _CTRL_CHARS.sub('', text)
    text = _BULLET_CHARS.sub('-', text)
    # Normalise unicode punctuation to ASCII
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = _MULTI_SPACE.sub(' ', text)
    text = _MULTI_NL.sub('\n\n', text)
    return text.strip()


# ── Section detection ─────────────────────────────────────────────────────────

_SECTION_HEADERS = {
    'summary':        r'(?:summary|objective|profile|about\s*me)',
    'experience':     r'(?:experience|employment|work\s*history|career|positions)',
    'education':      r'(?:education|academic|university|degree|qualification)',
    'skills':         r'(?:skills|technologies|technical\s*skills|competencies|tools)',
    'projects':       r'(?:projects|portfolio|personal\s*projects|open\s*source)',
    'certifications': r'(?:certifications?|certificates?|licenses?|credentials)',
    'achievements':   r'(?:achievements?|awards?|honours?|accomplishments?)',
}

def extract_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections.
    Returns a dict of { section_name: section_text }.
    Any text before the first detected section goes into 'header'.
    """
    lines      = text.split('\n')
    sections   = {}
    current    = 'header'
    buf        = []

    header_re = {k: re.compile(r'^\s*' + v + r'\s*:?\s*$', re.IGNORECASE)
                 for k, v in _SECTION_HEADERS.items()}

    for line in lines:
        matched = None
        for name, pattern in header_re.items():
            if pattern.match(line.strip()):
                matched = name
                break
        if matched:
            if buf:
                sections[current] = '\n'.join(buf).strip()
            current = matched
            buf = []
        else:
            buf.append(line)

    if buf:
        sections[current] = '\n'.join(buf).strip()

    return sections


# ── Chunking for transformer 512-token limit ─────────────────────────────────

def chunk_text(text: str, max_words: int = 200, overlap: int = 30) -> list[str]:
    """
    Splits text into overlapping word-windows so that a 512-token transformer
    can process a full resume without truncation.

    Args:
        text:      Cleaned full text.
        max_words: Maximum words per chunk (200 words ≈ ~270 tokens, safely under 512).
        overlap:   Words to repeat between consecutive chunks for context continuity.

    Returns:
        List of text chunks. Single-chunk list for short documents.
    """
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + max_words, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap   # overlap for context continuity

    return chunks


# ── Master function ───────────────────────────────────────────────────────────

def preprocess(raw_text: str) -> ParsedDoc:
    """
    Full preprocessing pipeline.
    Input:  raw text from extractor.py
    Output: ParsedDoc with cleaned text, chunks, sections, and stats.
    """
    cleaned  = clean_text(raw_text)
    chunks   = chunk_text(cleaned)
    sections = extract_sections(cleaned)

    return ParsedDoc(
        full_text  = cleaned,
        chunks     = chunks,
        sections   = sections,
        word_count = len(cleaned.split()),
        char_count = len(cleaned),
    )