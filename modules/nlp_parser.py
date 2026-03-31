"""
modules/nlp_parser.py
=====================
Backward-compatible wrapper over the new DL pipeline.
app.py still calls parse_resume() and parse_job_description() —
this module forwards those calls to the new DL modules.

Also retains rule-based experience/education extraction
(these work well without DL and don't benefit from embeddings).
"""
import re
import datetime
import logging
from modules.preprocessing  import preprocess
from modules.skill_extractor import extract_skills

log = logging.getLogger(__name__)


# ── Education extraction (rule-based — works perfectly) ──────────────────────

_EDU_KEYWORDS = {
    "phd":        ["ph.d", "phd", "doctor of philosophy", "doctorate", "d.phil"],
    "masters":    ["m.sc", "m.s.", "msc", "m.e.", "mba", "m.tech", "m.ca",
                   "master of", "master's", "masters", "pg diploma", "pgdm"],
    "bachelors":  ["b.sc", "b.s.", "b.e.", "b.tech", "b.a.", "bca", "bba",
                   "bachelor", "b.com", "b.eng", "undergraduate"],
    "associate":  ["associate degree", "a.s.", "a.a."],
    "high_school":["high school", "hsc", "secondary school", "12th", "10+2", "ssc"],
}

def extract_education(text: str) -> str:
    text_l = text.lower()
    for level in ["phd", "masters", "bachelors", "associate", "high_school"]:
        for kw in _EDU_KEYWORDS[level]:
            if kw in text_l:
                return level
    return "unknown"


# ── Experience extraction (improved rule-based) ───────────────────────────────

_ACADEMIC_YEAR   = re.compile(r'\b20\d{2}-\d{2}\b')
_FULL_YEAR_RANGE = re.compile(
    r'\b(20[0-2]\d|19[89]\d)\s*[-–—to]+\s*(20[0-3]\d|19[89]\d|present|current|now|till\s*date)\b',
    re.IGNORECASE
)
_MONTH_YEAR_RANGE = re.compile(
    r'(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'[\s.,\-]*(20[0-2]\d|19[89]\d)'
    r'\s*[-–—to]+\s*'
    r'(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
    r'[\s.,\-]*)?(20[0-3]\d|19[89]\d|present|current|now|till\s*date)',
    re.IGNORECASE
)

def _calc_years(start: str, end: str) -> float:
    current = datetime.date.today().year
    s = int(start)
    e = current if any(x in end.lower() for x in ('present','current','now','till')) else int(end)
    return max(0.0, float(e - s))

def extract_experience_years(text: str) -> float:
    cleaned = _ACADEMIC_YEAR.sub('', text)
    total, seen = 0.0, set()
    for m in _MONTH_YEAR_RANGE.finditer(cleaned):
        start, end = m.group(1), m.group(2)
        if not end: continue
        key = (start, end.lower())
        if key not in seen:
            seen.add(key); total += _calc_years(start, end)
    for m in _FULL_YEAR_RANGE.finditer(cleaned):
        start, end = m.group(1), m.group(2)
        key = (start, end.lower())
        if key not in seen:
            seen.add(key); total += _calc_years(start, end)
    return round(total, 1)


# ── Job title extraction ──────────────────────────────────────────────────────

_TITLE_WORDS = (
    r'Engineer|Developer|Analyst|Manager|Lead|Architect|Designer|Scientist|'
    r'Consultant|Specialist|Director|Officer|Intern|Administrator|Researcher|'
    r'Programmer|Coordinator|Executive|Associate'
)
_TITLE_RE   = re.compile(r'(?:^|[\n|•\-])\s*([A-Z][A-Za-z\s/&()]{2,50}(?:' + _TITLE_WORDS + r'))', re.MULTILINE)
_NOT_TITLE  = re.compile(r'^\d|\d{4}|^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', re.IGNORECASE)

def extract_job_titles(text: str) -> list[str]:
    titles = set()
    for m in _TITLE_RE.finditer(text):
        t = m.group(1).strip()
        if 5 < len(t) < 70 and not _NOT_TITLE.search(t):
            if sum(c.isdigit() for c in t) / max(len(t),1) < 0.1:
                titles.add(t)
    return sorted(titles)[:5]


# ── Public API (called by app.py) ─────────────────────────────────────────────

def parse_resume(text: str) -> dict:
    """Full resume parse — DL skills + rule-based experience/education."""
    doc = preprocess(text)
    return {
        "skills":           extract_skills(text),
        "experience_years": extract_experience_years(text),
        "education":        extract_education(text),
        "job_titles":       extract_job_titles(text),
        "sections":         doc.sections,
        "word_count":       doc.word_count,
    }

def parse_job_description(text: str) -> dict:
    """JD parse — extract required skills and all keywords."""
    all_skills = extract_skills(text)
    required, preferred = [], []
    text_l = text.lower()
    for skill in all_skills:
        idx = text_l.find(skill)
        ctx = text_l[max(0, idx-120):idx+120]
        if any(w in ctx for w in ["required","must have","must-have","essential","mandatory"]):
            required.append(skill)
        elif any(w in ctx for w in ["preferred","nice to have","plus","bonus","desired"]):
            preferred.append(skill)
        else:
            required.append(skill)
    stopwords = {"and","or","the","a","an","in","of","to","for","is","are","with","on","at","by"}
    keywords  = [w for w in re.findall(r'[a-z][a-z0-9+#.]{1,}', text_l)
                 if w not in stopwords and len(w) > 2]
    return {
        "required_skills":  sorted(set(required)),
        "preferred_skills": sorted(set(preferred)),
        "all_keywords":     sorted(set(keywords)),
    }