"""
modules/skill_extractor.py
==========================
Extracts skills using a two-stage NLP pipeline:

  Stage 1 — spaCy NER:
    Uses the en_core_web_sm model to detect named entities.
    ORG, PRODUCT, and GPE entities often correspond to technologies
    (e.g. "TensorFlow", "AWS", "PostgreSQL").

  Stage 2 — Semantic skill matching:
    Embeds candidate phrases and matches them against a lightweight
    skill taxonomy using cosine similarity on sentence embeddings.
    This catches paraphrased skills (e.g. "deep neural networks" → "deep learning").

Why not a static list?
  Static lists miss variants ("React.js" vs "ReactJS" vs "React"),
  paraphrases ("natural language understanding" vs "NLP"), and
  domain-specific skills not in the list. Embedding-based matching
  handles all of these gracefully.
"""
import re
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# ── Skill taxonomy (seed list for semantic matching) ─────────────────────────
# These are representative seeds. The embedder finds semantically similar terms
# even if the exact string doesn't appear in the resume.

SKILL_SEEDS = [
    # Programming languages
    "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "scala", "r programming", "matlab",
    # ML / AI
    "machine learning", "deep learning", "neural networks", "computer vision",
    "natural language processing", "reinforcement learning", "transformers",
    "large language models", "generative ai", "mlops", "model deployment",
    "feature engineering", "data preprocessing", "statistical modeling",
    # ML Libraries
    "tensorflow", "pytorch", "keras", "scikit-learn", "hugging face",
    "xgboost", "lightgbm", "opencv", "nltk", "spacy",
    # Data
    "pandas", "numpy", "sql", "nosql", "mongodb", "postgresql", "mysql",
    "spark", "hadoop", "kafka", "airflow", "dbt", "bigquery", "snowflake",
    # Web
    "react", "angular", "vue", "node.js", "flask", "django", "fastapi",
    "rest api", "graphql", "html", "css",
    # Cloud / DevOps
    "aws", "azure", "google cloud", "docker", "kubernetes", "terraform",
    "ci/cd", "github actions", "linux",
    # Tools
    "git", "jira", "agile", "scrum", "tableau", "power bi", "excel",
    # Soft skills (contextual)
    "project management", "team leadership", "cross-functional collaboration",
    "stakeholder communication", "problem solving",
]

_SKILL_EMBEDDINGS: dict = {}   # cached on first use


def _get_skill_embeddings() -> dict:
    global _SKILL_EMBEDDINGS
    if not _SKILL_EMBEDDINGS:
        from modules.embedder import embed_skill_list
        log.info("Building skill seed embeddings…")
        _SKILL_EMBEDDINGS = embed_skill_list(SKILL_SEEDS)
    return _SKILL_EMBEDDINGS


# ── Stage 1: spaCy NER ────────────────────────────────────────────────────────

_NLP = None

def _get_spacy():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            try:
                _NLP = spacy.load("en_core_web_sm")
            except OSError:
                log.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
                _NLP = False
        except ImportError:
            _NLP = False
    return _NLP if _NLP else None


def _extract_ner_candidates(text: str) -> list[str]:
    """
    Uses spaCy NER to extract entity phrases that are likely skills/technologies.
    Targets: ORG (TensorFlow, AWS), PRODUCT (iPhone, Excel), and noun chunks.
    """
    nlp = _get_spacy()
    candidates = []

    if nlp:
        doc = nlp(text[:5000])
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PRODUCT", "GPE", "WORK_OF_ART"):
                t = ent.text.strip()
                if 1 < len(t) < 50 and not t.isdigit():
                    candidates.append(t.lower())
        # Also extract short noun chunks (likely tech phrases)
        for chunk in doc.noun_chunks:
            t = chunk.text.strip().lower()
            if 2 < len(t) < 40 and not re.search(r'\d{4}', t):
                candidates.append(t)

    return list(set(candidates))


# ── Stage 2: Semantic matching ────────────────────────────────────────────────

def _semantic_match(candidates: list[str], threshold: float = 0.72) -> list[str]:
    """
    Embeds candidate phrases and matches against SKILL_SEEDS using cosine similarity.
    Returns the seed skill name for any candidate with similarity ≥ threshold.

    threshold=0.72 is tuned to:
      - Accept: "deep neural nets" → "deep learning" (0.81)
      - Accept: "react.js" → "react" (0.94)
      - Reject: "team player" → "python" (0.12)
    """
    if not candidates:
        return []

    try:
        from modules.embedder import embed_skill_list, cosine_similarity
    except ImportError:
        return []

    seed_embeddings = _get_skill_embeddings()
    cand_embeddings = embed_skill_list(candidates)
    matched = set()

    for cand_text, cand_vec in cand_embeddings.items():
        best_score = 0.0
        best_seed  = None
        for seed_text, seed_vec in seed_embeddings.items():
            score = cosine_similarity(cand_vec, seed_vec)
            if score > best_score:
                best_score = score
                best_seed  = seed_text
        if best_score >= threshold and best_seed:
            matched.add(best_seed)

    return sorted(matched)


# ── Direct seed scan (fast path) ─────────────────────────────────────────────

def _direct_scan(text: str) -> list[str]:
    """
    Fast substring scan for exact / near-exact seed matches.
    Catches cases the NER might miss (e.g. skill listed as plain text in a section).
    """
    text_lower = text.lower()
    found = []
    for skill in SKILL_SEEDS:
        escaped = re.escape(skill)
        if len(skill) <= 2:
            pattern = r'\b' + escaped + r'\b'
        else:
            pattern = r'(?<![a-z0-9+#.\-])' + escaped + r'(?![a-z0-9+#.\-])'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


# ── Master extraction function ────────────────────────────────────────────────

def extract_skills(text: str) -> list[str]:
    """
    Full two-stage skill extraction:
      1. Direct seed scan (fast, catches exact matches)
      2. spaCy NER → semantic matching (catches variants and paraphrases)

    Returns deduplicated, sorted list of skill strings.
    """
    # Stage 1: direct scan
    direct = set(_direct_scan(text))

    # Stage 2: NER + semantic matching
    ner_candidates = _extract_ner_candidates(text)
    semantic       = set(_semantic_match(ner_candidates))

    all_skills = sorted(direct | semantic)
    log.debug(f"Extracted {len(all_skills)} skills: {all_skills[:10]}…")
    return all_skills


def skill_gap(resume_skills: list[str], jd_skills: list[str]) -> dict:
    """
    Computes matched and missing skills using BOTH exact match and
    semantic similarity on embeddings.

    A JD skill is considered "matched" if:
      - It appears exactly in resume_skills, OR
      - Its embedding is within cosine distance 0.80 of any resume skill embedding

    This means "ML engineer" in JD matches "machine learning" in resume.
    """
    if not jd_skills:
        return {"matched": [], "missing": [], "semantic_matches": []}

    try:
        from modules.embedder import embed_skill_list, cosine_similarity
        SEMANTIC_THRESHOLD = 0.80

        resume_embs = embed_skill_list(resume_skills) if resume_skills else {}
        jd_embs     = embed_skill_list(jd_skills)

        matched         = []
        missing         = []
        semantic_matches = []   # near-matches worth noting

        for jd_skill, jd_vec in jd_embs.items():
            # Exact match
            if jd_skill in resume_skills:
                matched.append(jd_skill)
                continue

            # Semantic match
            best_score  = 0.0
            best_resume = None
            for res_skill, res_vec in resume_embs.items():
                score = cosine_similarity(jd_vec, res_vec)
                if score > best_score:
                    best_score  = score
                    best_resume = res_skill

            if best_score >= SEMANTIC_THRESHOLD:
                matched.append(jd_skill)
                if best_resume and best_resume != jd_skill:
                    semantic_matches.append(f"'{best_resume}' covers '{jd_skill}'")
            else:
                missing.append(jd_skill)

        return {
            "matched":          sorted(matched),
            "missing":          sorted(missing),
            "semantic_matches": semantic_matches,
        }

    except Exception as e:
        log.warning(f"Semantic skill gap failed, using exact: {e}")
        matched = [s for s in jd_skills if s in resume_skills]
        missing = [s for s in jd_skills if s not in resume_skills]
        return {"matched": matched, "missing": missing, "semantic_matches": []}