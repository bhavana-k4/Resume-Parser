"""
modules/matcher.py
==================
Replaces TF-IDF cosine similarity with the full DL pipeline.
Keeps the same function signature as the old matcher.py
so app.py needs zero changes.

Old pipeline: TF-IDF vectorizer → cosine similarity → blended score
New pipeline: Sentence embeddings → multi-signal similarity → NN-weighted score
"""
import logging
from modules.preprocessing   import preprocess
from modules.similarity_engine import compute_all_signals
from modules.scoring_engine    import compute_score

log = logging.getLogger(__name__)


def compute_match_score(
    resume_text:     str,
    jd_text:         str,
    resume_skills:   list[str],
    required_skills: list[str],
    weights: dict = None,        # ignored — kept for API compatibility
) -> dict:
    """
    Deep learning-powered match scoring.

    Drops the old TF-IDF approach entirely and uses:
      1. Sentence transformer embeddings (all-MiniLM-L6-v2)
      2. Multi-signal similarity (full-doc, section-level, coverage, skill gap)
      3. Neural network-inspired weighted scoring

    Returns the same dict structure as the old matcher.py so app.py
    and results.html work without any changes.
    """
    log.info("Running DL match pipeline…")

    resume_doc = preprocess(resume_text)
    jd_doc     = preprocess(jd_text)

    signals = compute_all_signals(resume_doc, jd_doc)
    result  = compute_score(signals)

    # Ensure backward-compatible keys
    result.setdefault("tfidf_score",    signals["full_doc_sim"])
    result.setdefault("skill_score",    signals["skill_ratio"])
    result.setdefault("matched_skills", signals.get("skill_matched", []))
    result.setdefault("missing_skills", signals.get("skill_missing", []))

    return result