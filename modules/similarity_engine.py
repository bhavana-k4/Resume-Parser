"""
modules/similarity_engine.py
============================
Computes multi-signal semantic similarity between a resume and job description.

Signals computed:
  1. Full-doc cosine similarity   — overall semantic overlap
  2. Section-level similarity     — experience section vs JD, skills vs JD
  3. Skill overlap ratio          — semantic skill gap analysis
  4. Contextual coverage          — how many JD sentences are "covered"
     by at least one resume sentence above a similarity threshold

All signals use transformer embeddings — no keyword counting.
"""
import logging
import numpy as np
from modules.embedder  import embed_text, embed_chunks, cosine_similarity
from modules.preprocessing import preprocess

log = logging.getLogger(__name__)

SECTION_WEIGHTS = {
    'experience':    0.40,
    'skills':        0.30,
    'projects':      0.20,
    'summary':       0.10,
    'education':     0.05,
    'achievements':  0.05,
    'certifications':0.05,
    'header':        0.05,
}


def _full_doc_similarity(resume_chunks: list[str], jd_text: str) -> float:
    """Cosine similarity between mean-pooled resume embedding and JD embedding."""
    resume_vec = embed_chunks(resume_chunks)
    jd_vec     = embed_text(jd_text)
    return max(0.0, cosine_similarity(resume_vec, jd_vec))


def _section_similarity(resume_sections: dict, jd_text: str) -> float:
    """
    Weighted average of per-section similarities.
    Experience and Skills sections are weighted highest.
    """
    jd_vec  = embed_text(jd_text)
    total_w = 0.0
    score   = 0.0

    for section, text in resume_sections.items():
        if not text or len(text.split()) < 5:
            continue
        w       = SECTION_WEIGHTS.get(section, 0.05)
        sec_vec = embed_text(text)
        sim     = max(0.0, cosine_similarity(sec_vec, jd_vec))
        score  += sim * w
        total_w += w

    return score / total_w if total_w > 0 else 0.0


def _contextual_coverage(resume_text: str, jd_text: str,
                          threshold: float = 0.55) -> float:
    """
    Measures what fraction of JD sentences are semantically 'covered'
    by at least one resume sentence.

    A JD sentence is covered if any resume sentence has cosine similarity
    ≥ threshold with it. This penalises resumes that are generally similar
    but miss specific JD requirements.
    """
    jd_sents     = [s.strip() for s in jd_text.split('.') if len(s.strip()) > 20][:20]
    resume_sents = [s.strip() for s in resume_text.split('.') if len(s.strip()) > 20][:30]

    if not jd_sents or not resume_sents:
        return 0.5   # neutral if we can't compute

    try:
        from sentence_transformers import SentenceTransformer
        from modules.embedder import _load_model
        model = _load_model()

        jd_vecs     = model.encode(jd_sents,     normalize_embeddings=True, show_progress_bar=False)
        resume_vecs = model.encode(resume_sents, normalize_embeddings=True, show_progress_bar=False)

        covered = 0
        for jd_vec in jd_vecs:
            sims = np.dot(resume_vecs, jd_vec)
            if np.max(sims) >= threshold:
                covered += 1

        return covered / len(jd_sents)

    except Exception as e:
        log.warning(f"Contextual coverage failed: {e}")
        return 0.5


def compute_all_signals(
    resume_doc,    # ParsedDoc from preprocessing.py
    jd_doc,        # ParsedDoc from preprocessing.py
) -> dict:
    """
    Master function — runs all similarity signals.

    Returns:
        {
          "full_doc_sim":    float  0–1,
          "section_sim":     float  0–1,
          "coverage_sim":    float  0–1,
          "skill_matched":   list[str],
          "skill_missing":   list[str],
          "skill_ratio":     float  0–1,
          "semantic_notes":  list[str],
        }
    """
    log.info("Computing full-doc similarity…")
    full_sim = _full_doc_similarity(resume_doc.chunks, jd_doc.full_text)

    log.info("Computing section-level similarity…")
    sec_sim  = _section_similarity(resume_doc.sections, jd_doc.full_text)

    log.info("Computing contextual coverage…")
    cov_sim  = _contextual_coverage(resume_doc.full_text, jd_doc.full_text)

    log.info("Computing semantic skill gap…")
    from modules.skill_extractor import extract_skills, skill_gap
    resume_skills = extract_skills(resume_doc.full_text)
    jd_skills     = extract_skills(jd_doc.full_text)
    gap           = skill_gap(resume_skills, jd_skills)

    skill_ratio = (
        len(gap['matched']) / max(len(jd_skills), 1)
        if jd_skills else 0.0
    )

    return {
        "full_doc_sim":   round(full_sim, 4),
        "section_sim":    round(sec_sim, 4),
        "coverage_sim":   round(cov_sim, 4),
        "skill_ratio":    round(skill_ratio, 4),
        "skill_matched":  gap["matched"],
        "skill_missing":  gap["missing"],
        "resume_skills":  resume_skills,
        "jd_skills":      jd_skills,
        "semantic_notes": gap["semantic_matches"],
    }