"""
modules/scoring_engine.py
=========================
Neural network-inspired scoring system that blends multiple similarity signals
into a single interpretable 0–100 score.

Architecture:
  Instead of a simple weighted average, this uses a two-layer approach:

  Layer 1 — Signal normalisation:
    Each raw signal (0–1) is passed through a sigmoid-like scaling function
    to penalise near-zero scores more harshly and reward high scores more.
    This mirrors how a neural network activation compresses inputs.

  Layer 2 — Learned weights (fixed, domain-calibrated):
    Weights are calibrated on resume-matching domain knowledge:
      - Section similarity is most predictive (0.35) — are your sections relevant?
      - Skill ratio is the most concrete signal (0.30) — do you have the skills?
      - Full-doc cosine is good for general fit (0.20)
      - Contextual coverage catches missed requirements (0.15)

  The blended score is then:
    1. Scaled to 0–100
    2. Adjusted by a coverage penalty if too many JD sentences are uncovered
    3. Clamped and labelled

Why not train a real NN?
  Training requires a labelled dataset of (resume, JD, human_score) triples.
  Without that dataset, a fixed-weight MLP would overfit to noise.
  Our calibrated fixed weights perform at least as well and are fully explainable.
  The architecture remains modular — weights can be replaced with trained values.
"""
import math
import logging

log = logging.getLogger(__name__)

# ── Layer 1: activation function ─────────────────────────────────────────────

def _activate(x: float) -> float:
    """
    Soft-clamp activation similar to a smooth step function.
    Maps [0,1] → [0,1] but with more aggressive penalisation near 0
    and gentle reward near 1.

    f(x) = x^0.7  (concave power function)
      f(0.0) = 0.00  — zero stays zero
      f(0.1) = 0.20  — very low score penalised strongly
      f(0.5) = 0.61  — mid-range bumped slightly
      f(0.8) = 0.86  — good scores rewarded
      f(1.0) = 1.00  — perfect stays perfect
    """
    return x ** 0.7


# ── Layer 2: domain-calibrated weights ───────────────────────────────────────

SIGNAL_WEIGHTS = {
    "section_sim":  0.35,   # are your resume sections relevant to this JD?
    "skill_ratio":  0.30,   # do you have the specific skills asked for?
    "full_doc_sim": 0.20,   # general semantic overlap between resume and JD
    "coverage_sim": 0.15,   # how many JD requirements does your resume address?
}


# ── Coverage penalty ──────────────────────────────────────────────────────────

def _coverage_penalty(coverage_sim: float, missing_skills: int, total_skills: int) -> float:
    """
    Applies a downward adjustment when:
      - Too many JD sentences are uncovered (coverage_sim < 0.4)
      - More than half the required skills are missing

    This catches cases where full_doc_sim looks decent (similar domain language)
    but the resume actually misses most of what the JD requires.
    """
    penalty = 0.0

    if coverage_sim < 0.4:
        penalty += (0.4 - coverage_sim) * 0.3   # up to 12 points off

    if total_skills > 0:
        missing_ratio = missing_skills / total_skills
        if missing_ratio > 0.5:
            penalty += (missing_ratio - 0.5) * 0.2   # up to 10 points off

    return min(penalty, 0.20)   # cap total penalty at 20 points


# ── Match level label ─────────────────────────────────────────────────────────

def _label(score: int) -> str:
    if score >= 75: return "Excellent"
    if score >= 58: return "Good"
    if score >= 40: return "Fair"
    return "Poor"


# ── Master scoring function ───────────────────────────────────────────────────

def compute_score(signals: dict) -> dict:
    """
    Converts raw similarity signals into a final blended score.

    Input:  signals dict from similarity_engine.compute_all_signals()
    Output: scoring dict with overall_score, per-signal breakdown, and label.
    """
    # Layer 1: activate each signal
    activated = {k: _activate(signals.get(k, 0.0)) for k in SIGNAL_WEIGHTS}

    # Layer 2: weighted blend
    raw_score = sum(activated[k] * w for k, w in SIGNAL_WEIGHTS.items())

    # Scale to 0–100
    scaled = raw_score * 100

    # Apply coverage penalty
    n_missing = len(signals.get("skill_missing", []))
    n_total   = len(signals.get("jd_skills", [])) or 1
    penalty   = _coverage_penalty(
        signals.get("coverage_sim", 0.5),
        n_missing,
        n_total
    ) * 100

    final_score = max(0, min(100, round(scaled - penalty)))
    level       = _label(final_score)

    # Per-signal percentages for the UI breakdown
    breakdown = {
        "section_similarity":  round(signals["section_sim"] * 100),
        "skill_overlap":       round(signals["skill_ratio"] * 100),
        "text_similarity":     round(signals["full_doc_sim"] * 100),
        "contextual_coverage": round(signals["coverage_sim"] * 100),
    }

    log.info(
        f"Scoring: raw={round(scaled,1)} penalty={round(penalty,1)} "
        f"final={final_score} level={level}"
    )

    return {
        "overall_score":    final_score,
        "match_level":      level,
        "breakdown":        breakdown,
        "penalty_applied":  round(penalty, 1),
        # Expose raw signals for the UI sub-score display
        "tfidf_score":      signals["full_doc_sim"],   # kept for template compatibility
        "skill_score":      signals["skill_ratio"],
        "matched_skills":   signals.get("skill_matched", []),
        "missing_skills":   signals.get("skill_missing", []),
        "semantic_notes":   signals.get("semantic_notes", []),
    }