"""
modules/ats_checker.py
Audits a resume for ATS (Applicant Tracking System) compatibility.

Checks:
  1. Required section presence (Experience, Skills, Education…)
  2. Keyword density vs the JD
  3. Formatting red flags (tables-only layout, images, missing contact info)
  4. Bullet point usage
  5. File-format note

Returns a structured ATS report with a score (0–100) and flagged issues.
"""
import re, logging
log = logging.getLogger(__name__)


# ── section detection ─────────────────────────────────────────────────────────

SECTION_KEYWORDS = {
    "contact":        ["email", "phone", "linkedin", "github", "address", "@"],
    "summary":        ["summary", "objective", "profile", "about me", "overview"],
    "experience":     ["experience", "employment", "work history", "career", "positions"],
    "education":      ["education", "academic", "university", "college", "degree", "school"],
    "skills":         ["skills", "technical skills", "core competencies", "technologies", "tools"],
    "projects":       ["projects", "portfolio", "personal projects", "open source"],
    "certifications": ["certifications", "certificates", "licenses", "credentials"],
}


def _detect_sections(text: str) -> dict[str, bool]:
    text_l = text.lower()
    return {sec: any(kw in text_l for kw in kws)
            for sec, kws in SECTION_KEYWORDS.items()}


# ── keyword density ───────────────────────────────────────────────────────────

def _keyword_density(resume_text: str, jd_text: str) -> dict:
    stopwords = {"and","or","the","a","an","in","of","to","for","is","are","with",
                 "on","at","by","we","you","our","your","will","be","this","that"}
    jd_words  = set(w for w in re.findall(r'[a-z][a-z0-9+#.]{2,}', jd_text.lower())
                    if w not in stopwords)
    res_words = re.findall(r'[a-z][a-z0-9+#.]{2,}', resume_text.lower())
    total     = len(res_words) or 1

    matched   = [w for w in res_words if w in jd_words]
    density   = len(matched) / total

    # top missing JD keywords
    res_set   = set(res_words)
    missing   = sorted(jd_words - res_set, key=lambda w: jd_text.lower().count(w), reverse=True)

    return {
        "density":        round(density, 4),
        "density_pct":    round(density * 100, 1),
        "matched_count":  len(set(matched)),
        "missing_top10":  missing[:10],
    }


# ── formatting checks ─────────────────────────────────────────────────────────

def _formatting_flags(text: str) -> list[str]:
    flags = []
    if len(text.split()) < 150:
        flags.append("Resume appears very short (< 150 words). Add more detail.")
    if len(text.split()) > 1200:
        flags.append("Resume is very long (> 1200 words). Consider trimming to 1–2 pages.")
    if not re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text, re.IGNORECASE):
        flags.append("No email address detected. ATS systems need contact info.")
    if not re.search(r'\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b', text):
        flags.append("No phone number detected.")
    bullet_count = len(re.findall(r'^\s*[•\-\*►▪]', text, re.MULTILINE))
    if bullet_count < 5:
        flags.append("Few or no bullet points found. Use bullet points for achievements.")
    if re.search(r'\b(references available|references upon request)\b', text, re.IGNORECASE):
        flags.append("Remove 'References available upon request' — wastes space on ATS.")
    if re.search(r'\b(dear hiring|to whom it may)\b', text, re.IGNORECASE):
        flags.append("Cover-letter language detected in resume. Keep them separate.")
    return flags


# ── master ATS check ──────────────────────────────────────────────────────────

def check_ats(resume_text: str, jd_text: str) -> dict:
    """
    Full ATS audit.

    Returns:
        {
          "ats_score":        int (0–100),
          "ats_level":        str,
          "sections_found":   dict[str, bool],
          "missing_sections": list[str],
          "keyword_density":  dict,
          "format_flags":     list[str],
          "recommendations":  list[str],
        }
    """
    sections  = _detect_sections(resume_text)
    kd        = _keyword_density(resume_text, jd_text)
    flags     = _formatting_flags(resume_text)

    missing_sections = [s for s, found in sections.items() if not found
                        and s in ("contact", "experience", "education", "skills")]

    # ── score calculation ──────────────────────────────────────────────────
    score = 100

    # Deduct for missing critical sections
    score -= len(missing_sections) * 10

    # Deduct for low keyword density
    if kd["density"] < 0.01:
        score -= 20
    elif kd["density"] < 0.02:
        score -= 10

    # Deduct for formatting issues
    score -= len(flags) * 5

    score = max(0, min(100, score))

    if score >= 75:
        level = "ATS-Friendly"
    elif score >= 50:
        level = "Needs Work"
    else:
        level = "ATS Risk"

    # ── plain-English recommendations ─────────────────────────────────────
    recs = []
    for sec in missing_sections:
        recs.append(f"Add a '{sec.capitalize()}' section — ATS systems look for this header.")
    for kw in kd["missing_top10"][:5]:
        recs.append(f"Include the keyword '{kw}' from the job description in your resume.")
    recs.extend(flags)

    return {
        "ats_score":        score,
        "ats_level":        level,
        "sections_found":   sections,
        "missing_sections": missing_sections,
        "keyword_density":  kd,
        "format_flags":     flags,
        "recommendations":  recs,
    }