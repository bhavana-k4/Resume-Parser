"""
modules/job_comparator.py
Compares one resume against multiple job descriptions and returns
a ranked list so candidates can see which roles fit them best.
"""
import logging
log = logging.getLogger(__name__)


def compare_jobs(
    resume_text:   str,
    resume_parsed: dict,
    jobs: list[dict],          # list of {"title": str, "company": str, "jd_text": str, "jd_parsed": dict}
) -> list[dict]:
    """
    Scores the resume against each job and returns them ranked best → worst.

    Each entry in the returned list:
        {
          "rank":            int,
          "title":           str,
          "company":         str,
          "overall_score":   int (0–100),
          "match_level":     str,
          "matched_skills":  list[str],
          "missing_skills":  list[str],
          "tfidf_score":     float,
          "skill_score":     float,
        }
    """
    from modules.matcher import compute_match_score

    results = []

    for job in jobs:
        title      = job.get("title", "Unknown Role")
        company    = job.get("company", "")
        jd_text    = job.get("jd_text", "")
        jd_parsed  = job.get("jd_parsed", {})

        if not jd_text.strip():
            log.warning(f"Skipping '{title}' — empty JD text.")
            continue

        match = compute_match_score(
            resume_text      = resume_text,
            jd_text          = jd_text,
            resume_skills    = resume_parsed.get("skills", []),
            required_skills  = jd_parsed.get("required_skills", []),
        )

        results.append({
            "title":          title,
            "company":        company,
            "overall_score":  match["overall_score"],
            "match_level":    match["match_level"],
            "matched_skills": match["matched_skills"],
            "missing_skills": match["missing_skills"],
            "tfidf_score":    match["tfidf_score"],
            "skill_score":    match["skill_score"],
        })

    # Sort best match first
    results.sort(key=lambda x: x["overall_score"], reverse=True)

    # Attach rank
    for i, r in enumerate(results, 1):
        r["rank"] = i

    return results


def format_comparison_table(results: list[dict]) -> list[dict]:
    """
    Returns a simplified list ready for rendering in an HTML table.
    """
    return [
        {
            "Rank":           r["rank"],
            "Role":           r["title"],
            "Company":        r["company"] or "—",
            "Match Score":    f"{r['overall_score']}%",
            "Level":          r["match_level"],
            "Matched Skills": len(r["matched_skills"]),
            "Missing Skills": len(r["missing_skills"]),
        }
        for r in results
    ]