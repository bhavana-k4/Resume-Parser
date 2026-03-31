"""
modules/gpt_analyzer.py
Uses Google Gemini (gemini-2.0-flash) via the new google-genai SDK.
Falls back gracefully on quota errors, missing keys, or any API failure.
"""
import re, json, logging, os
log = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"


def _get_client():
    try:
        from google import genai
    except ImportError:
        raise ImportError("Run: pip install google-genai")

    try:
        from config import GEMINI_API_KEY
    except ImportError:
        GEMINI_API_KEY = ""

    key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise ValueError("GEMINI_API_KEY not set. Add it to your .env file.")

    from google import genai as genai_mod
    return genai_mod.Client(api_key=key)


def _build_prompt(resume_text: str, jd_text: str, match_data: dict) -> str:
    matched = ', '.join(match_data.get('matched_skills', [])[:20]) or 'none detected'
    missing = ', '.join(match_data.get('missing_skills', [])[:20]) or 'none detected'
    score   = match_data.get('overall_score', 0)

    return f"""You are an expert technical recruiter and career coach.

A candidate's resume has been compared to a job description.
Match score: {score}/100
Skills the candidate HAS that match the JD: {matched}
Skills MISSING from the resume: {missing}

--- RESUME (first 2000 chars) ---
{resume_text[:2000]}

--- JOB DESCRIPTION (first 1500 chars) ---
{jd_text[:1500]}

Return a JSON object with EXACTLY these four keys:
{{
  "summary": "2 sentence verdict on overall fit",
  "gap_analysis": ["explanation of gap 1", "explanation of gap 2"],
  "suggestions": ["actionable tip 1", "actionable tip 2"],
  "rewrite_tips": ["Rewrite: 'old phrase' to 'improved phrase'"]
}}

Rules:
- gap_analysis: 3-6 items explaining WHY each missing skill matters for this role
- suggestions: 5-8 specific actions (courses, projects, certifications the candidate can pursue)
- rewrite_tips: 3-5 examples of weak resume bullets and how to strengthen them using JD language
- Be specific. Reference actual content from the JD and resume.
- Return ONLY valid JSON. No markdown fences, no preamble, no extra text.
"""


def _friendly_error(e: Exception) -> str:
    """Convert API exceptions into short, clean user-facing messages."""
    msg = str(e)
    if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
        return "quota_exceeded"
    if '403' in msg or 'API_KEY_INVALID' in msg:
        return "invalid_key"
    if '404' in msg or 'not found' in msg.lower():
        return "model_not_found"
    return "api_error"


def analyze(resume_text: str, jd_text: str, match_data: dict) -> dict:
    """
    Calls Gemini 2.0 Flash and returns structured gap analysis + suggestions.
    Always returns a clean dict — never exposes raw API error text to the UI.
    """
    fallback = {
        "summary": "AI analysis unavailable. Add GEMINI_API_KEY to your .env file.",
        "gap_analysis": [],
        "suggestions": [],
        "rewrite_tips": [],
        "ai_status": "unavailable",
    }

    try:
        client = _get_client()
    except (ImportError, ValueError) as e:
        log.warning(f"Gemini skipped: {e}")
        fallback["summary"] = str(e)
        return fallback

    try:
        prompt   = _build_prompt(resume_text, jd_text, match_data)
        response = client.models.generate_content(
            model    = GEMINI_MODEL,
            contents = prompt,
        )
        raw = response.text.strip()

        # Strip accidental markdown fences
        raw = re.sub(r'^```[a-zA-Z]*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw).strip()

        result = json.loads(raw)
        result["ai_status"] = "ok"
        return result

    except json.JSONDecodeError as e:
        log.error(f"Gemini returned invalid JSON: {e}")
        fallback["summary"] = "AI response could not be parsed. Try again."
        fallback["ai_status"] = "parse_error"
        return fallback

    except Exception as e:
        error_type = _friendly_error(e)
        log.warning(f"Gemini API issue ({error_type}): {e}")

        messages = {
            "quota_exceeded":  "quota_exceeded",
            "invalid_key":     "Gemini API key is invalid. Check your .env file.",
            "model_not_found": "Gemini model not available. Check your API key region.",
            "api_error":       "Gemini API temporarily unavailable. Try again shortly.",
        }
        fallback["summary"]    = messages.get(error_type, "AI temporarily unavailable.")
        fallback["ai_status"]  = error_type
        return fallback