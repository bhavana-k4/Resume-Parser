"""
app.py
Flask application — orchestrates all modules.

Routes:
  GET  /              → upload form
  POST /analyze       → single resume vs single JD
  POST /compare       → resume vs multiple JDs
  GET  /health        → sanity check
"""
import os, sys, json, logging
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import (Flask, request, render_template,
                   redirect, url_for, flash, jsonify)

# Make sure project root is on path when running from subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from modules.extractor    import extract_text, extract_from_string
from modules.nlp_parser   import parse_resume, parse_job_description
from modules.matcher      import compute_match_score
from modules.gpt_analyzer import analyze
from modules.ats_checker  import check_ats
from modules.job_comparator import compare_jobs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key          = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH


# ── helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS)


def save_upload(file) -> str:
    """Save an uploaded file and return the absolute path."""
    filename = secure_filename(file.filename)
    path     = os.path.join(config.UPLOAD_FOLDER, filename)
    file.save(path)
    return path


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.route("/analyze", methods=["POST"])
def analyze_route():
    # ── 1. Get resume text (upload only) ─────────────────────────────────
    resume_text = ""

    if 'resume_file' not in request.files or not request.files['resume_file'].filename:
        flash("Please upload a resume file (PDF, DOCX, or TXT).", "error")
        return redirect(url_for("index"))

    f = request.files['resume_file']
    if not allowed_file(f.filename):
        flash("Unsupported file type. Upload PDF, DOCX, or TXT.", "error")
        return redirect(url_for("index"))

    path   = save_upload(f)
    result = extract_text(path)
    if not result.success:
        flash(f"Could not read resume: {result.error}", "error")
        return redirect(url_for("index"))
    resume_text = result.text

    # ── 2. Get JD text ────────────────────────────────────────────────────
    jd_text = request.form.get("jd_text", "").strip()
    if not jd_text:
        flash("Please paste a job description.", "error")
        return redirect(url_for("index"))

    jd_result = extract_from_string(jd_text, "job_description")
    if not jd_result.success:
        flash("Job description is empty.", "error")
        return redirect(url_for("index"))
    jd_text = jd_result.text

    # ── 3. Parse ──────────────────────────────────────────────────────────
    log.info("Parsing resume and JD …")
    resume_parsed = parse_resume(resume_text)
    jd_parsed     = parse_job_description(jd_text)

    # ── 4. Match ──────────────────────────────────────────────────────────
    log.info("Computing match score …")
    match = compute_match_score(
        resume_text     = resume_text,
        jd_text         = jd_text,
        resume_skills   = resume_parsed["skills"],
        required_skills = jd_parsed["required_skills"],
    )

    # ── 5. ATS check ──────────────────────────────────────────────────────
    log.info("Running ATS check …")
    ats = check_ats(resume_text, jd_text)

    # ── 6. Gemini analysis ────────────────────────────────────────────────
    log.info("Calling Gemini for gap analysis …")
    gpt = analyze(resume_text, jd_text, match)

    # ── 7. Render ─────────────────────────────────────────────────────────
    return render_template("results.html",
        match          = match,
        ats            = ats,
        gpt            = gpt,
        resume_parsed  = resume_parsed,
        jd_parsed      = jd_parsed,
        resume_preview = resume_text[:600],
        jd_preview     = jd_text[:400],
        # Chart data (JSON-serialised for charts.js)
        chart_skills_matched  = json.dumps(match["matched_skills"][:12]),
        chart_skills_missing  = json.dumps(match["missing_skills"][:12]),
        chart_score_data      = json.dumps({
            "tfidf": round(match["tfidf_score"] * 100),
            "skill": round(match["skill_score"] * 100),
            "ats":   ats["ats_score"],
        }),
    )


@app.route("/compare", methods=["POST"])
def compare_route():
    # ── Resume ────────────────────────────────────────────────────────────
    resume_text = ""
    if 'resume_file' in request.files and request.files['resume_file'].filename:
        path   = save_upload(request.files['resume_file'])
        result = extract_text(path)
        if not result.success:
            return jsonify({"error": result.error}), 400
        resume_text = result.text
    elif request.form.get("resume_text", "").strip():
        resume_text = extract_from_string(request.form["resume_text"]).text
    else:
        return jsonify({"error": "No resume provided"}), 400

    # ── Multiple JDs (sent as JSON array in form field "jobs_json") ───────
    try:
        jobs_raw = json.loads(request.form.get("jobs_json", "[]"))
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid jobs_json format"}), 400

    if not jobs_raw:
        return jsonify({"error": "No job descriptions provided"}), 400

    resume_parsed = parse_resume(resume_text)
    jobs = []
    for j in jobs_raw:
        jd_text  = j.get("jd_text", "")
        jd_parsed = parse_job_description(jd_text)
        jobs.append({
            "title":     j.get("title", "Role"),
            "company":   j.get("company", ""),
            "jd_text":   jd_text,
            "jd_parsed": jd_parsed,
        })

    ranked = compare_jobs(resume_text, resume_parsed, jobs)
    return jsonify({"results": ranked})


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting Resume Parser …")
    app.run(debug=config.DEBUG, host="0.0.0.0", port=5000)