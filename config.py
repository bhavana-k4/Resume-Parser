import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

SECRET_KEY          = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
DEBUG               = os.getenv("DEBUG", "true").lower() == "true"
MAX_CONTENT_LENGTH  = 5 * 1024 * 1024          # 5 MB

BASE_DIR      = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DATA_DIR      = os.path.join(BASE_DIR, "data")
SKILLS_DB_PATH = os.path.join(DATA_DIR, "skills_db.json")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Scoring blend (must sum to 1.0)
SCORE_WEIGHTS = {"tfidf_cosine": 0.5, "skill_overlap": 0.5}

# ATS
ATS_REQUIRED_SECTIONS = [
    "experience", "education", "skills", "summary",
    "objective", "work", "employment", "projects", "certifications"
]
ATS_MIN_KEYWORD_DENSITY = 0.01   # 1 % of words should be JD keywords