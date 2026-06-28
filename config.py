import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# HuggingFace Token (Primary - FREE)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# OpenRouter Key (Fallback - FREE)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "omnicount_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_PORT = os.getenv("DB_PORT", "5432")

DB_PASSWORD_ENCODED = urllib.parse.quote_plus(DB_PASSWORD)
POSTGRES_URI = f"postgresql://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

PAGE_TITLE = "GrashDetection"
PAGE_ICON = "🐱"