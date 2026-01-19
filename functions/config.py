from __future__ import annotations

import os
from pathlib import Path

from google import genai

DEFAULT_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "poc-piloturl-nonprod")
DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "asia-southeast1")

ENDPOINT_DISPLAY_NAME = os.getenv("COURSE_ENDPOINT_DISPLAY_NAME", "courses_endpoint")
DEPLOYED_INDEX_ID = os.getenv("COURSE_DEPLOYED_INDEX_ID", "deployed_courses_endpoint")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-2.5-flash")

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing!")
genai_client = genai.Client(api_key=API_KEY)

_DEFAULT_COURSE_CSV = (
    Path(__file__).resolve().parent.parent
    / "_data"
    / "courses"
    / "course.csv"
)
COURSE_CSV_PATH = Path(os.getenv("COURSE_CSV_PATH", str(_DEFAULT_COURSE_CSV)))

COURSE_INFO_API_BASE_URL = os.getenv(
    "COURSE_INFO_API_BASE_URL",
    "https://test-result-data-api-810737581373.asia-southeast1.run.app",
)
COURSE_INFO_API_TIMEOUT_SECONDS = float(os.getenv("COURSE_INFO_API_TIMEOUT_SECONDS", "5"))
