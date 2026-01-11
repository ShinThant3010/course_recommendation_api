from __future__ import annotations

import os
from pathlib import Path

from google import genai

DEFAULT_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", "poc-piloturl-nonprod")
DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "asia-southeast1")

ENDPOINT_DISPLAY_NAME = os.getenv("COURSE_ENDPOINT_DISPLAY_NAME", "Courses Endpoint")
DEPLOYED_INDEX_ID = os.getenv("COURSE_DEPLOYED_INDEX_ID", "courses_deployment")
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
