from __future__ import annotations

import threading
import time
from typing import Any, List

from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
from google.genai.types import EmbedContentConfig
import vertexai

from .config import (
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DEPLOYED_INDEX_ID,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    ENDPOINT_DISPLAY_NAME,
    genai_client,
)
from .models import Course, CourseScore, Weakness
from .utils.course_info_client import get_course_info
from .utils.token_log import log_token_usage

# Initialize Vertex AI once for the internal API module.
vertexai.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
aiplatform.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
_ENDPOINT: MatchingEngineIndexEndpoint | None = None
_ENDPOINT_LOCK = threading.Lock()


def fetch_recommendations_for_weakness(
    weakness: Weakness,
    max_courses_per_weakness: int,
) -> List[CourseScore]:
    start = time.time()
    neighbors = _query_vertex_index(weakness.text, max_courses_per_weakness)
    elapsed = time.time() - start
    log_token_usage(
        usage=f"vector_search: {weakness.id}",
        input_tokens=None,
        output_tokens=None,
        runtime_seconds=elapsed,
    )
    recs = [_build_course_score(weakness, neighbor) for neighbor in neighbors]
    return _dedupe_by_course(recs)


def _build_course_score(weakness: Weakness, neighbor: Any) -> CourseScore:
    course_id = str(neighbor.id)
    metadata = get_course_info(course_id)
    course_meta = metadata.get("course") if isinstance(metadata, dict) else None
    source = course_meta if isinstance(course_meta, dict) else metadata
    lesson_title = source.get("lesson_title") or source.get("lessonTitle") or "Untitled course"
    desc = source.get("description") or source.get("short_description") or source.get("shortDescription") or ""
    link = source.get("link") or source.get("course_url") or ""
    course = Course(
        id=course_id,
        lesson_title=lesson_title,
        description=desc,
        link=link,
        metadata=metadata,
    )
    distance = float(getattr(neighbor, "distance", 0.0) or 0.0)
    score = 1 / (1 + distance)
    reason = f"Retrieved by semantic match to weakness '{weakness.text[:80]}...'."
    return CourseScore(
        course=course,
        weakness_id=weakness.id,
        score=score,
        reason=reason,
    )


def _query_vertex_index(query_text: str, limit: int) -> List[Any]:
    endpoint = _get_endpoint()
    query_vector = _embed_texts([query_text])[0]
    neighbors = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=limit,
        return_full_datapoint=False,
    )
    return neighbors[0] if neighbors else []


def _get_endpoint() -> MatchingEngineIndexEndpoint:
    global _ENDPOINT
    if _ENDPOINT is not None:
        return _ENDPOINT

    with _ENDPOINT_LOCK:
        if _ENDPOINT is not None:
            return _ENDPOINT
        endpoints = aiplatform.MatchingEngineIndexEndpoint.list()
        endpoint_name = ""
        for ep in endpoints:
            if ep.display_name == ENDPOINT_DISPLAY_NAME:
                endpoint_name = ep.resource_name
                break

        if not endpoint_name:
            raise ValueError(
                f"Matching Engine endpoint with display name '{ENDPOINT_DISPLAY_NAME}' not found."
            )
        _ENDPOINT = MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
        return _ENDPOINT


def _embed_texts(texts: List[str], dim: int = EMBEDDING_DIMENSION) -> List[List[float]]:
    """Embed texts in batches to respect 100-request limit."""
    batch_size = 100
    all_vectors: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = genai_client.models.embed_content(
            model=EMBEDDING_MODEL_NAME,
            contents=batch,
            config=EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=dim,
            ),
        )
        all_vectors.extend([e.values for e in resp.embeddings])
    return all_vectors


def _dedupe_by_course(recs: List[CourseScore]) -> List[CourseScore]:
    seen: set[str] = set()
    unique: List[CourseScore] = []
    for rec in recs:
        cid = rec.course.id
        if cid in seen:
            continue
        seen.add(cid)
        unique.append(rec)
    return unique
