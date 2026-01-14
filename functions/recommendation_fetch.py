from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

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

# Initialize Vertex AI once for the internal API module.
vertexai.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
aiplatform.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)


def fetch_recommendations_by_weakness(
    weaknesses: List[Weakness],
    max_courses_per_weakness: int,
) -> Dict[str, List[CourseScore]]:
    recs_by_weakness: Dict[str, List[CourseScore]] = {}

    max_workers = min(8, len(weaknesses)) if weaknesses else 0
    if max_workers <= 1:
        for weakness in weaknesses:
            wid, recs = _build_recommendations(weakness, max_courses_per_weakness)
            recs_by_weakness[wid] = recs
        return recs_by_weakness

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_build_recommendations, weakness, max_courses_per_weakness): weakness.id
            for weakness in weaknesses
        }
        for future in as_completed(future_map):
            wid, recs = future.result()
            recs_by_weakness[wid] = recs
    return recs_by_weakness


def flatten_recommendations(
    weaknesses: List[Weakness],
    recs_by_weakness: Dict[str, List[CourseScore]],
) -> List[CourseScore]:
    all_recs: List[CourseScore] = []
    for weakness in weaknesses:
        all_recs.extend(recs_by_weakness.get(weakness.id, []))
    return all_recs


def _build_recommendations(
    weakness: Weakness,
    max_courses_per_weakness: int,
) -> Tuple[str, List[CourseScore]]:
    neighbors = _query_vertex_index(weakness.text, max_courses_per_weakness)
    recs = [_build_course_score(weakness, neighbor) for neighbor in neighbors]
    return weakness.id, _dedupe_by_course(recs)


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

    endpoint = MatchingEngineIndexEndpoint(index_endpoint_name=endpoint_name)
    query_vector = _embed_texts([query_text])[0]
    neighbors = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=limit,
        return_full_datapoint=False,
    )
    return neighbors[0] if neighbors else []


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
