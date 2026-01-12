from __future__ import annotations

import csv
import json
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndexEndpoint
from google.genai.types import EmbedContentConfig
import vertexai

from .config import (
    COURSE_CSV_PATH,
    DEFAULT_LOCATION,
    DEFAULT_PROJECT_ID,
    DEPLOYED_INDEX_ID,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    ENDPOINT_DISPLAY_NAME,
    GENERATION_MODEL,
    genai_client,
)
from .models import Course, CourseScore, Weakness, WeaknessRecommendations

# Initialize Vertex AI once for the internal API module.
vertexai.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
aiplatform.init(project=DEFAULT_PROJECT_ID, location=DEFAULT_LOCATION)
llm_client = genai_client


def recommend_courses_by_weakness(
    weaknesses: Iterable[Dict[str, Any]] | Iterable[Weakness],
    max_courses: int,
) -> List[WeaknessRecommendations]:
    """
    Internal API entry point.
    Inputs: weaknesses (list of dicts or Weakness objects), max_courses per weakness.
    Output: list of recommendations grouped per weakness.
    """
    if max_courses < 1:
        raise ValueError("max_courses must be >= 1.")

    parsed_weaknesses = _normalize_weaknesses(weaknesses)
    course_lookup = _load_course_lookup()

    results: List[WeaknessRecommendations] = []
    all_recs: List[CourseScore] = []
    for weakness in parsed_weaknesses:
        neighbors = _query_vertex_index(weakness.text, max_courses)
        recs: List[CourseScore] = []
        for neighbor in neighbors:
            course_id = str(neighbor.id)
            metadata = course_lookup.get(course_id, {})
            lesson_title = metadata.get("lesson_title") or "Untitled course"
            desc = metadata.get("description") or ""
            link = metadata.get("link") or metadata.get("course_url") or ""
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
            recs.append(
                CourseScore(
                    course=course,
                    weakness_id=weakness.id,
                    score=score,
                    reason=reason,
                )
            )
        unique_recs = _dedupe_by_course(recs)
        all_recs.extend(unique_recs)
        results.append(
            WeaknessRecommendations(
                weakness=weakness,
                recommendations=unique_recs,
            )
        )

    reranked = _llm_rerank_courses(parsed_weaknesses, all_recs)
    if reranked:
        return _rebuild_results(parsed_weaknesses, reranked, max_courses)
    return results


def embed_texts(texts: List[str], dim: int = EMBEDDING_DIMENSION) -> List[List[float]]:
    """Embed texts in batches to respect 100-request limit."""
    #TODO: check if batch_size > 100 in any case, & how does RAG treat for such case
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


def _load_course_lookup() -> Dict[str, Dict[str, Any]]:
    # TODO: link with DataGatheringAPI 
    if not COURSE_CSV_PATH.exists():
        raise FileNotFoundError(f"Course CSV not found at {COURSE_CSV_PATH}")

    course_lookup: Dict[str, Dict[str, Any]] = {}
    with COURSE_CSV_PATH.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            course_id = row.get("id")
            if not course_id:
                continue
            course_lookup[course_id] = row
    return course_lookup


def _normalize_weaknesses(
    weaknesses: Iterable[Dict[str, Any]] | Iterable[Weakness],
) -> List[Weakness]:
    parsed: List[Weakness] = []
    for item in weaknesses:
        if isinstance(item, Weakness):
            parsed.append(item)
            continue
        weakness_text = item.get("weakness") or item.get("text")
        if not weakness_text:
            raise ValueError("Each weakness must include a 'weakness' or 'text' field.")
        weakness_id = item.get("id") or str(uuid.uuid4())
        importance = float(item.get("importance", 1.0))
        meta = {
            key: value
            for key, value in item.items()
            if key not in ["id", "weakness", "text", "importance"]
        }
        parsed.append(
            Weakness(
                id=weakness_id,
                text=weakness_text,
                importance=importance,
                metadata=meta,
            )
        )
    return parsed


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
    query_vector = embed_texts([query_text])[0]
    neighbors = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_vector],
        num_neighbors=limit,
        return_full_datapoint=False,
    )
    return neighbors[0] if neighbors else []


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


def _rebuild_results(
    weaknesses: List[Weakness],
    recommendations: List[CourseScore],
    max_courses: int,
) -> List[WeaknessRecommendations]:
    recs_by_weakness: Dict[str, List[CourseScore]] = {}
    for rec in recommendations:
        recs_by_weakness.setdefault(rec.weakness_id, []).append(rec)

    results: List[WeaknessRecommendations] = []
    for weakness in weaknesses:
        recs = recs_by_weakness.get(weakness.id, [])
        recs.sort(key=lambda r: r.score, reverse=True)
        results.append(
            WeaknessRecommendations(
                weakness=weakness,
                recommendations=recs[:max_courses],
            )
        )
    return results


def extract_token_counts(response: Any) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if not usage:
        return None, None
    if isinstance(usage, dict):
        prompt = usage.get("prompt_token_count") or usage.get("input_tokens")
        output = usage.get("candidates_token_count") or usage.get("output_tokens")
        return _safe_int(prompt), _safe_int(output)
    prompt = getattr(usage, "prompt_token_count", None) or getattr(usage, "input_tokens", None)
    output = getattr(usage, "candidates_token_count", None) or getattr(usage, "output_tokens", None)
    return _safe_int(prompt), _safe_int(output)


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def log_token_usage(
    usage: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    runtime_seconds: float,
) -> None:
    if input_tokens is None and output_tokens is None:
        return
    print(
        "[USAGE] {usage} input={input_tokens} output={output_tokens} runtime={runtime:.2f}s".format(
            usage=usage,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            runtime=runtime_seconds,
        )
    )


def _llm_rerank_courses(
    weaknesses: List[Weakness],
    recommendations: List[CourseScore],
    model: str = GENERATION_MODEL,
    max_candidates_per_weakness: int = 4,
) -> List[CourseScore]:
    """
    Uses LLM to validate and re-rank the vector-search recommendations.
    Optimized to reduce tokens: prompt per-weakness with capped candidates.
    Returns a new list sorted by LLM relevance score if successful; otherwise returns [].
    """
    if not recommendations:
        return []

    recs_by_weakness: Dict[str, List[CourseScore]] = {}
    for rec in recommendations:
        recs_by_weakness.setdefault(rec.weakness_id, []).append(rec)

    # Sort each weakness bucket by current score and cap to reduce prompt size
    for wid, recs in recs_by_weakness.items():
        recs.sort(key=lambda r: r.score, reverse=True)
        recs_by_weakness[wid] = recs[:max_candidates_per_weakness]

    # Quick lookup for weakness text
    weakness_lookup = {w.id: w.text for w in weaknesses}

    rescored: List[CourseScore] = []

    for wid, recs in recs_by_weakness.items():
        weakness_text = weakness_lookup.get(wid) or ""
        rec_lines = "\n".join(
            f'- id="{r.course.id}", title="{r.course.lesson_title}"'
            for r in recs
        )
        prompt = f"""
            You are scoring courses for a single weakness.

            Weakness:
            "{weakness_text}"

            Candidate courses (keep all, just score relevance 0-1):
            {rec_lines}

            Output JSON ONLY:
            [
              {{"course_id": "<id>", "relevance_score": <0-1>, "justification": "<very short>"}},
              ...
            ]
            """
        try:
            response = None
            start = time.time()
            response = llm_client.models.generate_content(
                model=model,
                contents=[{"parts": [{"text": prompt}]}],
            )
            raw = (response.text or "").strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
            if not isinstance(data, list):
                continue
            rec_lookup = {r.course.id: r for r in recs}
            for item in data:
                cid = item.get("course_id")
                if cid not in rec_lookup:
                    continue
                base = rec_lookup[cid]
                score = float(item.get("relevance_score", base.score))
                justification = item.get("justification") or base.reason
                rescored.append(
                    CourseScore(
                        course=base.course,
                        weakness_id=base.weakness_id,
                        score=score,
                        reason=justification,
                    )
                )
        except Exception as exc:
            print(f"[WARN] LLM re-rank failed for weakness {wid}: {exc}")
            # fall back to existing ordering for this weakness
            rescored.extend(recs)
        finally:
            elapsed = time.time() - start if "start" in locals() else 0.0
            input_toks, output_toks = extract_token_counts(response) if response else (None, None)
            log_token_usage(
                usage=f"agent4: rerank weakness {wid}",
                input_tokens=input_toks,
                output_tokens=output_toks,
                runtime_seconds=elapsed,
            )

    # Sort overall; caller groups per-weakness.
    rescored.sort(key=lambda cs: cs.score, reverse=True)
    return rescored
