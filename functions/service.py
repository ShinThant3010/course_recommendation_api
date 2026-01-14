from __future__ import annotations

import uuid
from typing import Any, Dict, Iterable, List

from .config import GENERATION_MODEL
from .models import CourseScore, Weakness, WeaknessRecommendations
from .recommendation_fetch import fetch_recommendations_by_weakness, flatten_recommendations
from .rerank import llm_rerank_courses


def recommend_courses_by_weakness(
    weaknesses: Iterable[Dict[str, Any]] | Iterable[Weakness],
    max_courses_overall: int,
    max_courses_per_weakness: int,
) -> List[WeaknessRecommendations]:
    """
    Internal API entry point.
    Inputs: weaknesses (list of dicts or Weakness objects),
    max_courses_overall and max_courses_per_weakness limits.
    Output: list of recommendations grouped per weakness.
    """
    if max_courses_overall < 1:
        raise ValueError("max_courses_overall must be >= 1.")
    if max_courses_per_weakness < 1:
        raise ValueError("max_courses_per_weakness must be >= 1.")

    # Normalize weaknesses
    parsed_weaknesses = _normalize_weaknesses(weaknesses)

    # Fetch initial recommendations
    recs_by_weakness = fetch_recommendations_by_weakness(parsed_weaknesses, max_courses_per_weakness)
    
    # Flatten recommendations
    all_recs = flatten_recommendations(parsed_weaknesses, recs_by_weakness)

    # Rerank recommendations using LLM
    reranked = llm_rerank_courses(
        parsed_weaknesses,
        all_recs,
        model=GENERATION_MODEL,
        max_candidates_per_weakness=max_courses_per_weakness,
    )

    # Determine base recommendations to use
    base_recs = reranked if reranked else all_recs
    
    # Deduplicate and cap overall recommendations
    deduped = _dedupe_by_best_score(base_recs)
    deduped.sort(key=lambda r: r.score, reverse=True)
    capped = deduped[:max_courses_overall]
    
    # Rebuild results grouped by weakness; apply per-weakness cap
    return _rebuild_results(parsed_weaknesses, capped, max_courses_per_weakness)


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
        description = item.get("description") or ""
        weakness_id = item.get("id") or str(uuid.uuid4())
        importance = float(item.get("importance", 1.0))
        parsed.append(
            Weakness(
                id=weakness_id,
                text=weakness_text,
                description=description,
                importance=importance,
            )
        )
    return parsed


def _rebuild_results(weaknesses: List[Weakness], recommendations: List[CourseScore], max_courses_per_weakness: int) -> List[WeaknessRecommendations]:
    
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
                recommendations=recs[:max_courses_per_weakness],
            )
        )
    return results


def _dedupe_by_best_score(recs: List[CourseScore]) -> List[CourseScore]:
    best_by_course: Dict[str, CourseScore] = {}
    for rec in recs:
        cid = rec.course.id
        existing = best_by_course.get(cid)
        if existing is None or rec.score > existing.score:
            best_by_course[cid] = rec
    return list(best_by_course.values())
