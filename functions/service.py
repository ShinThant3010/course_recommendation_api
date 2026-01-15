from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
from typing import Any, Dict, Iterable, List

from .config import GENERATION_MODEL
from .models import CourseScore, Weakness, WeaknessRecommendations
from .recommendation_fetch import fetch_recommendations_for_weakness
from .rerank import llm_rerank_for_weakness


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

    parsed_weaknesses = _normalize_weaknesses(weaknesses)
    recs_by_weakness = _recommend_by_weakness(
        parsed_weaknesses,
        max_courses_per_weakness,
    )
    all_recs: List[CourseScore] = []
    for weakness in parsed_weaknesses:
        all_recs.extend(recs_by_weakness.get(weakness.id, []))

    deduped = _dedupe_by_best_score(all_recs)
    deduped.sort(key=lambda r: r.score, reverse=True)
    capped = deduped[:max_courses_overall]
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


def _recommend_by_weakness(
    weaknesses: List[Weakness],
    max_courses_per_weakness: int,
) -> Dict[str, List[CourseScore]]:
    if not weaknesses:
        return {}

    max_workers = min(8, len(weaknesses))
    if max_workers <= 1:
        return {
            weakness.id: _recommend_for_weakness(weakness, max_courses_per_weakness)
            for weakness in weaknesses
        }

    recs_by_weakness: Dict[str, List[CourseScore]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_recommend_for_weakness, weakness, max_courses_per_weakness): weakness.id
            for weakness in weaknesses
        }
        for future in as_completed(future_map):
            wid = future_map[future]
            recs_by_weakness[wid] = future.result()
    return recs_by_weakness


def _recommend_for_weakness(
    weakness: Weakness,
    max_courses_per_weakness: int,
) -> List[CourseScore]:
    recs = fetch_recommendations_for_weakness(weakness, max_courses_per_weakness)
    reranked = llm_rerank_for_weakness(weakness, recs, model=GENERATION_MODEL)
    reranked.sort(key=lambda r: r.score, reverse=True)
    return reranked


def _rebuild_results(
    weaknesses: List[Weakness],
    recommendations: List[CourseScore],
    max_courses_per_weakness: int,
) -> List[WeaknessRecommendations]:
    recs_by_weakness: Dict[str, List[CourseScore]] = {}
    for rec in recommendations:
        recs_by_weakness.setdefault(rec.weakness_id, []).append(rec)

    results: List[WeaknessRecommendations] = []
    for weakness in weaknesses:
        recs = recs_by_weakness.get(weakness.id, [])
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
