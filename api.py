from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from fastapi import FastAPI, Response, Request, Header
from pydantic import BaseModel, Field

from functions.service import recommend_courses_by_weakness
from functions.models import WeaknessRecommendations, CourseScore, Weakness
from functions.utils.json_naming_converter import convert_keys_snake_to_camel
from functions.utils.token_log import get_token_entries, reset_token_log


class RecommendationRequest(BaseModel):
    weaknesses: List[Dict[str, Any]] = Field(
        description="List of weakness objects (must include 'weakness' or 'text')."
    )
    max_course: int = Field(default=5, ge=1,
        description="Maximum total courses returned.",
    )
    max_course_pr_weakness: int = Field(default=3, ge=1,
        description="Maximum courses retrieved per weakness.",
    )


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]
    log: Optional[List[Dict[str, Any]]] = None


app = FastAPI(
    title="Course Recommendation API",
    version="0.1.0",
)


@app.middleware("http")
async def add_runtime_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Runtime-Seconds"] = f"{elapsed:.2f}"
    return response


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post(
    "/v1/course-recommendations",
    response_model=RecommendationResponse,
    response_model_exclude_none=True,
)
def get_recommendations(
    request: RecommendationRequest,
    include_log: bool = Header(True, convert_underscores=False),
) -> RecommendationResponse:
    reset_token_log()
    results = recommend_courses_by_weakness(
        weaknesses=request.weaknesses,
        max_courses_overall=request.max_course,
        max_courses_per_weakness=request.max_course_pr_weakness,
    )
    serialized = _serialize_results(results)
    converted = convert_keys_snake_to_camel(
        serialized,
        preserve_container_keys={"metadata"},
    )
    if include_log:
        return RecommendationResponse(recommendations=converted, log=get_token_entries())
    
    return RecommendationResponse(recommendations=converted)


def _serialize_results(results: List[WeaknessRecommendations]) -> List[Dict[str, Any]]:
    return [
        {
            "weakness": _serialize_weakness(entry.weakness),
            "recommended_courses": [_serialize_course_score(cs) for cs in entry.recommendations],
        }
        for entry in results
    ]


def _serialize_weakness(weakness: Weakness) -> Dict[str, Any]:
    return {
        "id": weakness.id,
        "text": weakness.text,
        "description": weakness.description,
        "importance": weakness.importance,
    }


def _serialize_course_score(score: CourseScore) -> Dict[str, Any]:
    course = score.course
    return {
        "course_id": course.id,
        "lesson_title": course.lesson_title,
        "description": course.description,
        "link": course.link,
        "metadata": course.metadata or {},
        "weakness_id": score.weakness_id,
        "score": score.score,
        "reason": score.reason,
    }
