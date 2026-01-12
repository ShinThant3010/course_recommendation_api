from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from functions.service import recommend_courses_by_weakness
from functions.models import WeaknessRecommendations, CourseScore, Weakness
from functions.utils.json_naming_converter import convert_keys_snake_to_camel


class RecommendationRequest(BaseModel):
    weaknesses: List[Dict[str, Any]] = Field(
        description="List of weakness objects (must include 'weakness' or 'text')."
    )
    max_courses: int = Field(ge=1, description="Maximum courses per weakness.")


class RecommendationResponse(BaseModel):
    recommendations: List[Dict[str, Any]]


app = FastAPI(
    title="Course Recommendation API",
    version="0.1.0",
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/v1/course-recommendations", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest) -> RecommendationResponse:
    results = recommend_courses_by_weakness(
        weaknesses=request.weaknesses,
        max_courses=request.max_courses,
    )
    serialized = _serialize_results(results)
    converted = convert_keys_snake_to_camel(
        serialized,
        preserve_container_keys={"metadata"},
    )
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
        "importance": weakness.importance,
        "metadata": weakness.metadata or {},
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
