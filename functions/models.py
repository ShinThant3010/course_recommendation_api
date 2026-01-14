from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Course:
    id: str
    lesson_title: str
    description: str
    link: str
    metadata: Dict[str, Any] | None = None


@dataclass
class Weakness:
    id: str
    text: str
    description: str
    importance: float = 1.0


@dataclass
class CourseScore:
    course: Course
    weakness_id: str
    score: float
    reason: str


@dataclass
class WeaknessRecommendations:
    weakness: Weakness
    recommendations: List[CourseScore]
