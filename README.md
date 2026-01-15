# Course Recommendation API

Small internal Python API that mirrors Agent 4's course-recommendation flow.

Inputs:
- `weaknesses`: list of dicts (with `weakness` or `text`, plus `description`) or `Weakness` objects
- `max_courses_overall`: maximum courses returned across all weaknesses
- `max_courses_per_weakness`: maximum courses retrieved per weakness

Output:
- list of `WeaknessRecommendations` (one per weakness)

FastAPI:

```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

Example:

```python
from functions import recommend_courses_by_weakness

weaknesses = [
    {"weakness": "Struggles with linear equations", "description": "Needs algebra refresh."},
    {"weakness": "Misreads inference questions", "description": "Has trouble with main-idea questions."},
]

results = recommend_courses_by_weakness(
    weaknesses,
    max_courses_overall=5,
    max_courses_per_weakness=3,
)
for entry in results:
    print(entry.weakness.text)
    for rec in entry.recommendations:
        print("-", rec.course.lesson_title, rec.score)
```
