# Course Recommendation API

Small internal Python API that mirrors Agent 4's course-recommendation flow.

Inputs:
- `weaknesses`: list of dicts (with `weakness` or `text`) or `Weakness` objects
- `max_courses`: maximum courses per weakness

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
    {"weakness": "Struggles with linear equations", "pattern_type": "numeracy"},
    {"weakness": "Misreads inference questions", "pattern_type": "reading_comprehension"},
]

results = recommend_courses_by_weakness(weaknesses, max_courses=5)
for entry in results:
    print(entry.weakness.text)
    for rec in entry.recommendations:
        print("-", rec.course.lesson_title, rec.score)
```
