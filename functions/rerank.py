from __future__ import annotations

import json
import time
from typing import List

from .config import GENERATION_MODEL, genai_client
from .models import CourseScore, Weakness
from .utils.token_log import extract_token_counts, log_token_usage

llm_client = genai_client


def llm_rerank_for_weakness(
    weakness: Weakness,
    recommendations: List[CourseScore],
    model: str = GENERATION_MODEL,
) -> List[CourseScore]:
    """
    Uses LLM to validate and re-rank the vector-search recommendations for one weakness.
    """
    if not recommendations:
        return []

    prompt = _build_rerank_prompt(weakness.text, recommendations)
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
            return recommendations
        rec_lookup = {r.course.id: r for r in recommendations}
        rescored: List[CourseScore] = []
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
        return rescored or recommendations
    except Exception as exc:
        print(f"[WARN] LLM re-rank failed for weakness {weakness.id}: {exc}")
        return recommendations
    finally:
        elapsed = time.time() - start if "start" in locals() else 0.0
        input_toks, output_toks = extract_token_counts(response) if response else (None, None)
        log_token_usage(
            usage=f"agent4: rerank weakness {weakness.id}",
            input_tokens=input_toks,
            output_tokens=output_toks,
            runtime_seconds=elapsed,
        )


def _build_rerank_prompt(weakness_text: str, recommendations: List[CourseScore]) -> str:
    rec_lines = "\n".join(
        f'- id="{r.course.id}", title="{r.course.lesson_title}"'
        for r in recommendations
    )
    return f"""
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
