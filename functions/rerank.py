from __future__ import annotations

import json
import time
from typing import Dict, List

from .config import GENERATION_MODEL, genai_client
from .models import CourseScore, Weakness
from .utils.token_log import extract_token_counts, log_token_usage

llm_client = genai_client


def llm_rerank_courses(
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
        prompt = _build_rerank_prompt(weakness_text, recs)
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
