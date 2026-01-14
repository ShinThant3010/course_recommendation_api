from __future__ import annotations

import json
import threading
import socket
import urllib.error
import urllib.request
from typing import Any, Dict

from ..config import COURSE_INFO_API_BASE_URL, COURSE_INFO_API_TIMEOUT_SECONDS

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def get_course_info(course_id: str) -> Dict[str, Any]:
    if not course_id or not COURSE_INFO_API_BASE_URL:
        return {}

    with _LOCK:
        cached = _CACHE.get(course_id)
    if cached is not None:
        return cached

    data: Dict[str, Any] = {}
    try:
        data = _fetch_course_info(course_id)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, TimeoutError, socket.timeout) as exc:
        print(f"[WARN] Course info fetch failed for {course_id}: {exc}")
        data = {}

    with _LOCK:
        _CACHE[course_id] = data
    return data


def _fetch_course_info(course_id: str) -> Dict[str, Any]:
    base = COURSE_INFO_API_BASE_URL.rstrip("/")
    url = f"{base}/v1/course-info/{course_id}"
    with urllib.request.urlopen(url, timeout=COURSE_INFO_API_TIMEOUT_SECONDS) as resp:
        if getattr(resp, "status", None) not in (None, 200):
            return {}
        raw = resp.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))
