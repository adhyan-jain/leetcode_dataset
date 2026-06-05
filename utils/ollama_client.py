from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    raw: Dict[str, Any]
    response_text: str


def generate(
    *,
    prompt: str,
    model: str,
    ollama_url: str = "http://localhost:11434",
    timeout: int = 120,
    retries: int = 3,
    temperature: float = 0.2,
    sleep_between_requests: float = 0.0,
    prefer_json: bool = True,
) -> OllamaResponse:
    url = ollama_url.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    if prefer_json:
        payload["format"] = "json"

    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            response_text = data.get("response", "")
            if isinstance(response_text, dict):
                response_text = json.dumps(response_text, ensure_ascii=False)
            elif response_text is None:
                response_text = ""
            if sleep_between_requests:
                time.sleep(sleep_between_requests)
            return OllamaResponse(raw=data, response_text=str(response_text))
        except Exception as exc:  # pragma: no cover - network/remote failures
            last_error = exc
            logger.warning(
                "Ollama request failed on attempt %s/%s: %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(min(2.0 * attempt, 5.0))
    raise RuntimeError(f"Ollama request failed after {retries} attempts: {last_error}") from last_error

