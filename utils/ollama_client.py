from __future__ import annotations
from utils.logging_utils import log_execution, setup_global_logger

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    raw: Dict[str, Any]
    response_text: str


@log_execution
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
    import os
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - environment-specific dependency
        raise RuntimeError(
            "The 'requests' package is required for Ollama/Groq calls. "
            "Use the bundled runtime or install requests in the active environment."
        ) from exc
    grok_api = os.getenv("GROK_API")
    
    if grok_api:
        # Route to Groq API
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {grok_api}",
            "Content-Type": "application/json"
        }
        
        # Override model if requested
        if model.startswith("qwen"):
            current_model = "llama-3.3-70b-versatile"
        else:
            current_model = model
            
        payload = {
            "model": current_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if prefer_json:
            payload["response_format"] = {"type": "json_object"}
            
        last_error = None
        retries_groq = max(retries, 20)  # Always retry a lot for Groq 429s
        for attempt_num in range(1, retries_groq + 1):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                response_text = data["choices"][0]["message"]["content"]
                if sleep_between_requests:
                    time.sleep(sleep_between_requests)
                return OllamaResponse(raw=data, response_text=str(response_text))
            except Exception as exc:
                last_error = exc
                if "429" in str(exc):
                    sleep_time = 15.0 * attempt_num
                    logger.warning("Rate limited by Groq (429). Backing off for %.1fs...", sleep_time)
                    time.sleep(sleep_time)
                else:
                    logger.warning("Groq request failed on attempt %s: %s", attempt_num, exc)
                    time.sleep(min(2.0 * attempt_num, 5.0))
                
        raise RuntimeError(f"Groq request failed after all attempts: {last_error}") from last_error

    url = ollama_url.rstrip("/") + "/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 8192,
        },
    }
    if prefer_json:
        payload["format"] = "json"

    configs_to_try = []
    if model == "qwen2.5-coder:14b":
        # Try 5 times with 14b, varying temp and timeout, then fallback to 7b
        configs_to_try.append({"model": "qwen2.5-coder:14b", "temp": temperature, "timeout": timeout})
        configs_to_try.append({"model": "qwen2.5-coder:14b", "temp": min(temperature + 0.2, 1.0), "timeout": timeout + 60})
        configs_to_try.append({"model": "qwen2.5-coder:14b", "temp": min(temperature + 0.4, 1.0), "timeout": timeout + 120})
        configs_to_try.append({"model": "qwen2.5-coder:14b", "temp": min(temperature + 0.6, 1.0), "timeout": timeout + 180})
        configs_to_try.append({"model": "qwen2.5-coder:14b", "temp": min(temperature + 0.8, 1.0), "timeout": timeout + 240})
        configs_to_try.append({"model": "qwen2.5-coder:7b", "temp": temperature, "timeout": timeout})
    else:
        # Default behavior: try the requested model `retries` times
        for _ in range(retries):
            configs_to_try.append({"model": model, "temp": temperature, "timeout": timeout})

    last_error: Optional[Exception] = None
    attempt_num = 1
    
    for config in configs_to_try:
        current_model = config["model"]
        payload["model"] = current_model
        payload["options"]["temperature"] = config["temp"]
        current_timeout = config["timeout"]
        
        try:
            response = requests.post(url, json=payload, timeout=current_timeout)
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
                "Ollama request failed on attempt %s with model %s (temp=%.2f, timeout=%d): %s",
                attempt_num,
                current_model,
                config["temp"],
                current_timeout,
                exc,
            )
            time.sleep(min(2.0 * attempt_num, 5.0))
        
        if current_model == "qwen2.5-coder:14b" and attempt_num == 5:
            logger.error("Model qwen2.5-coder:14b failed after 5 attempts with varying configs. Falling back to qwen2.5-coder:7b.")
            
        attempt_num += 1

    raise RuntimeError(f"Ollama request failed after all attempts: {last_error}") from last_error

