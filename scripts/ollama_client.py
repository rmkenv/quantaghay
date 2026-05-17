"""
QuantAgri Hay — Ollama Cloud Client
Endpoint: https://ollama.com/api/chat
Auth:     Authorization: Bearer $OLLAMA_API_KEY

Confirmed available models:
  gpt-oss:20b        <- default
  gpt-oss:120b
  gemma3:4b
  gemma3:12b
  gemma4:31b
  deepseek-v3.1:671b
"""

import json
import os
import time
from typing import Any

import requests

ENDPOINT    = "https://ollama.com/api/chat"
MAX_RETRIES = 3
RETRY_DELAY = 8


def _get_key() -> str:
    key = os.getenv("OLLAMA_API_KEY", "")
    if not key:
        raise RuntimeError(
            "OLLAMA_API_KEY not set. "
            "Add it as a GitHub Actions secret."
        )
    return key


def chat(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.2,
    as_json: bool = True,
    system: str | None = None,
) -> str:
    model = model or os.getenv("OLLAMA_MODEL", "gpt-oss:20b")
    key   = _get_key()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": temperature},
    }
    if as_json:
        payload["format"] = "json"

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {key}",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)

            if resp.status_code == 404:
                raise RuntimeError(
                    f"Model '{model}' not found on Ollama Cloud API.\n"
                    f"Available: gpt-oss:20b, gpt-oss:120b, gemma3:4b, "
                    f"gemma3:12b, gemma4:31b, deepseek-v3.1:671b\n"
                    f"Check: https://ollama.com/api/tags"
                )
            if resp.status_code in (401, 403):
                raise RuntimeError("Invalid or expired OLLAMA_API_KEY")

            resp.raise_for_status()
            return resp.json()["message"]["content"]

        except RuntimeError:
            raise
        except requests.HTTPError as e:
            print(f"  [HTTP {resp.status_code}] attempt {attempt}/{MAX_RETRIES}: {e}")
        except requests.RequestException as e:
            print(f"  [NET ERR] attempt {attempt}/{MAX_RETRIES}: {e}")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(f"Ollama Cloud failed after {MAX_RETRIES} attempts")


def chat_json(prompt: str, model: str | None = None, system: str | None = None) -> dict:
    raw     = chat(prompt, model=model, as_json=True, system=system)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:]).rstrip("`").strip()
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON in response:\n{raw[:300]}")
    return json.loads(cleaned[start:end])
