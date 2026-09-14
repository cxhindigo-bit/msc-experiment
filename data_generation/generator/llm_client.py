import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .config import (
    DIALOGUE_INITIAL_ANSWER_MODE,
    DIALOGUE_MAX_TOKENS,
    DIALOGUE_TEMPERATURE,
    DIALOGUE_THINKING,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data_generation" / "cache"
load_dotenv(PROJECT_ROOT / ".env")

LLM_API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "")).strip()

# Design item: Dialogue-generation model
# Current setting: gpt-4.1-nano by default, independently configurable from the prediction pipelines.
LLM_MODEL = os.environ.get("DIALOGUE_LLM_MODEL", "gpt-4.1-nano").strip()

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").strip()

def _client():
    if not LLM_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY (or LLM_API_KEY) before generating dialogues")
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


def _cache_path(system, prompt, cache_key, response_schema, temperature):
    # Include request-defining settings so a changed request uses a different cache file.
    schema_text = json.dumps(response_schema, sort_keys=True)

    digest = hashlib.sha256(
        f"{LLM_MODEL}\n{LLM_BASE_URL}\n{temperature}\n{DIALOGUE_MAX_TOKENS}\n{schema_text}\n{system}\n{prompt}".encode()
    ).hexdigest()[:20]

    return CACHE_DIR / f"{cache_key}--{digest}.json"


def ask_json(system, prompt, cache_key, response_schema, temperature):
    path = _cache_path(system, prompt, cache_key, response_schema, temperature)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    # Send the configured model, schema, temperature, and output limit as one request.
    response = _client().chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "strict": True,
                "schema": response_schema,
            },
        },
        temperature=temperature,
        max_tokens=DIALOGUE_MAX_TOKENS,
    )
    data = json.loads(response.choices[0].message.content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def llm_manifest():
    # Record the dialogue-generation settings used for the saved dataset.
    return {
        "llm_backend": "openai",
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_api_style": "chat_completions",
        "llm_json_mode": "json_schema",
        "llm_temperature": DIALOGUE_TEMPERATURE,
        "llm_max_tokens": DIALOGUE_MAX_TOKENS,
        "llm_thinking": DIALOGUE_THINKING,
        "dialogue_initial_answer_mode": DIALOGUE_INITIAL_ANSWER_MODE,
    }
