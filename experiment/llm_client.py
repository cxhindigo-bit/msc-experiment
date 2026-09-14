"""Small cached client for the two prediction pipelines."""

import hashlib
import json

from openai import OpenAI

from .settings import (
    CACHE_DIR,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_THINKING,
)


def _cache_path(stage, cache_key, system, prompt, schema, temperature):
    # Include request-defining settings so changed requests do not reuse old responses.
    fingerprint = json.dumps(
        {
            "model": LLM_MODEL,
            "base_url": LLM_BASE_URL,
            "temperature": temperature,
            "schema": schema,
            "system": system,
            "prompt": prompt,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:20]
    return CACHE_DIR / stage / f"{cache_key}--{digest}.json"


def ask_json(stage, cache_key, system, prompt, schema, temperature):
    path = _cache_path(stage, cache_key, system, prompt, schema, temperature)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": stage, "strict": True, "schema": schema},
        },
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM API returned empty content; check the model and request settings")
    data = json.loads(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def runtime_manifest():
    # Record the model conditions used by initial-answer classification or Pure-LLM.
    return {
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_thinking": LLM_THINKING,
    }
