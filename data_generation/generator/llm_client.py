import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .config import DIALOGUE_TEMPERATURE


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data_generation" / "cache"
load_dotenv(PROJECT_ROOT / ".env")

# LM Studio 的 OpenAI 兼容接口不要求真实密钥，但 OpenAI 客户端要求非空字符串
# LM Studio does not require a real key locally, but the OpenAI client requires a non-empty value.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lm-studio").strip()

# 默认值对应实验电脑上 LM Studio 显示的 Qwen3.5 9B 模型标识。
# The default matches the Qwen3.5 9B model identifier shown by LM Studio.
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.5-9b").strip()

# LM Studio 默认在本机 1234 端口提供 OpenAI 兼容接口
# LM Studio serves its OpenAI-compatible API on local port 1234 by default.
LLM_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "http://127.0.0.1:1234/v1"
).strip()

# 客户端结构根据 LM Studio 官方示例改写，并非逐行复制：
# The client structure is adapted from, not copied verbatim from, LM Studio's example:
# https://github.com/lmstudio-ai/docs/blob/b02d17517b73c51f520cd5129855cdf30e0728f7/1_developer/3_openai-compat/chat-completions.md
CLIENT = OpenAI(api_key=LLM_API_KEY or "lm-studio", base_url=LLM_BASE_URL)


def _cache_path(system, prompt, cache_key, response_schema, temperature):
    # 为支持中断后继续生成而增加的缓存
    # This resumable cache is project-specific; it is not required by the LM Studio API.
    # “请求参数组成缓存键，再用 SHA-256 生成固定键”的思路参照 LiteLLM；
    # 本函数只保留当前实验需要的字段，并非复制 LiteLLM 的实现。
    # The request fingerprint and SHA-256 key are adapted from LiteLLM's design;
    # this smaller function is project-specific rather than a copied implementation.
    # https://github.com/BerriAI/litellm/blob/3f4810b8f225dfbfa182c9d9bf0990ce12f861f1/litellm/caching/caching.py#L315-L459
    # 将 Schema 按固定字段顺序转成文字，避免相同 Schema 因字段顺序不同而产生不同结果
    # Serialize the schema with stable key ordering so equivalent schemas produce the same text.
    schema_text = json.dumps(response_schema, sort_keys=True)

    # 使用所有会影响输出的设置生成配置指纹。模型、接口、temperature、Schema
    # 或 Prompt 中任何一项改变，都会产生新的指纹，旧缓存就不会被误用
    # Build a fingerprint from every setting that affects the output. Changing the model,
    # endpoint, temperature, schema, or prompt creates a new fingerprint and cache entry.
    digest = hashlib.sha256(
        f"{LLM_MODEL}\n{LLM_BASE_URL}\n{temperature}\n{schema_text}\n{system}\n{prompt}".encode()
    ).hexdigest()[:20]

    # cache_key 让文件名可以识别到具体会话，digest 区分该会话的不同生成配置
    # cache_key identifies the session; digest separates configurations for that session.
    # 示例 / Example: mastery-v5-L001_S01--7d41a82f913c8c52a420.json
    return CACHE_DIR / f"{cache_key}--{digest}.json"


def ask_json(system, prompt, cache_key, response_schema, temperature):
    path = _cache_path(system, prompt, cache_key, response_schema, temperature)
    # “文件存在则读取，否则请求后写入”的思路参照 OpenAI tiktoken 的本地缓存；
    # 当前 JSON 会话缓存是针对本实验重新实现的简化版本
    # The read-on-hit and write-on-miss idea is adapted from OpenAI tiktoken;
    # this JSON session cache is a simplified project-specific implementation.
    # https://github.com/openai/tiktoken/blob/4e71bbe0c078468e00fefbf94b39849389f346e5/tiktoken/load.py
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    # 请求字段和 JSON Schema 格式根据 LM Studio 官方示例改写：
    # The request fields and JSON Schema format are adapted from LM Studio's example:
    # https://github.com/lmstudio-ai/docs/blob/b02d17517b73c51f520cd5129855cdf30e0728f7/1_developer/3_openai-compat/structured-output.md
    response = CLIENT.chat.completions.create(
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
    )
    data = json.loads(response.choices[0].message.content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def llm_manifest():
    # 复现信息
    # project-specific reproducibility record
    return {
        "llm_backend": "lm_studio",
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_api_style": "chat_completions",
        "llm_json_mode": "json_schema",
        "llm_temperature": DIALOGUE_TEMPERATURE,
    }
