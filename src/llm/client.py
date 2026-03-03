from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import httpx
from dotenv import load_dotenv


@dataclass
class LLMConfig:
    """Generic LLM configuration that can target multiple backends."""

    backend: str
    api_base: str
    api_key: str | None
    text_model: str
    vision_model: str

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # Load .env once before reading environment variables
        load_dotenv()

        backend = os.getenv("LLM_BACKEND", "openrouter").lower()

        if backend == "ollama":
            api_base = os.getenv("LLM_API_BASE", "http://localhost:11434")
            api_key = None  # Ollama is local by default
            text_model = os.getenv("LLM_TEXT_MODEL", "llama3.1:8b")
            vision_model = os.getenv("LLM_VISION_MODEL", "llama3.2-vision")
        else:  # default: OpenRouter / OpenAI-compatible
            api_base = os.getenv(
                "LLM_API_BASE", "https://openrouter.ai/api/v1"
            ).rstrip("/")
            api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY")
            text_model = os.getenv("LLM_TEXT_MODEL", "gpt-4o-mini")
            vision_model = os.getenv("LLM_VISION_MODEL", text_model)

        return cls(
            backend=backend,
            api_base=api_base,
            api_key=api_key,
            text_model=text_model,
            vision_model=vision_model,
        )


def call_text_llm(prompt: str, cfg: LLMConfig) -> str:
    """
    Call a text-only LLM for summarization / reasoning.

    Supported backends:
    - openrouter (OpenAI-compatible /chat/completions)
    - ollama (/api/generate)
    """
    if cfg.backend == "ollama":
        url = f"{cfg.api_base}/api/generate"
        payload = {"model": cfg.text_model, "prompt": prompt, "stream": False}
        with httpx.Client(timeout=60) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return str(data.get("response", "")).strip()

    # Default: OpenRouter / OpenAI-compatible
    url = f"{cfg.api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    body = {
        "model": cfg.text_model,
        "messages": [{"role": "user", "content": prompt}],
    }
    with httpx.Client(timeout=60) as client:
        resp = client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


def call_vision_llm(image_b64: str, prompt: str, cfg: LLMConfig) -> Dict[str, Any]:
    """
    Call a vision-capable LLM that returns JSON extraction for a page.

    The prompt MUST instruct the model to output a pure JSON object with
    fields: text_blocks, tables, figures. We keep that contract here.
    """
    if cfg.backend == "ollama":
        url = f"{cfg.api_base}/api/generate"
        # Many Ollama vision models accept base64 images via the 'images' field.
        combined_prompt = (
            prompt
            + "\n\nRemember: respond with ONLY a valid JSON object, no extra text."
        )
        payload = {
            "model": cfg.vision_model,
            "prompt": combined_prompt,
            "images": [image_b64],
            "stream": False,
        }
        with httpx.Client(timeout=120) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw = str(data.get("response", "")).strip()
            return json.loads(raw)

    # Default: OpenRouter / OpenAI-compatible multimodal chat
    url = f"{cfg.api_base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    payload = {
        "model": cfg.vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

