from __future__ import annotations

import asyncio
from typing import Any

import requests

try:
    from deepeval.models import DeepEvalBaseLLM
except Exception:  # pragma: no cover - fallback for older versions
    from deepeval.models.base_model import DeepEvalBaseLLM


class OllamaEvalModel(DeepEvalBaseLLM):
    """Minimal DeepEval LLM wrapper for a local Ollama server."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def load_model(self) -> str:
        return self.model

    def generate(self, prompt: str, **kwargs: Any) -> str:
        # Debug: show judge prompt sent to Ollama
        print("\n[JUDGE PROMPT]\n" + prompt + "\n")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        # If DeepEval is requesting a structured response, force JSON output.
        if kwargs.get("schema") is not None:
            payload["format"] = "json"

        options = {"temperature": 0}
        user_options = kwargs.get("options") or kwargs.get("ollama_options")
        if isinstance(user_options, dict):
            options.update(user_options)
        payload["options"] = options
        resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        response_text = data.get("response", "")
        # Debug: show judge response from Ollama
        print("[JUDGE RESPONSE]\n" + response_text + "\n")
        return response_text

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.generate, prompt, **kwargs)

    def get_model_name(self) -> str:
        return f"ollama:{self.model}"
