"""Multi-provider LLM abstraction layer with automatic fallback.

Supports Google Gemini, Groq (via httpx), and local Ollama.
If no provider is available, the agent graph falls back to
deterministic heuristic-only execution (no LLM calls needed).
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


class LLMProvider(ABC):
    """Base interface for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """Generate a text completion from the model."""
        ...


class GeminiProvider(LLMProvider):
    """Provider for Google Gemini models using the google.genai SDK."""

    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("google.genai package is required for GeminiProvider")

        self.client = genai.Client(api_key=api_key)
        self.model: str = model or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        self.types = types

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})

        config_kwargs: Dict[str, Any] = {"temperature": temperature}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        config = self.types.GenerateContentConfig(**config_kwargs)

        models_to_try: List[str] = [self.model]
        for fallback_m in ["gemini-3.6-flash", "gemini-3.5-flash"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        last_err: Optional[Exception] = None
        for m in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=m, contents=contents, config=config
                )
                if response.text is not None:
                    return response.text
                return ""
            except Exception as e:
                last_err = e
                print(f"[GeminiProvider] Model '{m}' returned: {e}. Trying next model...")
                continue

        print(f"[GeminiProvider] All Gemini models failed. Last error: {last_err}")
        raise last_err or RuntimeError("Gemini generate failed")


class GroqProvider(LLMProvider):
    """Provider for Groq models using the httpx API."""

    def __init__(self, api_key: str, model: Optional[str] = None) -> None:
        if httpx is None:
            raise RuntimeError("httpx package is required for GroqProvider")
        self.api_key: str = api_key
        self.model: str = model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.url: str = "https://api.groq.com/openai/v1/chat/completions"

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        payload_messages: List[Dict[str, str]] = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            payload_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=15.0) as client:
            response = client.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            return str(data["choices"][0]["message"]["content"])


class OllamaProvider(LLMProvider):
    """Provider for local Ollama models using the httpx API."""

    def __init__(
        self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"
    ) -> None:
        if httpx is None:
            raise RuntimeError("httpx package is required for OllamaProvider")
        self.model: str = model
        self.base_url: str = base_url.rstrip("/")
        self.url: str = f"{self.base_url}/api/chat"

    def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        payload_messages: List[Dict[str, str]] = []
        if system_prompt:
            payload_messages.append({"role": "system", "content": system_prompt})
        for msg in messages:
            payload_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        with httpx.Client(timeout=60.0) as client:
            response = client.post(self.url, json=payload)
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            return str(data["message"]["content"])


def get_provider(role: str = "default") -> LLMProvider:
    """Select the primary LLM provider in fallback order: Groq -> Gemini -> Ollama."""
    groq_key: str = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    gemini_key: str = (
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    ).strip().strip('"').strip("'")

    providers: List[LLMProvider] = []

    def try_add_groq() -> None:
        if groq_key and len(groq_key) > 8 and not groq_key.startswith("YOUR_"):
            try:
                providers.append(GroqProvider(api_key=groq_key))
            except Exception:
                pass

    def try_add_gemini() -> None:
        if gemini_key and len(gemini_key) > 8 and not gemini_key.startswith("YOUR_"):
            try:
                providers.append(GeminiProvider(api_key=gemini_key))
            except Exception:
                pass

    def try_add_ollama() -> None:
        providers.append(OllamaProvider())

    # Strict fallback priority: Groq LPU -> Google Gemini -> Local Ollama
    try_add_groq()
    try_add_gemini()
    try_add_ollama()

    if not providers:
        return OllamaProvider()

    return providers[0]


def generate_with_fallback(
    providers: List[LLMProvider],
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    temperature: float = 0.1,
    json_mode: bool = False,
) -> str:
    """Try each provider in order and return the first successful response."""
    for provider in providers:
        try:
            return provider.generate(
                messages=messages,
                system_prompt=system_prompt,
                temperature=temperature,
                json_mode=json_mode,
            )
        except Exception:
            continue

    raise RuntimeError("All LLM providers failed to generate a response.")
