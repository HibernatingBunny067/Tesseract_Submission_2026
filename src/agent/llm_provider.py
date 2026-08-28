from __future__ import annotations

import os
import json
import httpx
from abc import ABC, abstractmethod
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(
        self, 
        messages: list[dict], 
        system_prompt: str = '', 
        temperature: float = 0.1, 
        json_mode: bool = False
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature.
            json_mode: Whether to enforce JSON output.
            
        Returns:
            The generated text.
        """
        pass


class GeminiProvider(LLMProvider):
    """Provider for Google's Gemini models using the google.genai SDK."""
    
    def __init__(self, api_key: str, model: str | None = None):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise RuntimeError("google.genai package is required for GeminiProvider")
            
        self.client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
        self.types = types

    def generate(
        self, 
        messages: list[dict], 
        system_prompt: str = '', 
        temperature: float = 0.1, 
        json_mode: bool = False
    ) -> str:
        contents = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
            contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
        
        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt
            
        config = self.types.GenerateContentConfig(**config_kwargs)
        
        # Try primary model then fallback models if 503/busy
        models_to_try = [self.model]
        for fallback_m in ["gemini-3.1-pro", "gemini-3.5-flash"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)
                
        last_err = None
        for m in models_to_try:
            try:
                response = self.client.models.generate_content(
                    model=m,
                    contents=contents,
                    config=config
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
    """Provider for Groq models using the raw httpx API."""
    
    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(
        self, 
        messages: list[dict], 
        system_prompt: str = '', 
        temperature: float = 0.1, 
        json_mode: bool = False
    ) -> str:
        try:
            payload_messages = []
            if system_prompt:
                payload_messages.append({"role": "system", "content": system_prompt})
            
            for msg in messages:
                payload_messages.append({"role": msg.get("role"), "content": msg.get("content")})
                
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": payload_messages,
                "temperature": temperature
            }
            if json_mode:
                payload["response_format"] = {"type": "json_object"}

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            with httpx.Client(timeout=15.0) as client:
                response = client.post(self.url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data['choices'][0]['message']['content']
        except Exception as e:
            print(f"[GroqProvider] Error: {e}")
            raise


class OllamaProvider(LLMProvider):
    """Provider for local Ollama models using the raw httpx API."""
    
    def __init__(self, model: str = 'llama3.1:8b', base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/api/chat"

    def generate(
        self, 
        messages: list[dict], 
        system_prompt: str = '', 
        temperature: float = 0.1, 
        json_mode: bool = False
    ) -> str:
        try:
            payload_messages = []
            if system_prompt:
                payload_messages.append({"role": "system", "content": system_prompt})
            
            for msg in messages:
                payload_messages.append({"role": msg.get("role"), "content": msg.get("content")})
                
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": payload_messages,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            if json_mode:
                payload["format"] = "json"
                
            with httpx.Client(timeout=60.0) as client:
                response = client.post(self.url, json=payload)
                response.raise_for_status()
                data = response.json()
                return data['message']['content']
        except Exception as e:
            print(f"[OllamaProvider] Error: {e}")
            raise


def get_provider(role: str = 'default') -> LLMProvider:
    """
    Get the primary LLM provider based on role and available API keys.
    """
    # Try multiple common env variable names
    gemini_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip().strip('"').strip("'")
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    
    providers: list[LLMProvider] = []
    
    def try_add_gemini():
        if gemini_key and len(gemini_key) > 10 and not gemini_key.startswith("YOUR_"):
            try:
                providers.append(GeminiProvider(api_key=gemini_key))
            except Exception:
                pass
                
    def try_add_groq():
        if groq_key and len(groq_key) > 10 and not groq_key.startswith("YOUR_"):
            try:
                providers.append(GroqProvider(api_key=groq_key))
            except Exception:
                pass
            
    def try_add_ollama():
        providers.append(OllamaProvider())
        
    if role == 'materials_advisor':
        try_add_groq()
        try_add_gemini()
        try_add_ollama()
    else:
        try_add_gemini()
        try_add_groq()
        try_add_ollama()
        
    if not providers:
        # Fallback to Ollama
        return OllamaProvider()
        
    return providers[0]


def generate_with_fallback(
    providers: list[LLMProvider], 
    messages: list[dict], 
    system_prompt: str = '', 
    temperature: float = 0.1, 
    json_mode: bool = False
) -> str:
    """
    Tries each provider in order and returns the first successful response.
    Raises RuntimeError if all providers fail.
    """
    for provider in providers:
        try:
            return provider.generate(
                messages=messages, 
                system_prompt=system_prompt, 
                temperature=temperature, 
                json_mode=json_mode
            )
        except Exception:
            continue
            
    raise RuntimeError("All LLM providers failed to generate a response.")
