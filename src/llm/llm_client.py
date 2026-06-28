"""
Multi-provider LLM client.
Supports OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini), and Ollama.
Uses a factory pattern so the rest of the codebase is provider-agnostic.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger


# ── Base ───────────────────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    def __init__(
        self,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> None:
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Send a chat turn and return the assistant response as a string.

        Args:
            system_prompt: High-level system instructions.
            user_message:  The current user turn (may include RAG context).
            history:       Optional list of previous turns in
                           [{"role": "user"|"assistant", "content": str}, …] format.
        """


# ── OpenAI ─────────────────────────────────────────────────────────────────

class OpenAIClient(BaseLLMClient):
    """OpenAI compatible chat models."""

    def __init__(self, model: str = "gpt-4o", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        from openai import OpenAI
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @property
    def provider_name(self) -> str:
        return "openai"

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content


# ── Anthropic ──────────────────────────────────────────────────────────────

class AnthropicClient(BaseLLMClient):
    """Anthropic Claude models."""

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[dict] = []
        if history:
            # Anthropic uses the same role schema as OpenAI
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        resp = self._client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.content[0].text


# ── Google Gemini ──────────────────────────────────────────────────────────

class GoogleClient(BaseLLMClient):
    """Google Gemini models via the generativeai SDK."""

    def __init__(self, model: str = "gemini-1.5-pro", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self._genai = genai

    @property
    def provider_name(self) -> str:
        return "google"

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
            generation_config=self._genai.types.GenerationConfig(
                max_output_tokens=self.max_tokens,
                temperature=self.temperature,
            ),
        )
        # Convert OpenAI-style history to Gemini format
        gemini_history = []
        if history:
            for msg in history:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat_session = model.start_chat(history=gemini_history)
        return chat_session.send_message(user_message).text

# ── Meta Ollama ──────────────────────────────────────────────────────────

class OllamaClient(BaseLLMClient):
    """Ollama models via the ollama SDK."""

    def __init__(self, model: str = "llama3.1", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        import ollama
        self._client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

    @property
    def provider_name(self) -> str:
        return "ollama"

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )

        if isinstance(resp, dict):
            return resp["message"]["content"]
        return resp.message.content


# ── Factory ────────────────────────────────────────────────────────────────

class LLMFactory:
    """Create the correct BaseLLMClient for a given provider name."""

    _REGISTRY: dict[str, type[BaseLLMClient]] = {
        "openai":    OpenAIClient,
        "anthropic": AnthropicClient,
        "google":    GoogleClient,
        "ollama":    OllamaClient,
    }

    _DEFAULTS: dict[str, str] = {
        "openai":    "gpt-4o",
        "anthropic": "claude-sonnet-4-6",
        "google":    "gemini-1.5-pro",
        "ollama":    "llama3.2:1b",
    }

    @classmethod
    def create(
        cls,
        provider: str,
        config: dict[str, Any] | None = None,
    ) -> BaseLLMClient:
        provider = provider.lower()
        if provider not in cls._REGISTRY:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Available: {list(cls._REGISTRY)}"
            )
        cfg   = config or {}
        model = cfg.get("model", cls._DEFAULTS[provider])
        klass = cls._REGISTRY[provider]
        logger.info(f"LLM client: provider={provider} model={model}")
        return klass(
            model=model,
            max_tokens=cfg.get("max_tokens", 2048),
            temperature=cfg.get("temperature", 0.1),
        )

    @classmethod
    def available_providers(cls) -> list[str]:
        return list(cls._REGISTRY.keys())
