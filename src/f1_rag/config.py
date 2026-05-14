"""Typed application settings, loaded from environment + .env.

Settings are intentionally read-only and resolved once per process via
``get_settings()``. The function uses ``lru_cache`` so the same instance
is returned everywhere, which keeps callers honest about not mutating
config at runtime — and keeps tests honest, since they can override
fields by instantiating ``Settings(...)`` directly without touching a
global.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "openai", "groq", "ollama"]

# Per-provider default model. Chosen so that flipping ``LLM_PROVIDER``
# alone (without setting ``LLM_MODEL``) produces a working configuration.
_DEFAULT_MODELS: dict[LLMProvider, str] = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2:3b",
}


class Settings(BaseSettings):
    """Process-wide configuration.

    Field names are lowercase; pydantic-settings maps them case-insensitively
    to env vars (e.g. ``OPENAI_API_KEY`` -> ``openai_api_key``).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API credentials. Only the active provider's key is required at runtime.
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None

    # LLM selection. ``llm_model`` is optional; if unset, ``resolved_llm_model``
    # falls back to the provider-appropriate default in ``_DEFAULT_MODELS``.
    llm_provider: LLMProvider = "anthropic"
    llm_model: str | None = None

    # Groq exposes an OpenAI-compatible endpoint -- we reach it via the OpenAI
    # SDK with this base_url override. Configurable in case Groq changes paths.
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # Ollama is reached over HTTP; the default host matches a local install.
    ollama_host: str = "http://localhost:11434"

    # Embeddings run locally via sentence-transformers (no API key needed).
    # BGE-v1.5 small is trained for asymmetric query/passage retrieval and handles
    # question-shaped queries markedly better than MiniLM's symmetric similarity.
    # The asymmetric query prefix is applied automatically in retrieve.py.
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Storage and external APIs.
    chroma_dir: Path = Path("chroma_db")
    jolpica_base_url: str = "https://api.jolpi.ca/ergast/f1/"

    # Wikipedia's API policy expects a descriptive UA with contact info.
    # Edit the placeholder URL / email before doing serious bulk runs.
    wiki_user_agent: str = "f1-rag/0.1 (https://github.com/your-github/f1-rag)"

    @property
    def resolved_llm_model(self) -> str:
        """Return the explicit ``llm_model`` if set, else the provider default."""
        return self.llm_model or _DEFAULT_MODELS[self.llm_provider]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process-wide ``Settings`` instance.

    Call ``get_settings.cache_clear()`` to force a reload (useful in tests
    after mutating environment variables).
    """
    return Settings()
