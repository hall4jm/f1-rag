"""Smoke tests proving the package imports and config loads end to end.

These exist so CI catches a broken import path or a Settings field
that doesn't parse, before any of the real modules are wired up.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import f1_rag
from f1_rag.config import LLMProvider, Settings, get_settings

VALID_PROVIDERS = set(get_args(LLMProvider))


def test_package_has_version() -> None:
    assert isinstance(f1_rag.__version__, str)
    assert f1_rag.__version__


def test_settings_load_with_defaults() -> None:
    # _env_file=None bypasses the user's local .env so we verify the package's
    # actual class defaults, not whatever happens to be in dev config.
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.llm_provider in VALID_PROVIDERS
    assert s.resolved_llm_model  # non-empty for any provider
    assert s.embedding_model == "BAAI/bge-small-en-v1.5"
    assert isinstance(s.chroma_dir, Path)


def test_resolved_llm_model_falls_back_per_provider() -> None:
    def s(**kw: object) -> Settings:
        return Settings(_env_file=None, **kw)  # type: ignore[call-arg]

    # Explicit override wins.
    assert s(llm_provider="anthropic", llm_model="custom").resolved_llm_model == "custom"
    # Each provider has a working default.
    assert s(llm_provider="anthropic").resolved_llm_model.startswith("claude")
    assert s(llm_provider="openai").resolved_llm_model.startswith("gpt")
    assert s(llm_provider="groq").resolved_llm_model.startswith("llama")
    assert s(llm_provider="ollama").resolved_llm_model.startswith("llama")


def test_every_provider_has_a_default_model() -> None:
    # Guards against forgetting to add a default when a new provider lands.
    for p in VALID_PROVIDERS:
        assert Settings(_env_file=None, llm_provider=p).resolved_llm_model, (  # type: ignore[call-arg]
            f"no default model for provider {p!r}"
        )


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_query_instruction_returns_prefix_for_bge_models() -> None:
    from f1_rag.retrieve import query_instruction_for

    bge_prefix = "Represent this sentence for searching relevant passages: "
    assert query_instruction_for("BAAI/bge-small-en-v1.5") == bge_prefix
    assert query_instruction_for("BAAI/bge-base-en-v1.5") == bge_prefix
    assert query_instruction_for("BAAI/bge-large-en-v1.5") == bge_prefix


def test_query_instruction_returns_empty_for_symmetric_models() -> None:
    from f1_rag.retrieve import query_instruction_for

    assert query_instruction_for("sentence-transformers/all-MiniLM-L6-v2") == ""
    assert query_instruction_for("some-other-symmetric-model") == ""
