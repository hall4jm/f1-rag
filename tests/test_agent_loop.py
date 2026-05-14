"""Tests for the agent loop in pipeline.answer_agentic.

We mock retrieve(), generate_with_tools(), and the JolpicaClient so the
loop logic is exercised end-to-end without any LLM or HTTP calls.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from f1_rag.chunk import ChunkMetadata
from f1_rag.generate import (
    ToolCall,
    ToolCallResponse,
    _extract_embedded_tool_calls,
    _recover_from_tool_use_failure,
)
from f1_rag.pipeline import answer_agentic
from f1_rag.retrieve import RetrievedChunk


def _fake_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="Some race text.",
        metadata=ChunkMetadata(
            race_title="2022 Hungarian Grand Prix",
            season=2022,
            round=13,
            section_title="Race",
            source_url="https://en.wikipedia.org/wiki/2022_Hungarian_Grand_Prix",
            chunk_index=0,
            kind="race",
        ),
        distance=0.3,
    )


def test_agent_returns_immediately_when_llm_produces_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("f1_rag.pipeline.retrieve", lambda *a, **kw: [_fake_chunk()])
    monkeypatch.setattr(
        "f1_rag.pipeline.generate_with_tools",
        lambda **kw: ToolCallResponse(final_text="The answer."),
    )
    result = answer_agentic("test question", jolpica_client=MagicMock())

    assert result.answer_text == "The answer."
    assert result.rounds_used == 1
    assert result.terminated_early is False
    assert result.tool_calls == []


def test_agent_executes_one_tool_then_returns_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("f1_rag.pipeline.retrieve", lambda *a, **kw: [_fake_chunk()])

    calls = iter(
        [
            ToolCallResponse(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_race_results",
                        arguments={"season": 2022, "round": 13},
                    )
                ]
            ),
            ToolCallResponse(final_text="Max Verstappen won."),
        ]
    )
    monkeypatch.setattr("f1_rag.pipeline.generate_with_tools", lambda **kw: next(calls))

    fake_client = MagicMock()
    captured: list[tuple[str, dict[str, Any]]] = []

    def fake_execute_tool(name: str, args: dict[str, Any], _client: Any) -> dict[str, Any]:
        captured.append((name, args))
        return {"season": args["season"], "round": args["round"], "n_results": 20, "results": []}

    monkeypatch.setattr("f1_rag.pipeline.execute_tool", fake_execute_tool)

    result = answer_agentic("Who won?", jolpica_client=fake_client)

    assert result.answer_text == "Max Verstappen won."
    assert result.rounds_used == 2
    assert result.terminated_early is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_race_results"
    assert result.tool_calls[0].arguments == {"season": 2022, "round": 13}
    assert captured == [("get_race_results", {"season": 2022, "round": 13})]


def test_agent_terminates_early_after_max_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("f1_rag.pipeline.retrieve", lambda *a, **kw: [_fake_chunk()])

    # Always wants another tool call -- never produces final_text.
    def never_final(**kw: Any) -> ToolCallResponse:
        return ToolCallResponse(
            tool_calls=[
                ToolCall(id="x", name="get_race_results", arguments={"season": 2022, "round": 1})
            ]
        )

    monkeypatch.setattr("f1_rag.pipeline.generate_with_tools", never_final)
    monkeypatch.setattr(
        "f1_rag.pipeline.execute_tool", lambda *a, **kw: {"season": 2022, "round": 1, "results": []}
    )

    result = answer_agentic("Loop forever?", max_rounds=3, jolpica_client=MagicMock())

    assert result.terminated_early is True
    assert result.rounds_used == 3
    assert len(result.tool_calls) == 3


class TestExtractEmbeddedToolCalls:
    """The recovery parser for models that emit tool calls as text content."""

    def test_recovers_the_malformed_shape_observed_from_llama33_on_groq(self) -> None:
        # The exact malformed shape we observed in production: opening tag has a
        # stray ')' instead of '>'. Recovery still has to extract name + args.
        content = '<function)get_race_results({"season":2022,"round":13}</function>'
        calls = _extract_embedded_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "get_race_results"
        assert calls[0].arguments == {"season": 2022, "round": 13}

    def test_recovers_standard_function_tag(self) -> None:
        content = '<function>get_qualifying({"season": 2022, "round": 13})</function>'
        calls = _extract_embedded_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "get_qualifying"
        assert calls[0].arguments == {"season": 2022, "round": 13}

    def test_recovers_tool_call_block_form(self) -> None:
        content = (
            '<tool_call>{"name": "get_driver_standings", '
            '"arguments": {"season": 2024}}</tool_call>'
        )
        calls = _extract_embedded_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "get_driver_standings"
        assert calls[0].arguments == {"season": 2024}

    def test_returns_empty_for_normal_prose(self) -> None:
        content = "Max Verstappen won the race from 10th on the grid. It was a damp race."
        assert _extract_embedded_tool_calls(content) == []

    def test_handles_nested_json_arguments(self) -> None:
        content = '<function>q({"filter": {"season": 2022, "round": 13}})</function>'
        calls = _extract_embedded_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].arguments == {"filter": {"season": 2022, "round": 13}}

    def test_recovers_groq_failed_generation_equals_comma_form(self) -> None:
        # Exact failed_generation we observed from Groq's tool_use_failed error
        # for Llama 3.3 70B: opening tag uses '=' before the name and ',' after.
        content = '<function=get_race_results,{"season": 2022, "round": 13}</function>'
        calls = _extract_embedded_tool_calls(content)
        assert len(calls) == 1
        assert calls[0].name == "get_race_results"
        assert calls[0].arguments == {"season": 2022, "round": 13}


class TestRecoverFromToolUseFailure:
    """The BadRequestError -> tool_calls recovery path."""

    @staticmethod
    def _fake_exc(body: dict[str, Any]) -> Exception:
        # OpenAI's BadRequestError exposes the parsed JSON body on `.body`.
        # We don't need the real class -- any exception with .body works.
        exc = Exception("simulated BadRequestError")
        exc.body = body  # type: ignore[attr-defined]
        return exc

    def test_extracts_tool_calls_from_groq_error_body(self) -> None:
        exc = self._fake_exc(
            {
                "error": {
                    "code": "tool_use_failed",
                    "failed_generation": (
                        '<function=get_race_results,{"season": 2022, "round": 13}</function>'
                    ),
                }
            }
        )
        recovered = _recover_from_tool_use_failure(exc)
        assert len(recovered) == 1
        assert recovered[0].name == "get_race_results"
        assert recovered[0].arguments == {"season": 2022, "round": 13}

    def test_returns_empty_for_different_error_code(self) -> None:
        exc = self._fake_exc(
            {"error": {"code": "rate_limit_exceeded", "message": "too many tokens"}}
        )
        assert _recover_from_tool_use_failure(exc) == []

    def test_returns_empty_for_exception_without_body(self) -> None:
        assert _recover_from_tool_use_failure(Exception("no body attr")) == []

    def test_returns_empty_when_failed_generation_isnt_a_tool_call(self) -> None:
        exc = self._fake_exc(
            {"error": {"code": "tool_use_failed", "failed_generation": "just some text"}}
        )
        assert _recover_from_tool_use_failure(exc) == []


def test_agent_records_tool_error_as_error_in_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("f1_rag.pipeline.retrieve", lambda *a, **kw: [_fake_chunk()])
    calls = iter(
        [
            ToolCallResponse(
                tool_calls=[ToolCall(id="z", name="boom", arguments={"x": 1})]
            ),
            ToolCallResponse(final_text="Sorry, the tool failed."),
        ]
    )
    monkeypatch.setattr("f1_rag.pipeline.generate_with_tools", lambda **kw: next(calls))

    def explode(*_args: Any, **_kw: Any) -> dict[str, Any]:
        raise RuntimeError("simulated tool failure")

    monkeypatch.setattr("f1_rag.pipeline.execute_tool", explode)

    result = answer_agentic("trigger an error", jolpica_client=MagicMock())

    assert result.terminated_early is False
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].result_summary.startswith("ERROR:")
    assert "simulated tool failure" in result.tool_calls[0].result_summary
