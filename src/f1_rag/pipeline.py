"""End-to-end RAG pipeline: question -> retrieval -> LLM -> typed Answer.

Two entry points:
- ``answer(question)`` -> ``Answer`` -- non-streaming, used by the eval
  harness and scripts.
- ``stream_answer(question)`` -> ``(chunks, text_iterator)`` -- retrieval
  runs synchronously so the UI can render citation expanders the moment
  retrieval finishes; the iterator yields LLM text deltas into the chat
  bubble. Same path, same prompt, just streamed.

The prompt template lives as a module constant so it's one obvious place
to read and tune.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel

from f1_rag.config import LLMProvider, Settings, get_settings
from f1_rag.generate import generate, generate_with_tools
from f1_rag.jolpica import JolpicaClient
from f1_rag.retrieve import RetrievedChunk, retrieve
from f1_rag.tools import TOOL_DEFINITIONS, execute_tool

logger = logging.getLogger(__name__)

DEFAULT_AGENTIC_MAX_ROUNDS = 5

SYSTEM_PROMPT = """You are an F1 historian assistant. You answer questions about Formula 1 races, drivers, teams, strategy, and history using ONLY the retrieved Wikipedia passages provided to you.

Rules:
- Cite sources inline using [N] markers, where N is the passage number. Cite the specific passage that supports each claim.
- Multiple citations per claim are fine: "Verstappen won the race [1][3]."
- If the passages do not contain enough information to answer, say "I don't have enough information to answer that confidently from the sources I have." Do NOT guess or fill in from outside the passages.
- Keep answers proportionate to the question. Don't pad.
- Distinguish facts (from passages) from analysis where natural ("the article describes the strategy as controversial").
"""

USER_PROMPT_TEMPLATE = """Question: {question}

Retrieved passages:

{passages}

Answer using only these passages. Use [N] inline citations."""


AGENTIC_SYSTEM_PROMPT = """You are an F1 historian assistant with two information sources:

1. RETRIEVED PASSAGES — Wikipedia text covering 2018–2025 F1 races, numbered [1], [2], ...
   Best for narrative, strategy, controversies, technical explanations, season context.

2. TOOLS — a structured F1 data API (Jolpica/Ergast).
   Best for factual lookups: race results, qualifying, championship standings, calendars.

**HOW TO CALL TOOLS — critical:** Use the structured tool-call mechanism the system
provides. Do NOT write `<function>...</function>` tags, pseudo-XML, JSON blocks, or any
function-call syntax in your response text — such text is never executed and will be
shown verbatim to the user, breaking the conversation. If you decide a tool is needed,
emit only the structured tool call; the system will run it and return the result, and
you can then write the final answer in plain prose.

Rules:
- For factual lookups (who won, what time, what grid, what position), call the appropriate tool.
- For narrative or analytical questions (why, how, controversies), use the passages.
- For mixed questions, use both. Multiple tool calls in one turn are fine.
- Cite passages inline with [N] markers as before.
- When you reference tool results, write "(via Jolpica F1 API)" — do not invent [N] markers for tool data.
- If neither tools nor passages have the information, say "I don't have enough information to answer that confidently from my sources."
- Be decisive: once a tool returns the answer, use it. Don't re-call the same tool with the same args.
"""


AGENTIC_USER_PROMPT_TEMPLATE = """Question: {question}

Retrieved passages:

{passages}

You also have tools available. Use them for factual lookups; use the passages for narrative."""


class Citation(BaseModel):
    """One numbered citation for the UI's expandable-source panel."""

    index: int  # 1-based, matches [N] markers in the answer text
    race_title: str
    section_title: str
    source_url: str
    distance: float


class Answer(BaseModel):
    """Full non-streaming RAG response."""

    answer_text: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]


class ToolCallLog(BaseModel):
    """One executed tool call, recorded for the writeup / debug view."""

    round_index: int  # 1-based
    name: str
    arguments: dict[str, Any]
    result_summary: str  # short string for human review; full result fed to LLM


class AgenticAnswer(BaseModel):
    """v2 response: same shape as ``Answer`` plus the tool-call trace."""

    answer_text: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk]
    tool_calls: list[ToolCallLog]
    rounds_used: int  # how many LLM turns the agent loop took
    terminated_early: bool  # True iff hit max_rounds without a final answer


def format_passages(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered block the LLM will reference."""
    blocks = [
        f"[{i}] Source: {c.metadata.source_url} (section: {c.metadata.section_title})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """1-based citation list matching the [N] markers handed to the LLM."""
    return [
        Citation(
            index=i,
            race_title=c.metadata.race_title,
            section_title=c.metadata.section_title,
            source_url=c.metadata.source_url,
            distance=c.distance,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Public helper: same prompt the pipeline sends to the LLM (useful for eval / debug)."""
    return USER_PROMPT_TEMPLATE.format(question=question, passages=format_passages(chunks))


def answer(
    question: str,
    *,
    k: int = 5,
    where: dict[str, Any] | None = None,
    provider: LLMProvider | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> Answer:
    """Run the full RAG pipeline non-streaming. Returns a complete ``Answer``."""
    s = settings if settings is not None else get_settings()
    chunks = retrieve(question, k=k, where=where)
    user_prompt = build_user_prompt(question, chunks)
    text = generate(
        SYSTEM_PROMPT,
        user_prompt,
        provider=provider,
        model=model,
        settings=s,
    )
    return Answer(
        answer_text=text,
        citations=build_citations(chunks),
        retrieved_chunks=chunks,
    )


def answer_agentic(
    question: str,
    *,
    k: int = 5,
    where: dict[str, Any] | None = None,
    max_rounds: int = DEFAULT_AGENTIC_MAX_ROUNDS,
    provider: LLMProvider | None = None,
    model: str | None = None,
    settings: Settings | None = None,
    jolpica_client: JolpicaClient | None = None,
) -> AgenticAnswer:
    """v2 RAG: same retrieval, plus tool calls into the Jolpica F1 API.

    Hybrid by design — retrieval runs first (k chunks) and the LLM sees both the
    passages and the tool list. It decides whether to call a tool, get the
    structured answer, or rely on the passages. Loops up to ``max_rounds`` times
    in case the LLM chains tool calls.
    """
    s = settings if settings is not None else get_settings()
    client = jolpica_client if jolpica_client is not None else JolpicaClient(s)

    chunks = retrieve(question, k=k, where=where)
    passages = format_passages(chunks)
    initial_user = AGENTIC_USER_PROMPT_TEMPLATE.format(question=question, passages=passages)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENTIC_SYSTEM_PROMPT},
        {"role": "user", "content": initial_user},
    ]
    tool_log: list[ToolCallLog] = []

    for round_idx in range(1, max_rounds + 1):
        logger.info("agent round %d: calling LLM with %d message(s)", round_idx, len(messages))
        resp = generate_with_tools(
            messages=messages,
            tools=TOOL_DEFINITIONS,
            provider=provider,
            model=model,
            settings=s,
        )

        if resp.final_text is not None and not resp.tool_calls:
            logger.info("agent round %d: produced final answer (%d chars)", round_idx, len(resp.final_text))
            return AgenticAnswer(
                answer_text=resp.final_text,
                citations=build_citations(chunks),
                retrieved_chunks=chunks,
                tool_calls=tool_log,
                rounds_used=round_idx,
                terminated_early=False,
            )

        # The LLM wants to call tools. Record the assistant turn in OpenAI shape,
        # then execute each call and append the tool-role results.
        messages.append(_assistant_tool_request_message(resp.tool_calls))

        for call in resp.tool_calls:
            logger.info(
                "agent round %d: tool=%s args=%s", round_idx, call.name, call.arguments
            )
            try:
                result = execute_tool(call.name, call.arguments, client)
                error: str | None = None
            except Exception as exc:  # surface tool failures to the LLM rather than crashing
                logger.warning("tool %r raised %s", call.name, exc)
                result = {"error": f"{type(exc).__name__}: {exc}"}
                error = str(exc)

            tool_log.append(
                ToolCallLog(
                    round_index=round_idx,
                    name=call.name,
                    arguments=call.arguments,
                    result_summary=(
                        f"ERROR: {error}" if error else _summarise_tool_result(call.name, result)
                    ),
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, default=str),
                }
            )

    # Hit max_rounds without a final answer. Return whatever the LLM said last
    # alongside a flag so the caller knows it was truncated.
    logger.warning("agent hit max_rounds=%d without final_text", max_rounds)
    return AgenticAnswer(
        answer_text=(
            "(Agent loop reached max_rounds without producing a final answer. "
            "See tool_calls for what was attempted.)"
        ),
        citations=build_citations(chunks),
        retrieved_chunks=chunks,
        tool_calls=tool_log,
        rounds_used=max_rounds,
        terminated_early=True,
    )


def _assistant_tool_request_message(tool_calls: list[Any]) -> dict[str, Any]:
    """Format an assistant turn whose body is a list of pending tool calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in tool_calls
        ],
    }


def _summarise_tool_result(name: str, result: dict[str, Any]) -> str:
    """One-line preview of a tool's return value for the tool-call log."""
    if name in {"get_race_results", "get_qualifying"}:
        season = result.get("season")
        round_ = result.get("round")
        n = result.get("n_results", 0)
        return f"{name}({season=}, {round_=}) -> {n} entries"
    if name in {"get_driver_standings", "get_constructor_standings", "get_race_schedule"}:
        season = result.get("season")
        key = "standings" if "standings" in result else "schedule"
        n = len(result.get(key, []))
        return f"{name}({season=}) -> {n} entries"
    if name == "get_driver_info":
        return f"get_driver_info({result.get('driver_id')!r}) -> found={result.get('found')}"
    return f"{name} -> {list(result.keys())}"


def stream_answer(
    question: str,
    *,
    k: int = 5,
    where: dict[str, Any] | None = None,
    provider: LLMProvider | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> tuple[list[RetrievedChunk], Iterator[str]]:
    """Streaming variant for the UI.

    Retrieval is synchronous and returns first so the caller can render
    citation expanders immediately. The returned iterator yields LLM
    text deltas.
    """
    s = settings if settings is not None else get_settings()
    chunks = retrieve(question, k=k, where=where)
    user_prompt = build_user_prompt(question, chunks)
    stream = generate(
        SYSTEM_PROMPT,
        user_prompt,
        provider=provider,
        model=model,
        stream=True,
        settings=s,
    )
    return chunks, stream
