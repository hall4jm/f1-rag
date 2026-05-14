"""Provider-agnostic LLM wrapper.

One function -- ``generate(system_prompt, user_prompt, *, provider, model, stream, ...)``
-- dispatches to Anthropic, OpenAI, Groq (free tier), or Ollama based on
``provider``. The rest of the codebase never imports the SDKs directly,
so swapping providers is one kwarg.

Groq exposes an OpenAI-compatible API, so it reuses the OpenAI SDK with
a ``base_url`` override -- no extra dependency. That lets the deployed
app run on Groq's free tier (no per-request cost) while local dev can
use Anthropic/OpenAI/Ollama with a one-env-var change.

Streaming returns an ``Iterator[str]`` of text deltas; non-streaming
returns a single ``str``. ``@overload`` makes the return type precise
when ``stream`` is a literal at the call site.

SDK imports are lazy (inside each dispatcher) so an Anthropic-only run
doesn't pay the OpenAI / Ollama import cost.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any, Literal, overload

from pydantic import BaseModel

from f1_rag.config import LLMProvider, Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 1024


class ToolCall(BaseModel):
    """One tool invocation requested by the LLM."""

    id: str  # provider-supplied id; needed to correlate the tool result back
    name: str
    arguments: dict[str, Any]


class ToolCallResponse(BaseModel):
    """LLM's response in a tool-using turn: either final text or pending tool calls.

    Exactly one of ``final_text`` / ``tool_calls`` is meaningful per response.
    """

    final_text: str | None = None
    tool_calls: list[ToolCall] = []


@overload
def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    stream: Literal[False] = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    settings: Settings | None = None,
) -> str: ...


@overload
def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    stream: Literal[True],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    settings: Settings | None = None,
) -> Iterator[str]: ...


def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    stream: bool = False,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    settings: Settings | None = None,
) -> str | Iterator[str]:
    """Run a single-turn chat completion against the configured provider.

    Falls back to ``settings`` (which falls back to env / .env) when ``provider``
    or ``model`` is None. Raises ``RuntimeError`` if the active provider's
    credentials are missing; raises ``ValueError`` for unknown providers.
    """
    s = settings if settings is not None else get_settings()
    p: LLMProvider = provider if provider is not None else s.llm_provider
    m: str = model if model is not None else s.resolved_llm_model

    if p == "anthropic":
        return _generate_anthropic(
            system_prompt, user_prompt, model=m, stream=stream, max_tokens=max_tokens, settings=s
        )
    if p == "openai":
        return _generate_openai(
            system_prompt, user_prompt, model=m, stream=stream, max_tokens=max_tokens, settings=s
        )
    if p == "groq":
        return _generate_groq(
            system_prompt, user_prompt, model=m, stream=stream, max_tokens=max_tokens, settings=s
        )
    if p == "ollama":
        return _generate_ollama(
            system_prompt, user_prompt, model=m, stream=stream, max_tokens=max_tokens, settings=s
        )
    raise ValueError(f"unknown provider: {p!r}")


def _openai_compatible_chat(
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    system_prompt: str,
    user_prompt: str,
    stream: bool,
    max_tokens: int,
) -> str | Iterator[str]:
    """Chat completion against any OpenAI-API-compatible endpoint (OpenAI, Groq, etc.)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if stream:
        def _gen() -> Iterator[str]:
            response = client.chat.completions.create(
                model=model, messages=messages, stream=True, max_tokens=max_tokens
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta is not None:
                    yield delta

        return _gen()

    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens
    )
    return resp.choices[0].message.content or ""


def _generate_anthropic(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    stream: bool,
    max_tokens: int,
    settings: Settings,
) -> str | Iterator[str]:
    from anthropic import Anthropic

    if settings.anthropic_api_key is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set (env or .env)")
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    messages = [{"role": "user", "content": user_prompt}]

    if stream:
        def _gen() -> Iterator[str]:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            ) as s:
                yield from s.text_stream

        return _gen()

    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )
    # Non-tool-use responses contain a single TextBlock; grab its text.
    return msg.content[0].text


def _generate_openai(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    stream: bool,
    max_tokens: int,
    settings: Settings,
) -> str | Iterator[str]:
    if settings.openai_api_key is None:
        raise RuntimeError("OPENAI_API_KEY is not set (env or .env)")
    return _openai_compatible_chat(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=None,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stream=stream,
        max_tokens=max_tokens,
    )


def _generate_groq(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    stream: bool,
    max_tokens: int,
    settings: Settings,
) -> str | Iterator[str]:
    if settings.groq_api_key is None:
        raise RuntimeError(
            "GROQ_API_KEY is not set (env or .env). Get one free at https://console.groq.com."
        )
    return _openai_compatible_chat(
        api_key=settings.groq_api_key.get_secret_value(),
        base_url=settings.groq_base_url,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        stream=stream,
        max_tokens=max_tokens,
    )


def generate_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    settings: Settings | None = None,
) -> ToolCallResponse:
    """Single LLM turn with tool support. OpenAI-shape only (covers openai + groq).

    ``messages`` is the full conversation history in OpenAI format (system / user /
    assistant / tool roles). ``tools`` is the OpenAI-format tool-definition list.

    Returns either ``final_text`` (the LLM produced a natural-language answer) or
    ``tool_calls`` (the LLM wants the caller to execute these and continue the
    conversation). The agent loop in pipeline.answer_agentic owns the iteration.
    """
    s = settings if settings is not None else get_settings()
    p: LLMProvider = provider if provider is not None else s.llm_provider
    m: str = model if model is not None else s.resolved_llm_model

    if p in ("openai", "groq"):
        return _openai_compatible_with_tools(
            messages=messages,
            tools=tools,
            model=m,
            max_tokens=max_tokens,
            api_key=_openai_compatible_key(s, p),
            base_url=s.groq_base_url if p == "groq" else None,
        )
    raise NotImplementedError(
        f"Tool use is not implemented for provider {p!r}. "
        "Use openai or groq for the agentic path; Anthropic / Ollama tool support is future work."
    )


def _openai_compatible_key(settings: Settings, provider: LLMProvider) -> str:
    if provider == "groq":
        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is not set (env or .env)")
        return settings.groq_api_key.get_secret_value()
    if provider == "openai":
        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not set (env or .env)")
        return settings.openai_api_key.get_secret_value()
    raise ValueError(f"_openai_compatible_key called with non-OAI provider {provider!r}")


def _openai_compatible_with_tools(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    api_key: str,
    base_url: str | None,
) -> ToolCallResponse:
    """Single OpenAI-shape chat completion with tool support; non-streaming.

    Two layers of defence against models that emit tool calls as text instead of
    via the structured API:

    1. **Groq's pre-flight rejection** — Groq itself detects malformed tool-call
       text and raises HTTP 400 ``tool_use_failed`` with the offending text in
       ``failed_generation``. We catch that, parse it, and synthesise the right
       tool_calls so the agent loop can continue.
    2. **Soft passthrough** — if the model emits tool-call text and Groq doesn't
       reject it (e.g. the format is just unusual), the post-response branch
       below picks it up from ``msg.content``.
    """
    from openai import BadRequestError, OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )
    except BadRequestError as exc:
        recovered = _recover_from_tool_use_failure(exc)
        if recovered:
            logger.warning(
                "recovered %d tool call(s) from Groq tool_use_failed error: %s",
                len(recovered),
                [tc.name for tc in recovered],
            )
            return ToolCallResponse(tool_calls=recovered)
        raise  # not a tool_use_failed we can rescue -- bubble the original error
    msg = resp.choices[0].message
    raw_tool_calls = getattr(msg, "tool_calls", None) or []

    if raw_tool_calls:
        parsed: list[ToolCall] = []
        for tc in raw_tool_calls:
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Malformed arguments still get surfaced so the loop can report
                # them rather than silently dropping the call.
                arguments = {"__raw__": tc.function.arguments}
            parsed.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
        return ToolCallResponse(tool_calls=parsed)

    # No structured tool calls -- check if the model emitted them as text.
    content = msg.content or ""
    recovered = _extract_embedded_tool_calls(content)
    if recovered:
        logger.warning(
            "recovered %d embedded tool call(s) from text content -- model didn't use "
            "the structured API. Names: %s",
            len(recovered),
            [tc.name for tc in recovered],
        )
        return ToolCallResponse(tool_calls=recovered)

    return ToolCallResponse(final_text=content)


# Patterns we've observed from open models that emit tool calls as text:
#   <function>name({"k":"v"})</function>
#   <function=name>{"k":"v"}</function>
#   <function)name({"k":"v"}</function>     # Llama 3.3 70B on Groq, observed
#   <tool_call>{"name":"X","arguments":{...}}</tool_call>
_EMBEDDED_NAMED_TAG = re.compile(
    r"<function[^a-zA-Z]*(\w+)\s*[(=]?\s*(.*?)</function>",
    re.DOTALL,
)
_EMBEDDED_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)


def _recover_from_tool_use_failure(exc: Exception) -> list[ToolCall]:
    """Extract tool calls from a Groq ``tool_use_failed`` 400 error body.

    Returns an empty list if the exception isn't a recoverable tool-use failure
    (different error code, no failed_generation, or unparseable text).
    """
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return []
    error = body.get("error")
    if not isinstance(error, dict) or error.get("code") != "tool_use_failed":
        return []
    failed_text = error.get("failed_generation")
    if not isinstance(failed_text, str):
        return []
    return _extract_embedded_tool_calls(failed_text)


def _extract_embedded_tool_calls(content: str) -> list[ToolCall]:
    """Recover tool calls emitted as text instead of via the structured API.

    Returns an empty list if no recognisable tool-call pattern is in the text --
    the caller should then treat the content as a final answer.
    """
    calls: list[ToolCall] = []

    # Pattern A: <function ...>name(...args...)</function>
    for match in _EMBEDDED_NAMED_TAG.finditer(content):
        name = match.group(1)
        body = match.group(2)
        args = _extract_first_json_object(body)
        if args is None:
            continue
        calls.append(ToolCall(id=f"recovered_{len(calls)}", name=name, arguments=args))

    # Pattern B: <tool_call>{"name": "X", "arguments": {...}}</tool_call>
    for match in _EMBEDDED_TOOL_CALL_BLOCK.finditer(content):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        arguments = payload.get("arguments") or payload.get("args") or {}
        if name and isinstance(arguments, dict):
            calls.append(ToolCall(id=f"recovered_{len(calls)}", name=name, arguments=arguments))

    return calls


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Find the first balanced ``{ ... }`` block in ``text`` and json.loads it.

    Regex can't balance braces, so we walk the string. Returns None if no valid
    JSON object is found.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    return None
    return None


def _generate_ollama(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    stream: bool,
    max_tokens: int,
    settings: Settings,
) -> str | Iterator[str]:
    import ollama

    client = ollama.Client(host=settings.ollama_host)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    # Ollama's `num_predict` is the equivalent of max_tokens.
    options = {"num_predict": max_tokens}

    if stream:
        def _gen() -> Iterator[str]:
            for part in client.chat(
                model=model, messages=messages, stream=True, options=options
            ):
                content = part["message"]["content"]
                if content:
                    yield content

        return _gen()

    resp = client.chat(model=model, messages=messages, options=options)
    return resp["message"]["content"]
