"""Streamlit chat UI for f1-rag.

The UI is intentionally thin: it owns chat state and rendering, but
delegates retrieval + generation to ``f1_rag.pipeline.stream_answer``.
Same code path as the eval harness; what you see is what gets scored.

Visual design notes:
- Dark theme + F1 red accent via .streamlit/config.toml.
- Hero header with corpus stats (live chunk count, hardcoded article count + range).
- Example-question chips render only on first load (no conversation yet).
- Citations render as expandable cards with metadata + a Wikipedia link.
- Sidebar groups settings into LLM / Retrieval / Conversation sections.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal, cast

import streamlit as st


def _bridge_streamlit_secrets_to_env() -> None:
    """Expose Streamlit Cloud secrets as environment variables.

    On Streamlit Cloud, secrets configured in the dashboard are exposed only via
    ``st.secrets``, not as env vars. ``pydantic-settings`` reads env vars, so
    without this bridge our Settings class won't see GROQ_API_KEY etc. Must run
    *before* any module that triggers ``Settings()`` instantiation.

    Safe to call locally: if no secrets.toml exists, ``st.secrets`` access
    raises and we silently fall back to whatever ``.env`` provides.
    """
    try:
        secrets = dict(st.secrets)
    except Exception as exc:  # no secrets file is fine in local dev
        logging.getLogger(__name__).debug("no streamlit secrets to bridge: %s", exc)
        return
    for key, value in secrets.items():
        os.environ.setdefault(key, str(value))


_bridge_streamlit_secrets_to_env()


# Imports below MUST come after the secrets bridge so pydantic-settings sees
# the env vars the first time it loads (Settings() is lru_cached).
from f1_rag.config import LLMProvider, get_settings  # noqa: E402
from f1_rag.pipeline import ToolCallLog, answer_agentic, stream_answer  # noqa: E402
from f1_rag.retrieve import RetrievedChunk, Searcher  # noqa: E402

Mode = Literal["agentic", "v1"]
MODE_LABELS: dict[Mode, str] = {
    "agentic": "Agentic — RAG + Jolpica tools",
    "v1": "Pure RAG (no tools)",
}

logger = logging.getLogger(__name__)

# UI-visible providers. Anthropic and OpenAI remain wired up in generate.py for
# anyone running locally with their own key (set LLM_PROVIDER=anthropic in .env),
# but the deployed app only exposes the providers that work out of the box --
# Groq (free tier, deploy default) and Ollama (fully local).
PROVIDER_OPTIONS: list[LLMProvider] = ["groq", "ollama"]
DEFAULT_MODELS: dict[LLMProvider, str] = {
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2:3b",
}

# Hand-picked starter prompts that show the breadth of the corpus + agentic split.
EXAMPLE_QUESTIONS: list[str] = [
    "Why was Ferrari's tyre strategy at the 2022 Hungarian GP criticised?",
    "Who won the 2022 Hungarian Grand Prix, and from what grid position?",
    "What was porpoising in the 2022 F1 season, and why was it a problem?",
    "How did Mercedes' performance change between 2021 and 2022?",
]

# Light CSS polish. Kept minimal so it survives Streamlit version bumps.
_CUSTOM_CSS = """
<style>
    /* Hide Streamlit's top-right "Deploy" button -- it overlaps the rightmost
       header metric and serves no purpose on an already-deployed app. The
       "..." menu stays so the developer still has rerun / settings / clear-cache. */
    [data-testid="stDeployButton"] { display: none !important; }
    /* Top padding clears the persistent toolbar height (~2.5rem) plus breathing room. */
    .block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1100px; }
    h1 { font-weight: 700; letter-spacing: -0.02em; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .stChatMessage { padding: 0.85rem 1rem; }
    /* citation expanders: subtle red left-border so they read as a sidebar within the bubble */
    div[data-testid="stExpander"] details {
        border-left: 2px solid #E10600;
        border-radius: 0 6px 6px 0;
        background: rgba(225, 6, 0, 0.04);
    }
    /* example-question buttons: full-width, left-aligned, less shouty */
    div[data-testid="column"] button[kind="secondary"] {
        text-align: left;
        white-space: normal;
        height: auto;
        padding: 0.75rem 1rem;
    }
    /* horizontal rule with breathing room */
    hr { margin: 1.5rem 0; opacity: 0.3; }
</style>
"""


@st.cache_resource  # type: ignore[misc]
def _warm_searcher() -> Searcher:
    """Load the embedding model + Chroma collection once per Streamlit process."""
    return Searcher(get_settings())


@st.cache_resource  # type: ignore[misc]
def _corpus_chunk_count() -> int:
    """Live chunk count from the Chroma collection — cheap, runs once at boot."""
    return _warm_searcher().collection.count()


def _init_state() -> None:
    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str, "chunks": list | None}
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def _render_header() -> None:
    """Hero title + corpus stats strip."""
    title_col, s1, s2, s3 = st.columns([3, 1, 1, 1])
    with title_col:
        st.markdown("# BoxBox")
        st.markdown(
            "Hybrid F1 chatbot. "
            "Race results and strategy from **190 Wikipedia articles**, "
            "with factual lookups routed through the **Jolpica F1 API**."
        )
    s1.metric("Articles", "190", help="F1 race + season-summary articles, 2018–2025")
    try:
        s2.metric(
            "Chunks", f"{_corpus_chunk_count():,}",
            help="Retrieval units in the ChromaDB collection",
        )
    except Exception:
        s2.metric("Chunks", "—", help="Collection not built — run scripts/build_index.py")
    s3.metric("Cost / query", "$0.00", help="Free-tier Groq + local embeddings")
    st.divider()


def _render_sidebar() -> tuple[LLMProvider, str, int, Mode]:
    settings = get_settings()
    with st.sidebar:
        st.markdown("## Settings")

        st.caption("Mode")
        mode = cast(
            Mode,
            st.radio(
                "Mode",
                list(MODE_LABELS.keys()),
                index=0,  # agentic default — surfaces the v2 capability
                format_func=lambda m: MODE_LABELS[m],
                label_visibility="collapsed",
                help=(
                    "Agentic mode lets the LLM call the Jolpica F1 API for factual "
                    "lookups (winners, qualifying, standings); pure RAG answers from "
                    "retrieved Wikipedia passages only."
                ),
            ),
        )

        st.divider()

        st.caption("LLM")
        # If settings.llm_provider is set to a provider we no longer surface
        # (anthropic/openai), fall back to the first visible option.
        default_provider_index = (
            PROVIDER_OPTIONS.index(settings.llm_provider)
            if settings.llm_provider in PROVIDER_OPTIONS
            else 0
        )
        provider = cast(
            LLMProvider,
            st.radio(
                "Provider",
                PROVIDER_OPTIONS,
                index=default_provider_index,
                label_visibility="collapsed",
            ),
        )
        model = st.text_input(
            "Model",
            value=DEFAULT_MODELS[provider],
            key=f"model_{provider}",
            help="Override the default model for the selected provider.",
        )

        st.divider()

        st.caption("Retrieval")
        k = st.slider(
            "Top-k chunks",
            min_value=3, max_value=10, value=5,
            help="How many passages to retrieve and pass to the LLM.",
        )

        st.divider()

        st.caption("Conversation")
        if st.button("Clear history", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

        st.divider()

        st.caption("About")
        st.markdown(
            "Embeddings: **BGE-small-en-v1.5** (local).  \n"
            "Vector store: **ChromaDB**.  \n"
            "LLM: provider-agnostic.  \n"
            "Data: Wikipedia (CC BY-SA) + [Jolpica](https://jolpi.ca/) F1 API."
        )
        return provider, model, k, mode


def _render_example_questions() -> None:
    """Starter chips shown only when no conversation has happened yet."""
    st.markdown("##### Try one of these to get started")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(q, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()


def _render_agent_routing(tool_calls: list[ToolCallLog] | None) -> None:
    """Caption + collapsible trace showing how the agent answered (tools vs passages)."""
    if tool_calls is None:
        return  # v1 mode — no routing info to surface
    if not tool_calls:
        st.caption("Routed via Wikipedia passages only — no tools needed for this question.")
        return
    tool_names = ", ".join(f"`{tc.name}`" for tc in tool_calls)
    st.caption(f"Routed via {len(tool_calls)} tool call(s): {tool_names}")
    with st.expander("Tool call details"):
        for tc in tool_calls:
            st.markdown(f"**Round {tc.round_index} · `{tc.name}`**")
            st.code(json.dumps(tc.arguments, indent=2), language="json")
            st.caption(tc.result_summary)
            st.markdown("---")


def _render_citations(chunks: list[RetrievedChunk] | None) -> None:
    """Expandable source cards under an assistant message."""
    if not chunks:
        return
    st.markdown("##### Sources")
    for i, c in enumerate(chunks, start=1):
        label = f"**[{i}]**  {c.metadata.race_title}  ·  *{c.metadata.section_title}*"
        with st.expander(label):
            meta_col, dist_col = st.columns([3, 1])
            round_str = str(c.metadata.round) if c.metadata.round is not None else "—"
            meta_col.markdown(
                f"**Season:** {c.metadata.season}  **·**  "
                f"**Round:** {round_str}  **·**  "
                f"**Kind:** {c.metadata.kind}"
            )
            dist_col.markdown(f"`distance: {c.distance:.3f}`")
            st.markdown("---")
            st.markdown(c.text)
            st.markdown(f"[Open on Wikipedia →]({c.metadata.source_url})")


def _process_question(
    question: str, *, provider: LLMProvider, model: str, k: int, mode: Mode
) -> None:
    """Run one question through the pipeline and append both turns to history."""
    st.session_state.messages.append(
        {"role": "user", "content": question, "chunks": None, "tool_calls": None, "mode": mode}
    )
    with st.chat_message("user"):
        st.markdown(question)

    chunks: list[RetrievedChunk] | None = None
    tool_calls: list[ToolCallLog] | None = None
    text: str = ""

    with st.chat_message("assistant"):
        try:
            if mode == "agentic":
                # v2 is non-streaming (the agent loop must complete before the final
                # answer is known). Spinner + structured trace after the fact.
                with st.spinner("Routing through tools and passages…"):
                    ans = answer_agentic(
                        question, k=k, provider=provider, model=model
                    )
                text = ans.answer_text
                chunks = ans.retrieved_chunks
                tool_calls = ans.tool_calls
                st.markdown(text)
                _render_agent_routing(tool_calls)
                _render_citations(chunks)
            else:
                with st.spinner("Retrieving…"):
                    chunks, stream = stream_answer(
                        question, k=k, provider=provider, model=model
                    )
                text = st.write_stream(stream)
                _render_citations(chunks)
        except Exception as exc:  # surface failures, log full traceback
            logger.exception("query pipeline failed")
            st.error(f"{type(exc).__name__}: {exc}")
            return

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": text,
            "chunks": chunks,
            "tool_calls": tool_calls,
            "mode": mode,
        }
    )


def main() -> None:
    st.set_page_config(
        page_title="BoxBox",
        page_icon="🏎️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    _init_state()
    _warm_searcher()  # first call pays the model-load cost; subsequent are instant

    _render_header()
    provider, model, k, mode = _render_sidebar()

    # Replay any existing history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_agent_routing(msg.get("tool_calls"))
                _render_citations(msg.get("chunks"))

    # Starter chips on first load only
    if not st.session_state.messages and st.session_state.pending_question is None:
        _render_example_questions()

    # Inputs: chat_input or a click on an example chip (queued via session_state)
    typed = st.chat_input("Ask about race results, strategy, controversies…")
    queued = st.session_state.pending_question
    if queued is not None:
        st.session_state.pending_question = None
    question = typed or queued

    if question:
        _process_question(question, provider=provider, model=model, k=k, mode=mode)


if __name__ == "__main__":
    main()
