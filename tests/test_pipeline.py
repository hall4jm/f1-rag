"""Unit tests for pipeline.py pure-logic helpers.

We mock ``retrieve`` and ``generate`` for the ``answer()`` integration check
so tests don't need a populated Chroma collection or an API key.
"""

from __future__ import annotations

import pytest

from f1_rag.chunk import ChunkMetadata
from f1_rag.pipeline import (
    Answer,
    answer,
    build_citations,
    build_user_prompt,
    format_passages,
)
from f1_rag.retrieve import RetrievedChunk


def fake_chunk(
    *,
    text: str = "body",
    race: str = "2022 Hungarian Grand Prix",
    section: str = "Race",
    url: str = "https://en.wikipedia.org/wiki/2022_Hungarian_Grand_Prix",
    season: int = 2022,
    round_: int | None = 13,
    distance: float = 0.4,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=ChunkMetadata(
            race_title=race,
            season=season,
            round=round_,
            section_title=section,
            source_url=url,
            chunk_index=0,
            kind="race" if round_ is not None else "season",
        ),
        distance=distance,
    )


def test_format_passages_numbers_chunks_one_based() -> None:
    chunks = [
        fake_chunk(text="first", race="2022 Hungarian Grand Prix"),
        fake_chunk(text="second", race="2020 Hungarian Grand Prix"),
    ]
    rendered = format_passages(chunks)
    assert rendered.startswith("[1] Source:")
    assert "[2] Source:" in rendered
    assert "first" in rendered
    assert "second" in rendered
    # Each passage block includes its source URL and section.
    assert "section: Race" in rendered


def test_format_passages_empty_returns_empty_string() -> None:
    assert format_passages([]) == ""


def test_build_citations_indexes_from_one_and_preserves_order() -> None:
    chunks = [
        fake_chunk(race="A", section="S1", distance=0.30),
        fake_chunk(race="B", section="S2", distance=0.45),
    ]
    cites = build_citations(chunks)
    assert [c.index for c in cites] == [1, 2]
    assert [c.race_title for c in cites] == ["A", "B"]
    assert [c.section_title for c in cites] == ["S1", "S2"]
    assert cites[0].distance == 0.30


def test_build_user_prompt_includes_question_and_passages() -> None:
    chunks = [fake_chunk(text="some race body text", race="2022 Hungarian Grand Prix")]
    prompt = build_user_prompt("Why did Ferrari lose?", chunks)
    assert "Why did Ferrari lose?" in prompt
    assert "some race body text" in prompt
    assert "[1] Source:" in prompt
    assert prompt.rstrip().endswith("Use [N] inline citations.")


def test_answer_returns_typed_pydantic_with_mocked_retrieve_and_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        fake_chunk(race="2022 Hungarian Grand Prix", section="Race"),
        fake_chunk(race="2020 Hungarian Grand Prix", section="Background", round_=11),
    ]
    monkeypatch.setattr("f1_rag.pipeline.retrieve", lambda *a, **kw: chunks)
    monkeypatch.setattr(
        "f1_rag.pipeline.generate",
        lambda *a, **kw: "Ferrari put Leclerc on hard tyres [1].",
    )

    result = answer("Why did Ferrari lose Hungary 2022?")
    assert isinstance(result, Answer)
    assert result.answer_text == "Ferrari put Leclerc on hard tyres [1]."
    assert len(result.citations) == 2
    assert result.citations[0].index == 1
    assert result.citations[1].race_title == "2020 Hungarian Grand Prix"
    assert len(result.retrieved_chunks) == 2
