"""Unit tests for chunk.py with a fake word-level tokenizer.

A real tokenizer would require loading the sentence-transformers model, which
makes tests slow and downloads a model on first CI run. The fake here mimics
the HF fast-tokenizer interface (callable returning ``input_ids`` and
``offset_mapping``) just enough to drive the chunking logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from f1_rag.chunk import MIN_BODY_TOKENS, chunk_article
from f1_rag.scrape import ArticleSection, RaceArticle


class WordTokenizer:
    """Whitespace-split tokenizer matching the HF fast-tokenizer call shape."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        offsets: list[tuple[int, int]] = []
        ids: list[int] = []
        i = 0
        for token_id, word in enumerate(text.split()):
            start = text.find(word, i)
            assert start >= 0, f"could not locate {word!r} from offset {i}"
            end = start + len(word)
            offsets.append((start, end))
            ids.append(token_id)
            i = end
        result: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def make_article(
    *,
    season: int = 2022,
    round_: int | None = 13,
    kind: str = "race",
    title: str = "2022 Hungarian Grand Prix",
    sections: list[ArticleSection] | None = None,
) -> RaceArticle:
    return RaceArticle(
        url=f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
        title=title,
        season=season,
        round=round_,
        fetched_at=datetime.now(timezone.utc),
        summary="lede paragraph",
        sections=sections or [],
        full_text="full text",
        kind=kind,  # type: ignore[arg-type]
    )


def long_text(n_words: int) -> str:
    return " ".join(f"word{i}" for i in range(n_words))


@pytest.fixture
def tok() -> WordTokenizer:
    return WordTokenizer()


def test_empty_sections_yields_no_chunks(tok: WordTokenizer) -> None:
    assert chunk_article(make_article(sections=[]), tok) == []


def test_section_below_min_body_tokens_is_dropped(tok: WordTokenizer) -> None:
    short = ArticleSection(title="Pre-race", text=long_text(MIN_BODY_TOKENS - 1))
    assert chunk_article(make_article(sections=[short]), tok) == []


def test_section_within_budget_yields_one_chunk_with_header_and_metadata(
    tok: WordTokenizer,
) -> None:
    section = ArticleSection(title="Race summary", text=long_text(100))
    result = chunk_article(make_article(sections=[section]), tok, chunk_size=300, overlap=50)
    assert len(result) == 1
    c = result[0]
    assert c.id == "2022-13-000"
    # Header surfaces both race and section titles for retrieval.
    assert "2022 Hungarian Grand Prix" in c.text
    assert "Race summary" in c.text
    # Body text is verbatim (original casing/whitespace preserved via offset slicing).
    assert "word0 word1 word2" in c.text
    # Metadata mirrors the article.
    assert c.metadata.season == 2022
    assert c.metadata.round == 13
    assert c.metadata.kind == "race"
    assert c.metadata.section_title == "Race summary"
    assert c.metadata.chunk_index == 0


def test_long_section_splits_into_overlapping_chunks(tok: WordTokenizer) -> None:
    section = ArticleSection(title="Race", text=long_text(800))
    result = chunk_article(make_article(sections=[section]), tok, chunk_size=200, overlap=40)
    assert len(result) >= 3

    # Sequential ids
    assert result[0].id == "2022-13-000"
    assert result[1].id == "2022-13-001"

    # Consecutive chunks share body content via the overlap window
    chunk0_words = set(result[0].text.split())
    chunk1_words = set(result[1].text.split())
    header_words = {"#", "##", "2022", "Hungarian", "Grand", "Prix", "Race"}
    body_overlap = (chunk0_words & chunk1_words) - header_words
    assert any(w.startswith("word") for w in body_overlap), (
        "expected at least one body word to appear in both consecutive chunks"
    )


def test_chunk_indices_are_unique_and_sequential_across_sections(tok: WordTokenizer) -> None:
    sections = [
        ArticleSection(title=name, text=long_text(60))
        for name in ("Background", "Qualifying", "Race")
    ]
    result = chunk_article(make_article(sections=sections), tok, chunk_size=300, overlap=50)
    assert [c.metadata.chunk_index for c in result] == list(range(len(result)))
    assert len({c.id for c in result}) == len(result)


def test_season_summary_uses_season_id_prefix_and_null_round(tok: WordTokenizer) -> None:
    section = ArticleSection(title="Calendar", text=long_text(80))
    article = make_article(
        kind="season",
        round_=None,
        title="2022 Formula One World Championship",
        sections=[section],
    )
    result = chunk_article(article, tok)
    assert len(result) == 1
    assert result[0].id == "2022-season-000"
    assert result[0].metadata.round is None
    assert result[0].metadata.kind == "season"


def test_chunk_size_must_exceed_overlap(tok: WordTokenizer) -> None:
    article = make_article(sections=[ArticleSection(title="X", text=long_text(100))])
    with pytest.raises(ValueError):
        chunk_article(article, tok, chunk_size=50, overlap=60)
