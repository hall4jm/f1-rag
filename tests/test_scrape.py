"""Unit tests for scrape.py -- no network calls.

We exercise the pure-Python pieces (regex, slug, section flattening,
calendar ordering, disambiguation heuristic) against hand-built fakes
that quack like ``wikipediaapi.WikipediaPage`` / ``WikipediaPageSection``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from f1_rag.scrape import (
    _RACE_TITLE_RE,
    _slugify,
    article_path,
    extract_race_titles,
    flatten_sections,
    is_disambiguation,
)


@dataclass
class FakeSection:
    """Minimal stand-in for wikipediaapi.WikipediaPageSection."""

    title: str
    text: str = ""
    sections: list["FakeSection"] = field(default_factory=list)


@dataclass
class FakePage:
    """Minimal stand-in for wikipediaapi.WikipediaPage."""

    text: str = ""
    summary: str = ""
    links: dict[str, object] = field(default_factory=dict)

    def exists(self) -> bool:
        return True


class TestRaceTitleRegex:
    @pytest.mark.parametrize(
        "title",
        [
            "2022 Hungarian Grand Prix",
            "2018 Brazilian Grand Prix",
            "2023 São Paulo Grand Prix",
            "2024 Las Vegas Grand Prix",
            "2025 Emilia Romagna Grand Prix",
        ],
    )
    def test_matches_real_titles(self, title: str) -> None:
        assert _RACE_TITLE_RE.match(title) is not None

    @pytest.mark.parametrize(
        "title",
        [
            "2022 Italian motorcycle Grand Prix",  # MotoGP
            "Hungarian Grand Prix",  # no year prefix
            "2022 Formula One World Championship",
            "Drivers' Championship",
            "Pierre Gasly",
        ],
    )
    def test_rejects_non_race_titles(self, title: str) -> None:
        assert _RACE_TITLE_RE.match(title) is None


def test_slugify_handles_punctuation_and_unicode() -> None:
    # Non-ASCII chars become dashes -- acceptable for filenames; readers can still
    # find the article via the embedded title field.
    assert _slugify("2022 São Paulo Grand Prix") == "2022-s-o-paulo-grand-prix"
    assert _slugify("2024 Miami Grand Prix!") == "2024-miami-grand-prix"
    assert _slugify("") == "untitled"


def test_article_path_for_race(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = article_path(tmp_path, 2022, 13, "2022 Hungarian Grand Prix")
    assert p == tmp_path / "2022" / "13-2022-hungarian-grand-prix.json"


def test_article_path_for_season_summary(tmp_path) -> None:  # type: ignore[no-untyped-def]
    assert article_path(tmp_path, 2022, None) == tmp_path / "2022" / "season-summary.json"


def test_article_path_requires_title_when_round_set(tmp_path) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        article_path(tmp_path, 2022, 1)


def test_flatten_sections_inlines_subsection_text() -> None:
    tree = [
        FakeSection(
            title="Race",
            text="lede",
            sections=[
                FakeSection(title="Lap 1", text="action one"),
                FakeSection(title="Lap 2", text="action two"),
            ],
        ),
        FakeSection(title="See also", text="ignored"),
    ]
    flat = flatten_sections(tree)  # type: ignore[arg-type]
    assert len(flat) == 1
    assert flat[0].title == "Race"
    # Subsection titles and text are inlined under the parent.
    assert "lede" in flat[0].text
    assert "Lap 1" in flat[0].text
    assert "action two" in flat[0].text


def test_flatten_sections_drops_empty_and_navbox_sections() -> None:
    tree = [
        FakeSection(title="Empty", text=""),
        FakeSection(title="References", text="should be skipped"),
        FakeSection(title="External links", text="should be skipped"),
    ]
    assert flatten_sections(tree) == []  # type: ignore[arg-type]


def test_extract_race_titles_orders_by_text_position() -> None:
    page = FakePage(
        text=(
            "The 2022 Bahrain Grand Prix opened the season. "
            "The 2022 Australian Grand Prix followed."
        ),
        links={
            # Insertion order swapped to prove text-position wins over dict order.
            "2022 Australian Grand Prix": object(),
            "2022 Bahrain Grand Prix": object(),
            "2022 Italian motorcycle Grand Prix": object(),  # filtered out (MotoGP)
            "2021 Bahrain Grand Prix": object(),  # filtered out (wrong year)
            "Pierre Gasly": object(),  # filtered out (not a race)
        },
    )
    result = extract_race_titles(page, 2022)  # type: ignore[arg-type]
    assert result == ["2022 Bahrain Grand Prix", "2022 Australian Grand Prix"]


def test_extract_race_titles_falls_back_to_link_order_for_piped_links() -> None:
    # When a title doesn't appear verbatim (piped link case), it falls to the
    # end of the list, preserving relative order from page.links.
    page = FakePage(
        text="No verbatim race title appears in this prose.",
        links={
            "2022 Bahrain Grand Prix": object(),
            "2022 Saudi Arabian Grand Prix": object(),
        },
    )
    result = extract_race_titles(page, 2022)  # type: ignore[arg-type]
    assert result == ["2022 Bahrain Grand Prix", "2022 Saudi Arabian Grand Prix"]


def test_is_disambiguation_detects_marker() -> None:
    page = FakePage(
        summary="Hungarian Grand Prix may refer to: any of the following Formula One races..."
    )
    assert is_disambiguation(page) is True  # type: ignore[arg-type]


def test_is_disambiguation_negative_on_normal_race_summary() -> None:
    page = FakePage(
        summary="The 2022 Hungarian Grand Prix was a Formula One motor race won by Max Verstappen."
    )
    assert is_disambiguation(page) is False  # type: ignore[arg-type]
