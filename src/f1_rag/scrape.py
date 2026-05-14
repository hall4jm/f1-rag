"""Wikipedia scraping for Formula 1 race pages and season summaries.

Per-season flow: fetch the season summary page ("YYYY Formula One World
Championship"), harvest links matching "YYYY <Race> Grand Prix" in
document order (for stable round numbering), then fetch each race page.
Results land as JSON under ``data/raw/<season>/``, one file per article.

The structured-section view is what the chunker (phase 3) will consume;
``full_text`` is kept alongside as an escape hatch for linear chunking.

Idempotent by default -- re-running only fetches articles that don't
already have a JSON file on disk. The season summary page is always
re-fetched (we need its links to discover the calendar) but the file
itself is only rewritten when missing or ``--force``.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import requests
import wikipediaapi
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from f1_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Matches "2022 Hungarian Grand Prix", "2022 São Paulo Grand Prix", etc.
# Negative lookahead rejects MotoGP titles like "2022 Italian motorcycle Grand Prix",
# which can incidentally appear in season-page link soup.
_RACE_TITLE_RE = re.compile(r"^(\d{4}) (?!.*motorcycle).+ Grand Prix$")

# Sections that carry no narrative value for the RAG corpus.
_DROP_SECTIONS: frozenset[str] = frozenset(
    {"See also", "References", "Notes", "External links", "Further reading"}
)


class ArticleSection(BaseModel):
    """One top-level section of a Wikipedia article, with subsection text inlined."""

    title: str
    text: str


class RaceArticle(BaseModel):
    """Scraped Wikipedia article for one race or one season summary.

    ``kind`` extends the schema specified in the brief so downstream code
    (chunker, retriever) can distinguish race pages from season summaries
    without having to inspect ``round``.
    """

    url: str
    title: str
    season: int
    round: int | None = Field(
        default=None,
        description="Race order within the season (1-based). None for season summary pages.",
    )
    fetched_at: datetime
    summary: str
    sections: list[ArticleSection]
    full_text: str
    kind: Literal["race", "season"]


def make_client(user_agent: str) -> wikipediaapi.Wikipedia:
    """Build a wikipedia-api client with a polite UA string (required by the library)."""
    return wikipediaapi.Wikipedia(user_agent=user_agent, language="en")


def _collect_section_text(node: wikipediaapi.WikipediaPageSection) -> str:
    """Recursively gather all text under a section, with subsection headings preserved."""
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node.sections:
        child_body = _collect_section_text(child)
        if child_body:
            parts.append(f"\n\n{child.title}\n\n{child_body}")
    return "".join(parts).strip()


def flatten_sections(
    sections: list[wikipediaapi.WikipediaPageSection],
) -> list[ArticleSection]:
    """Flatten the API's recursive section tree to one entry per top-level heading.

    Subsection text is inlined under its parent so each entry is a self-contained
    block of prose -- a good unit for the chunker to operate over.
    """
    flat: list[ArticleSection] = []
    for s in sections:
        if s.title in _DROP_SECTIONS:
            continue
        body = _collect_section_text(s)
        if not body:
            continue
        flat.append(ArticleSection(title=s.title, text=body))
    return flat


@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(4),
    reraise=True,
)
def fetch_page(client: wikipediaapi.Wikipedia, title: str) -> wikipediaapi.WikipediaPage:
    """Fetch a Wikipedia page, retrying transient network errors with backoff.

    404 / missing pages are *not* exceptions in wikipedia-api -- they surface as
    ``page.exists() == False`` and should not enter the retry loop.
    """
    return client.page(title)


def is_disambiguation(page: wikipediaapi.WikipediaPage) -> bool:
    """Heuristic check for disambiguation pages.

    ``wikipedia-api`` doesn't expose page categories cleanly, so we sniff the lede.
    Reliable for the GP article namespace, which has a stable disambiguation style.
    """
    if not page.exists():
        return False
    return "may refer to:" in page.summary[:200].lower()


def extract_race_titles(season_page: wikipediaapi.WikipediaPage, year: int) -> list[str]:
    """Return race article titles for one season, ordered to match the calendar.

    Ordering strategy: first occurrence position in ``season_page.text``. Titles
    that appear only through piped links (e.g. ``[[2022 Hungarian Grand Prix|the
    Hungarian round]]``) won't be found in the rendered prose; those fall to the
    end of the list, preserving their relative order from ``page.links``.
    """
    candidates = [
        title
        for title in season_page.links
        if _RACE_TITLE_RE.match(title) and title.startswith(f"{year} ")
    ]
    body = season_page.text

    found: list[tuple[int, str]] = []
    unfound: list[str] = []
    for title in candidates:
        pos = body.find(title)
        if pos >= 0:
            found.append((pos, title))
        else:
            unfound.append(title)

    found.sort(key=lambda pair: pair[0])
    return [title for _, title in found] + unfound


def _to_article(
    page: wikipediaapi.WikipediaPage,
    season: int,
    round_: int | None,
    kind: Literal["race", "season"],
) -> RaceArticle:
    """Build a typed RaceArticle from a wikipedia-api page object."""
    return RaceArticle(
        url=page.fullurl,
        title=page.title,
        season=season,
        round=round_,
        fetched_at=datetime.now(timezone.utc),
        summary=page.summary,
        sections=flatten_sections(page.sections),
        full_text=page.text,
        kind=kind,
    )


def _slugify(title: str) -> str:
    """Filesystem-safe slug from a Wikipedia title."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "untitled"


def article_path(root: Path, season: int, round_: int | None, title: str | None = None) -> Path:
    """Return the on-disk path for an article (whether or not it has been saved yet)."""
    season_dir = root / str(season)
    if round_ is None:
        return season_dir / "season-summary.json"
    if title is None:
        raise ValueError("title is required when round_ is set")
    return season_dir / f"{round_:02d}-{_slugify(title)}.json"


def save_article(article: RaceArticle, root: Path) -> Path:
    """Persist an article to ``data/raw/<season>/...``. Returns the path written."""
    path = article_path(root, article.season, article.round, article.title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(article.model_dump_json(indent=2), encoding="utf-8")
    return path


def scrape_season(
    year: int,
    root: Path,
    *,
    client: wikipediaapi.Wikipedia,
    force: bool = False,
) -> None:
    """Scrape one season: summary page + all of its race pages.

    The season summary page is always fetched (we need its links to discover the
    calendar) but only written to disk when missing or ``force=True``.
    """
    season_title = f"{year} Formula One World Championship"
    logger.info("season %d: fetching %r", year, season_title)
    season_page = fetch_page(client, season_title)

    if not season_page.exists():
        logger.error("season %d: %r does not exist on Wikipedia; skipping season", year, season_title)
        return

    season_file = article_path(root, year, None)
    if not season_file.exists() or force:
        save_article(_to_article(season_page, season=year, round_=None, kind="season"), root)
    else:
        logger.debug("season %d: summary already cached at %s", year, season_file)

    race_titles = extract_race_titles(season_page, year)
    logger.info("season %d: %d race candidate(s) discovered", year, len(race_titles))

    for round_, title in enumerate(
        tqdm(race_titles, desc=str(year), unit="race", leave=False), start=1
    ):
        target = article_path(root, year, round_, title)
        if target.exists() and not force:
            logger.debug("season %d round %d: cached, skipping %r", year, round_, title)
            continue

        page = fetch_page(client, title)
        if not page.exists():
            logger.warning("season %d round %d: %r missing on Wikipedia", year, round_, title)
            continue
        if is_disambiguation(page):
            logger.warning(
                "season %d round %d: %r is a disambiguation page; skipping", year, round_, title
            )
            continue

        save_article(_to_article(page, season=year, round_=round_, kind="race"), root)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. ``python -m f1_rag.scrape --help`` for usage."""
    parser = argparse.ArgumentParser(
        description="Scrape Wikipedia F1 race + season pages into data/raw/."
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Scrape only this single season (overrides --start-year / --end-year).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-fetch and overwrite existing JSON files."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Output directory root (default: data/raw).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    settings: Settings = get_settings()
    client = make_client(settings.wiki_user_agent)

    if args.season is not None:
        years = [args.season]
    else:
        if args.start_year > args.end_year:
            parser.error("--start-year must be <= --end-year")
        years = list(range(args.start_year, args.end_year + 1))

    for year in years:
        scrape_season(year, args.data_root, client=client, force=args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
