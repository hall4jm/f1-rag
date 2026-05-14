"""Chunking scraped articles into retrieval units.

Strategy:
1. Iterate over an article's sections (already filtered at scrape time --
   no See also / References / etc.).
2. Prepend a context header (``# <race_title>\\n## <section_title>``)
   so each chunk is self-describing for the embedder and the LLM.
3. If the body fits in (chunk_size - header) tokens, emit one chunk.
4. Otherwise slide a token-aligned window over the body, taking
   substrings of the *original* text via char offsets so casing and
   whitespace are preserved (decoding tokens loses both for uncased
   BERT tokenizers like MiniLM's).
5. Drop windows with fewer than ``MIN_BODY_TOKENS`` of content -- they
   add noise to retrieval without adding signal.

Token sizing uses the embedding model's own tokenizer (passed in), not
tiktoken. all-MiniLM-L6-v2 has a hard 512-token max, and cl100k_base
counts ~30% lower than MiniLM tokens for the same text, so sizing with
cl100k risks silent truncation at embed time.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from f1_rag.scrape import RaceArticle

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_CHUNK_SIZE = 480
DEFAULT_OVERLAP = 80
MIN_BODY_TOKENS = 50


class ChunkMetadata(BaseModel):
    """Metadata surfaced as citations and used as filter fields at query time.

    ``round`` defaults to None so we can round-trip through Chroma's metadata
    store: nulls are stripped on write (Chroma rejects them), which means season
    chunks come back without a ``round`` key on read.
    """

    race_title: str
    season: int
    round: int | None = None
    section_title: str
    source_url: str
    chunk_index: int
    kind: str


class Chunk(BaseModel):
    """One retrieval unit: stable id, displayable text, structured metadata."""

    id: str
    text: str
    metadata: ChunkMetadata


def _header(article: RaceArticle, section_title: str) -> str:
    return f"# {article.title}\n## {section_title}\n\n"


def _base_id(article: RaceArticle) -> str:
    """Stable, sortable id prefix for chunks from one article."""
    if article.kind == "season":
        return f"{article.season}-season"
    # round can't be None for kind=='race' (enforced at scrape time).
    assert article.round is not None
    return f"{article.season}-{article.round:02d}"


def chunk_article(
    article: RaceArticle,
    tokenizer: "PreTrainedTokenizerBase",
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Return all chunks for one article, section-first then token-window split.

    ``tokenizer`` must be a fast HuggingFace tokenizer (supports
    ``return_offsets_mapping``); ``SentenceTransformer(...).tokenizer`` qualifies.
    """
    if chunk_size <= overlap:
        raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({overlap})")

    chunks: list[Chunk] = []
    base = _base_id(article)
    chunk_index = 0

    for section in article.sections:
        header = _header(article, section.title)
        header_tokens = tokenizer(header, add_special_tokens=False)["input_ids"]
        body_budget = chunk_size - len(header_tokens)
        if body_budget < MIN_BODY_TOKENS:
            # Header alone eats the budget; chunk would be all metadata.
            continue

        encoded = tokenizer(
            section.text, add_special_tokens=False, return_offsets_mapping=True
        )
        body_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if len(body_ids) < MIN_BODY_TOKENS:
            continue

        # Cap overlap so each window advances by at least one third of its budget;
        # prevents pathological slow progress on tiny budgets.
        body_overlap = min(overlap, body_budget // 3)
        step = body_budget - body_overlap

        i = 0
        while i < len(body_ids):
            end = min(i + body_budget, len(body_ids))
            if end - i < MIN_BODY_TOKENS:
                break  # trailing remainder too small to be useful
            char_start = offsets[i][0]
            char_end = offsets[end - 1][1]
            body_text = section.text[char_start:char_end]

            chunks.append(
                Chunk(
                    id=f"{base}-{chunk_index:03d}",
                    text=header + body_text,
                    metadata=ChunkMetadata(
                        race_title=article.title,
                        season=article.season,
                        round=article.round,
                        section_title=section.title,
                        source_url=article.url,
                        chunk_index=chunk_index,
                        kind=article.kind,
                    ),
                )
            )
            chunk_index += 1

            if end >= len(body_ids):
                break
            i += step

    return chunks


def load_article(path: Path) -> RaceArticle:
    """Load a scraped article JSON from disk into a typed RaceArticle."""
    return RaceArticle.model_validate_json(path.read_text(encoding="utf-8"))
