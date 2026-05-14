"""Build (or rebuild) the f1_races Chroma collection from data/raw/.

Loads every scraped article, chunks it with the embedding model's own
tokenizer (so chunks fit MiniLM's 512-token max), embeds in a single
batched pass, and upserts into a persistent Chroma collection.

Usage::

    uv run python scripts/build_index.py --mode rebuild
    uv run python scripts/build_index.py --mode append --data-root data/raw/2024
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import tiktoken
from tqdm import tqdm

from f1_rag.chunk import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    Chunk,
    chunk_article,
    load_article,
)
from f1_rag.config import get_settings
from f1_rag.embed import (
    DEFAULT_BATCH_SIZE,
    embed_texts,
    get_or_reset_collection,
    load_embedding_model,
    make_chroma_client,
    upsert_chunks,
)

logger = logging.getLogger(__name__)


def collect_chunks(
    data_root: Path,
    tokenizer: object,
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[int, list[Chunk]]:
    """Walk ``data_root`` for *.json articles, return (n_articles, all_chunks)."""
    json_paths = sorted(data_root.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"no scraped articles found under {data_root}")

    chunks: list[Chunk] = []
    for path in tqdm(json_paths, desc="chunking", unit="article"):
        article = load_article(path)
        chunks.extend(
            chunk_article(article, tokenizer, chunk_size=chunk_size, overlap=overlap)  # type: ignore[arg-type]
        )

    return len(json_paths), chunks


def _print_stats(
    n_articles: int,
    chunks: list[Chunk],
    model_tokens: int,
    cl100k_tokens: int,
    embed_elapsed_s: float,
    collection_name: str,
    chroma_dir: Path,
) -> None:
    """Print final summary -- this is what hiring managers will read off README screenshots."""
    print(
        "\nIndex build complete:\n"
        f"  articles processed : {n_articles}\n"
        f"  chunks written     : {len(chunks):,}\n"
        f"  model tokens       : {model_tokens:,}\n"
        f"  cl100k tokens      : {cl100k_tokens:,}  (reported for cross-provider comparison)\n"
        f"  embedding wall time: {embed_elapsed_s:.1f}s\n"
        f"  collection         : {collection_name!r} at {chroma_dir}\n"
        f"  cost               : $0.00 (local sentence-transformers)\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. ``--help`` for full usage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--mode",
        choices=["append", "rebuild"],
        default="rebuild",
        help="rebuild drops the collection first; append upserts on top of it.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--collection", default="f1_races")
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Sentence-transformers batch size during encoding.",
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

    settings = get_settings()

    logger.info("loading embedding model %r ...", settings.embedding_model)
    model = load_embedding_model(settings)
    tokenizer = model.tokenizer

    logger.info("chunking articles under %s ...", args.data_root)
    n_articles, chunks = collect_chunks(
        args.data_root, tokenizer, chunk_size=args.chunk_size, overlap=args.overlap
    )

    if not chunks:
        logger.error("zero chunks produced; aborting")
        return 1

    # Token stats in both spaces for reviewer readability.
    model_tokens = sum(
        len(tokenizer(c.text, add_special_tokens=False)["input_ids"]) for c in chunks
    )
    cl100k_enc = tiktoken.get_encoding("cl100k_base")
    cl100k_tokens = sum(len(cl100k_enc.encode(c.text)) for c in chunks)

    logger.info(
        "embedding %d chunks (%d model-tokens / %d cl100k tokens) ...",
        len(chunks),
        model_tokens,
        cl100k_tokens,
    )
    start = time.perf_counter()
    embeddings = embed_texts(
        [c.text for c in chunks],
        model,
        batch_size=args.encode_batch_size,
        show_progress=True,
    )
    embed_elapsed = time.perf_counter() - start

    logger.info("connecting to Chroma at %s ...", settings.chroma_dir)
    client = make_chroma_client(settings)
    collection = get_or_reset_collection(client, args.collection, mode=args.mode)

    upsert_chunks(collection, chunks, embeddings)

    _print_stats(
        n_articles,
        chunks,
        model_tokens,
        cl100k_tokens,
        embed_elapsed,
        args.collection,
        settings.chroma_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
