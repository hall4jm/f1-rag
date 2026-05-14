"""Embedding chunks via sentence-transformers and writing them to ChromaDB.

The embedding model is loaded lazily by the caller and passed in -- not
created here -- so that a single model instance can serve both the index
build (this module) and query time (retrieve.py). The model lives on the
CPU; downloads land in ~/.cache/huggingface on first use.

Normalized embeddings (``normalize_embeddings=True``) are essential for
the cosine-space Chroma collection: without normalization, MiniLM
outputs have varying magnitudes and L2/IP rankings disagree with cosine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from f1_rag.chunk import Chunk
from f1_rag.config import Settings

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection
    from chromadb.api import ClientAPI

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64
DEFAULT_UPSERT_BATCH = 1000


def load_embedding_model(settings: Settings) -> SentenceTransformer:
    """Load (and on first run, download) the sentence-transformers model."""
    return SentenceTransformer(settings.embedding_model)


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> list[list[float]]:
    """Encode texts to normalized embeddings. Returns plain Python lists.

    Batching is handled internally by ``SentenceTransformer.encode``; we just
    forward ``batch_size``. ``normalize_embeddings=True`` makes cosine = dot.
    """
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return arr.tolist()  # type: ignore[no-any-return]


def make_chroma_client(settings: Settings) -> "ClientAPI":
    """Open (creating if missing) a persistent Chroma client at settings.chroma_dir."""
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_or_reset_collection(
    client: "ClientAPI",
    name: str,
    *,
    mode: Literal["append", "rebuild"],
) -> "Collection":
    """Return a Chroma collection, dropping it first if ``mode == 'rebuild'``.

    ``embedding_function=None`` declares that we'll always supply embeddings
    explicitly on write -- Chroma won't try to re-embed on its own.
    ``hnsw:space=cosine`` matches our normalized embeddings.
    """
    if mode == "rebuild":
        names = [
            c.name if hasattr(c, "name") else str(c) for c in client.list_collections()
        ]
        if name in names:
            client.delete_collection(name)
            logger.info("dropped existing collection %r", name)
        else:
            logger.debug("rebuild requested but no existing collection %r", name)

    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def _scalar_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Drop None-valued keys; Chroma metadata accepts only scalar values."""
    return {k: v for k, v in chunk.metadata.model_dump().items() if v is not None}


def upsert_chunks(
    collection: "Collection",
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    batch_size: int = DEFAULT_UPSERT_BATCH,
) -> None:
    """Write chunks + embeddings to Chroma in batches.

    Upserting (not adding) makes the script idempotent w.r.t. chunk ids: re-running
    in ``append`` mode overwrites only the chunks whose ids match.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        )

    n = len(chunks)
    for start in tqdm(range(0, n, batch_size), desc="upsert", unit="batch"):
        end = min(start + batch_size, n)
        batch = chunks[start:end]
        collection.upsert(
            ids=[c.id for c in batch],
            documents=[c.text for c in batch],
            embeddings=embeddings[start:end],
            metadatas=[_scalar_metadata(c) for c in batch],
        )
