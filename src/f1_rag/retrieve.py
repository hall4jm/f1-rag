"""Top-k retrieval over the Chroma f1_races collection.

A ``Searcher`` owns one ``SentenceTransformer`` and one Chroma collection
handle -- model load is the slow bit (~3-5 s on CPU first time), so a
long-lived process (Streamlit, the eval harness) keeps one Searcher around
for the lifetime of the run.

Public surface is two callables:
- ``retrieve(query, k=5, *, where=None)`` -- one-shot, uses an lru_cached
  process-wide Searcher under the hood.
- ``Searcher.search(...)`` -- if you already hold a Searcher (e.g. via
  ``st.cache_resource``) and want to avoid the indirection.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import chromadb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from f1_rag.chunk import ChunkMetadata
from f1_rag.config import Settings, get_settings

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

DEFAULT_COLLECTION = "f1_races"
DEFAULT_K = 5

# Asymmetric retrieval models want a search-instruction prefix on the *query* only;
# passages are encoded plain. Adding the prefix on a symmetric model (MiniLM, etc.)
# wouldn't break anything but won't help either, so we keep the mapping explicit.
_BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_QUERY_INSTRUCTION_BY_MODEL: dict[str, str] = {
    "BAAI/bge-small-en-v1.5": _BGE_INSTRUCTION,
    "BAAI/bge-base-en-v1.5": _BGE_INSTRUCTION,
    "BAAI/bge-large-en-v1.5": _BGE_INSTRUCTION,
}


def query_instruction_for(model_name: str) -> str:
    """Return the query-side instruction prefix for asymmetric models, else empty."""
    return _QUERY_INSTRUCTION_BY_MODEL.get(model_name, "")


class RetrievedChunk(BaseModel):
    """One chunk returned by similarity search: model-facing text + typed metadata + score."""

    text: str
    metadata: ChunkMetadata
    distance: float


class Searcher:
    """Stateful retriever: holds the embedding model and the Chroma collection."""

    def __init__(
        self,
        settings: Settings,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self.model_name = settings.embedding_model
        self.model = SentenceTransformer(settings.embedding_model)
        self.query_instruction = query_instruction_for(self.model_name)
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        self.collection: Collection = client.get_collection(collection_name)

    def search(
        self,
        query: str,
        *,
        k: int = DEFAULT_K,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Embed the query and return the top-k chunks by cosine distance.

        ``where`` is a Chroma metadata filter (``{"season": 2022}``,
        ``{"kind": "race"}``, etc.) -- useful for the v2 agent path
        and for narrowing year-drift on temporally specific queries.

        For asymmetric models (BGE-v1.5 family), the query is prefixed with a
        search instruction before encoding -- this is what makes question-shaped
        queries retrieve passage-style chunks correctly.
        """
        prefixed = self.query_instruction + query
        emb = self.model.encode(prefixed, normalize_embeddings=True).tolist()
        kwargs: dict[str, Any] = {"query_embeddings": [emb], "n_results": k}
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        return [
            RetrievedChunk(
                text=text,
                metadata=ChunkMetadata.model_validate(meta),
                distance=dist,
            )
            for text, meta, dist in zip(
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]


@lru_cache(maxsize=1)
def get_searcher() -> Searcher:
    """Return a process-wide Searcher (lazily loaded on first access)."""
    return Searcher(get_settings())


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    *,
    where: dict[str, Any] | None = None,
    searcher: Searcher | None = None,
) -> list[RetrievedChunk]:
    """Top-k retrieval. Pass ``searcher`` for explicit DI; else uses the cached singleton."""
    if searcher is None:
        searcher = get_searcher()
    return searcher.search(query, k=k, where=where)
