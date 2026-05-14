"""Quick retrieval smoke test: embed a query, print the top-k matching chunks.

Hand-driven sanity check, not the production query path -- that lands as
``f1_rag.retrieve`` in phase 4. Useful right after a fresh
``build_index.py`` run to confirm the collection is populated and that
similarity rankings make sense for a query you can eyeball.

Usage::

    uv run python scripts/query_smoke.py
    uv run python scripts/query_smoke.py "Why was Ferrari's 2022 Hungary strategy criticised?"
    uv run python scripts/query_smoke.py "porpoising 2022" -k 5
"""

from __future__ import annotations

import argparse
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from f1_rag.config import get_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else None
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Hungarian Grand Prix 2022 tyre strategy",
        help="Free-form query string (default: a Hungary 2022 example).",
    )
    parser.add_argument("-k", type=int, default=3, help="Number of chunks to return.")
    parser.add_argument("--collection", default="f1_races")
    parser.add_argument(
        "--snippet-chars", type=int, default=240,
        help="Truncate each chunk preview to this many characters.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    print(f"loading {settings.embedding_model} ...", file=sys.stderr)
    model = SentenceTransformer(settings.embedding_model)
    emb = model.encode(args.query, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    col = client.get_collection(args.collection)

    result = col.query(query_embeddings=[emb], n_results=args.k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]

    print(f'\nquery: "{args.query}"\n')
    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
        race = meta.get("race_title", "?")
        section = meta.get("section_title", "?")
        print(f"[{rank}] dist={dist:.3f}  {race}  /  {section}")
        snippet = doc[: args.snippet_chars].replace("\n", " ")
        print(f"    {snippet}...\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
