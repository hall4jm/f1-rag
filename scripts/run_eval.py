"""Run the f1-rag eval harness end-to-end.

Loads ``evals/questions.yaml`` (or a path you provide), runs the pipeline
on every question, computes retrieval + Ragas metrics, and writes a
markdown report + raw JSON to ``evals/results/<timestamp>.{md,json}``.

Requires the optional eval deps: ``uv sync --group eval`` first.

Usage::

    uv run python scripts/run_eval.py
    uv run python scripts/run_eval.py --category strategy --k 5
    uv run python scripts/run_eval.py --no-ragas        # retrieval-only, fast iteration
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from f1_rag.config import get_settings
from f1_rag.evaluate import (
    ALL_CATEGORIES,
    EvalResult,
    default_run_metadata,
    format_comparison_report,
    format_markdown_report,
    load_gold,
    run_eval,
    save_results_json,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """CLI entry. ``--help`` for full usage."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--gold", type=Path, default=Path("evals/questions.yaml"))
    parser.add_argument(
        "--category",
        choices=list(ALL_CATEGORIES),
        default=None,
        help="Only run questions of this category.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--mode", choices=["v1", "agentic"], default="v1",
        help="v1 = pure RAG; agentic = RAG + Jolpica tool calls.",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both v1 and agentic modes, output a side-by-side comparison report.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("evals/results"),
        help="Where to write the report (.md) and raw results (.json).",
    )
    parser.add_argument(
        "--no-ragas", action="store_true",
        help="Skip Ragas metrics. Fast iteration over retrieval-only changes.",
    )
    parser.add_argument(
        "--provider", default=None,
        help="Override LLM_PROVIDER for this run (anthropic | openai | groq | ollama).",
    )
    parser.add_argument(
        "--model", default=None, help="Override LLM_MODEL for this run.",
    )
    parser.add_argument(
        "--judge-model", default=None,
        help=(
            "Override the Ragas judge LLM model (independent from --model). "
            "Use a smaller model (e.g. llama-3.1-8b-instant) to stay under daily token caps."
        ),
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    settings = get_settings()

    logger.info("loading gold set from %s", args.gold)
    questions = load_gold(args.gold)
    if args.category:
        questions = [q for q in questions if q.category == args.category]
    if not questions:
        logger.error("no questions to run (after filters); aborting")
        return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def _run_one(mode: str) -> tuple[list[EvalResult], str, Path, Path]:
        logger.info(
            "running eval on %d question(s) — mode=%s provider=%s model=%s ragas=%s",
            len(questions),
            mode,
            args.provider or settings.llm_provider,
            args.model or settings.resolved_llm_model,
            not args.no_ragas,
        )
        results = run_eval(
            questions,
            mode=mode,  # type: ignore[arg-type]
            k=args.k,
            use_ragas=not args.no_ragas,
            provider=args.provider,
            model=args.model,
            judge_model=args.judge_model,
            settings=settings,
        )
        metadata = default_run_metadata(settings, k=args.k)
        metadata["Mode"] = mode
        if args.no_ragas:
            metadata["Ragas"] = "skipped (--no-ragas)"
        else:
            metadata["Ragas judge model"] = args.judge_model or settings.resolved_llm_model
        report = format_markdown_report(results, metadata=metadata)

        md_path = args.out_dir / f"{timestamp}_{mode}.md"
        json_path = args.out_dir / f"{timestamp}_{mode}.json"
        md_path.write_text(report, encoding="utf-8")
        save_results_json(results, json_path)
        return results, report, md_path, json_path

    if args.compare:
        v1_results, _v1_report, v1_md, v1_json = _run_one("v1")
        v2_results, _v2_report, v2_md, v2_json = _run_one("agentic")

        comp_metadata = default_run_metadata(settings, k=args.k)
        if args.no_ragas:
            comp_metadata["Ragas"] = "skipped (--no-ragas)"
        else:
            comp_metadata["Ragas judge model"] = args.judge_model or settings.resolved_llm_model
        comp_report = format_comparison_report(v1_results, v2_results, metadata=comp_metadata)

        comp_md_path = args.out_dir / f"{timestamp}_comparison.md"
        comp_md_path.write_text(comp_report, encoding="utf-8")

        print(comp_report)
        print(f"\n[wrote {v1_md}]")
        print(f"[wrote {v1_json}]")
        print(f"[wrote {v2_md}]")
        print(f"[wrote {v2_json}]")
        print(f"[wrote {comp_md_path}]")
        return 0

    _, report, md_path, json_path = _run_one(args.mode)
    print(report)
    print(f"\n[wrote {md_path}]")
    print(f"[wrote {json_path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
