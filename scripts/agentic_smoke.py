"""Single-question agentic-pipeline smoke test.

Runs ``answer_agentic`` on one question and prints the answer + the
tool-call trace. Useful for verifying the agent loop works end-to-end
without burning the daily eval budget on a full comparison run.

Usage::

    uv run python scripts/agentic_smoke.py
    uv run python scripts/agentic_smoke.py "Who took pole at the 2022 Hungarian GP?"
    uv run python scripts/agentic_smoke.py "Who won the 2023 championship?" --model llama-3.1-8b-instant
"""

from __future__ import annotations

import argparse
import logging
import sys

from f1_rag.pipeline import answer_agentic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "question",
        nargs="?",
        default="Who won the 2022 Hungarian Grand Prix and from what grid position did he start?",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override LLM_MODEL. Use a smaller model (e.g. llama-3.1-8b-instant) for cheap iteration.",
    )
    parser.add_argument("-k", type=int, default=5, help="Retrieved chunks to pass alongside tools.")
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    result = answer_agentic(
        args.question, k=args.k, max_rounds=args.max_rounds, model=args.model
    )

    print(f'\nQuestion: "{args.question}"\n')
    print(f"=== Answer ({result.rounds_used} round(s)"
          f"{', truncated' if result.terminated_early else ''}) ===\n")
    print(result.answer_text)
    print()
    print("=== Tool calls ===")
    if not result.tool_calls:
        print("  (none — the model answered from passages alone)")
    else:
        for tc in result.tool_calls:
            print(f"  [round {tc.round_index}] {tc.name}({tc.arguments})")
            print(f"      -> {tc.result_summary}")
    print()
    print("=== Retrieved chunks (top {0}) ===".format(len(result.retrieved_chunks)))
    for i, c in enumerate(result.retrieved_chunks, start=1):
        print(f"  [{i}] {c.metadata.race_title} / {c.metadata.section_title}  dist={c.distance:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
