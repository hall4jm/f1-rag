"""Eval harness for f1-rag.

Loads the gold Q&A set, runs every question through ``pipeline.answer``,
computes retrieval metrics (hit@k, MRR) directly, and delegates the
answer-quality metrics (faithfulness, answer_relevancy, context_precision,
context_recall) to Ragas with a configurable judge LLM. Out-of-scope
questions are scored separately on refusal-phrase detection.

This module is **not** imported by the runtime app -- Ragas pulls heavy
deps (langchain, datasets, pyarrow) that we don't want on the Streamlit
deploy. Hence the ``[dependency-groups].eval`` group.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote

from pydantic import BaseModel, Field
from tqdm import tqdm

from f1_rag.config import LLMProvider, Settings, get_settings
from f1_rag.pipeline import AgenticAnswer, Answer, answer, answer_agentic
from f1_rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

Category = Literal["factual", "strategy", "comparison", "multi_race", "out_of_scope"]
ALL_CATEGORIES: tuple[Category, ...] = (
    "factual",
    "strategy",
    "comparison",
    "multi_race",
    "out_of_scope",
)

EvalMode = Literal["v1", "agentic"]

REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"don'?t have (enough )?information", re.IGNORECASE),
    re.compile(r"cannot answer", re.IGNORECASE),
    re.compile(r"can'?t answer", re.IGNORECASE),
    re.compile(r"not in (the |my )?(retrieved )?(passages|sources|context)", re.IGNORECASE),
    re.compile(r"outside (the |my )?(scope|corpus|sources)", re.IGNORECASE),
    re.compile(r"no information (in|about|on)", re.IGNORECASE),
    re.compile(r"sources (do not|don'?t) (contain|cover|include)", re.IGNORECASE),
    re.compile(r"unable to (find|answer)", re.IGNORECASE),
)


class GoldQuestion(BaseModel):
    """One row of the gold Q&A set."""

    id: str
    category: Category
    question: str
    gold_answer: str
    gold_source_urls: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvalResult(BaseModel):
    """Per-question eval output: pipeline output + metrics."""

    question: GoldQuestion
    mode: EvalMode = "v1"
    answer_text: str
    retrieved_chunks: list[RetrievedChunk]
    retrieved_urls: list[str]
    # Retrieval metrics -- None for out_of_scope questions (no gold URLs).
    retrieval_hit_at_k: float | None = None
    retrieval_mrr: float | None = None
    # Refusal flag -- only meaningful for out_of_scope.
    refusal_detected: bool = False
    # Agentic-only fields (empty / 0 for v1).
    tool_calls_used: list[str] = Field(default_factory=list)
    rounds_used: int = 0
    terminated_early: bool = False
    # Ragas metrics -- None for out_of_scope or when Ragas didn't run.
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


# ---------- helpers: gold loading + URL normalization + metrics ----------


def load_gold(path: Path) -> list[GoldQuestion]:
    """Load the gold YAML and validate each entry.

    PyYAML is imported lazily so importing this module doesn't require the
    eval dep group -- the retrieval-metric / refusal-detection tests can
    run on a stock dev install.
    """
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = raw.get("questions", []) if isinstance(raw, dict) else raw
    return [GoldQuestion.model_validate(item) for item in items]


def normalize_url(url: str) -> str:
    """URL-decode + lowercase + strip trailing slash -- comparison-safe form."""
    return unquote(url).rstrip("/").lower()


def hit_at_k(retrieved_urls: list[str], gold_urls: list[str]) -> float:
    """1.0 if any retrieved URL matches any gold URL after normalization, else 0.0."""
    if not gold_urls:
        return 0.0
    gold_set = {normalize_url(u) for u in gold_urls}
    return float(any(normalize_url(u) in gold_set for u in retrieved_urls))


def mrr(retrieved_urls: list[str], gold_urls: list[str]) -> float:
    """Reciprocal rank of the first matching URL (1.0 = top hit, 0.0 = no match in top-k)."""
    if not gold_urls:
        return 0.0
    gold_set = {normalize_url(u) for u in gold_urls}
    for rank, url in enumerate(retrieved_urls, start=1):
        if normalize_url(url) in gold_set:
            return 1.0 / rank
    return 0.0


def detect_refusal(answer_text: str) -> bool:
    """True if the answer contains any refusal phrase. Coarse but transparent."""
    return any(p.search(answer_text) for p in REFUSAL_PATTERNS)


# ---------- core run loop ----------


def run_eval(
    questions: list[GoldQuestion],
    *,
    mode: EvalMode = "v1",
    k: int = 5,
    use_ragas: bool = True,
    provider: LLMProvider | None = None,
    model: str | None = None,
    judge_model: str | None = None,
    settings: Settings | None = None,
) -> list[EvalResult]:
    """Run the pipeline on every question and compute per-question metrics.

    ``mode`` selects v1 (pure RAG) or agentic (RAG + Jolpica tools).
    ``judge_model`` overrides the Ragas judge LLM independently of the pipeline
    model -- useful for staying under rate limits.
    """
    s = settings if settings is not None else get_settings()
    results: list[EvalResult] = []

    for q in tqdm(questions, desc=f"eval[{mode}]", unit="q"):
        if q.gold_answer.strip() == "TODO":
            logger.info("skipping %s: gold_answer is TODO", q.id)
            continue

        ans: Answer | AgenticAnswer
        if mode == "v1":
            ans = answer(q.question, k=k, provider=provider, model=model, settings=s)
            tool_calls_used: list[str] = []
            rounds_used = 0
            terminated_early = False
        else:
            agent_ans = answer_agentic(
                q.question, k=k, provider=provider, model=model, settings=s
            )
            ans = agent_ans
            tool_calls_used = [tc.name for tc in agent_ans.tool_calls]
            rounds_used = agent_ans.rounds_used
            terminated_early = agent_ans.terminated_early

        retrieved_urls = [c.metadata.source_url for c in ans.retrieved_chunks]

        if q.category == "out_of_scope":
            retrieval_hit = None
            retrieval_mrr_score = None
        else:
            retrieval_hit = hit_at_k(retrieved_urls, q.gold_source_urls)
            retrieval_mrr_score = mrr(retrieved_urls, q.gold_source_urls)

        results.append(
            EvalResult(
                question=q,
                mode=mode,
                answer_text=ans.answer_text,
                retrieved_chunks=ans.retrieved_chunks,
                retrieved_urls=retrieved_urls,
                retrieval_hit_at_k=retrieval_hit,
                retrieval_mrr=retrieval_mrr_score,
                refusal_detected=detect_refusal(ans.answer_text),
                tool_calls_used=tool_calls_used,
                rounds_used=rounds_used,
                terminated_early=terminated_early,
            )
        )

    if use_ragas:
        _attach_ragas_scores(results, settings=s, judge_model=judge_model)

    return results


# ---------- v1 vs v2 comparison ----------


def format_comparison_report(
    v1_results: list[EvalResult],
    v2_results: list[EvalResult],
    *,
    metadata: dict[str, str] | None = None,
) -> str:
    """Side-by-side v1 vs v2 report. This is THE phase-6 output."""
    lines: list[str] = ["# f1-rag v1 vs v2 Comparison\n"]

    if metadata:
        for key, value in metadata.items():
            lines.append(f"- **{key}:** {value}")
        lines.append("")

    v1_by_id = {r.question.id: r for r in v1_results}
    v2_by_id = {r.question.id: r for r in v2_results}
    common = sorted(set(v1_by_id) & set(v2_by_id))

    # Aggregate deltas
    lines.append("## Aggregate scores\n")
    v1_overall = _means_for(v1_results)
    v2_overall = _means_for(v2_results)
    lines.append("| Metric | v1 (pure RAG) | v2 (RAG + tools) | Δ |")
    lines.append("|---|---|---|---|")
    metric_keys = (*_RETRIEVAL_METRICS, *_RAGAS_METRICS, "refusal_rate")
    for key in metric_keys:
        v1m = v1_overall.get(key)
        v2m = v2_overall.get(key)
        delta = ""
        if v1m and v2m:
            d = v2m[0] - v1m[0]
            delta = f"{d:+.3f}"
        lines.append(f"| {key} | {_fmt(v1m)} | {_fmt(v2m)} | {delta} |")
    lines.append("")

    # Per-question side-by-side
    lines.append("## Per-question outcomes\n")
    lines.append(
        "| ID | Category | v1 refused | v2 refused | v1 Hit@k | v2 Hit@k | v2 tools used |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for qid in common:
        v1 = v1_by_id[qid]
        v2 = v2_by_id[qid]
        v1_ref = "✓" if v1.refusal_detected else ""
        v2_ref = "✓" if v2.refusal_detected else ""
        v1_hit = f"{v1.retrieval_hit_at_k:.0f}" if v1.retrieval_hit_at_k is not None else "—"
        v2_hit = f"{v2.retrieval_hit_at_k:.0f}" if v2.retrieval_hit_at_k is not None else "—"
        tools = ", ".join(v2.tool_calls_used) if v2.tool_calls_used else "(none)"
        lines.append(
            f"| {qid} | {v1.question.category} | {v1_ref} | {v2_ref} | "
            f"{v1_hit} | {v2_hit} | {tools} |"
        )
    lines.append("")

    # Most interesting wins / losses
    lines.append("## Where v2 changed v1's behaviour\n")
    wins: list[str] = []
    losses: list[str] = []
    for qid in common:
        v1 = v1_by_id[qid]
        v2 = v2_by_id[qid]
        if v1.question.category == "out_of_scope":
            # We *want* refusal here -- v2 changing it is a regression, not a win.
            if not v2.refusal_detected and v1.refusal_detected:
                losses.append(qid)
            continue
        if v1.refusal_detected and not v2.refusal_detected:
            wins.append(qid)
        elif not v1.refusal_detected and v2.refusal_detected:
            losses.append(qid)

    if wins:
        lines.append("**v2 answered where v1 refused (in-scope wins):**\n")
        for qid in wins:
            v1 = v1_by_id[qid]
            v2 = v2_by_id[qid]
            lines.append(f"### {qid} ({v1.question.category})")
            lines.append(f"**Question:** {v1.question.question}\n")
            lines.append(f"**v1:** {v1.answer_text}\n")
            lines.append(f"**v2:** {v2.answer_text}\n")
            lines.append(f"**v2 tool calls:** {', '.join(v2.tool_calls_used) or '(none)'}\n")
    else:
        lines.append("**v2 answered where v1 refused:** (none)\n")

    if losses:
        lines.append("**Regressions (v2 refused or hallucinated where v1 behaved correctly):**\n")
        for qid in losses:
            v1 = v1_by_id[qid]
            v2 = v2_by_id[qid]
            lines.append(f"- **{qid}** ({v1.question.category}): v1 was OK, v2 changed behaviour")
            lines.append(f"  - v1: {v1.answer_text}")
            lines.append(f"  - v2: {v2.answer_text}")
        lines.append("")
    else:
        lines.append("**Regressions:** (none)\n")

    return "\n".join(lines) + "\n"


# ---------- Ragas integration ----------


def _attach_ragas_scores(
    results: list[EvalResult], *, settings: Settings, judge_model: str | None = None
) -> None:
    """Compute Ragas metrics in-place on in-scope results."""
    in_scope = [r for r in results if r.question.category != "out_of_scope"]
    if not in_scope:
        logger.info("no in-scope results; skipping Ragas")
        return

    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Ragas / datasets not installed: {exc}. "
            "Run `uv sync --group eval` first."
        ) from exc

    judge_llm = _make_judge_llm(settings, judge_model=judge_model)
    judge_emb = _make_judge_embeddings(settings)

    dataset = Dataset.from_dict(
        {
            "question": [r.question.question for r in in_scope],
            "answer": [r.answer_text for r in in_scope],
            "contexts": [[c.text for c in r.retrieved_chunks] for r in in_scope],
            "ground_truth": [r.question.gold_answer for r in in_scope],
        }
    )

    logger.info("running Ragas on %d in-scope results...", len(in_scope))
    ragas_result = ragas_evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_emb,
    )

    # ragas_result indexes per-metric arrays; convert to plain Python floats.
    for i, r in enumerate(in_scope):
        r.faithfulness = _safe_float(ragas_result["faithfulness"][i])
        r.answer_relevancy = _safe_float(ragas_result["answer_relevancy"][i])
        r.context_precision = _safe_float(ragas_result["context_precision"][i])
        r.context_recall = _safe_float(ragas_result["context_recall"][i])


def _safe_float(value: Any) -> float | None:
    """Convert Ragas score values to Python floats, returning None for NaN / missing."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _make_judge_llm(settings: Settings, *, judge_model: str | None = None) -> Any:
    """Build a Ragas-compatible LLM for the configured provider.

    Defaults to the *pipeline* provider so a free-tier Groq run can self-evaluate
    without extra credentials. ``judge_model`` overrides the model name used
    (independent from the pipeline model) -- handy for using a smaller / cheaper
    model as judge to stay under daily token caps.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    p = settings.llm_provider
    model_name = judge_model if judge_model is not None else settings.resolved_llm_model

    if p == "groq":
        if settings.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY required to run Ragas with a Groq judge")
        llm = ChatOpenAI(
            model=model_name,
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
        )
    elif p == "openai":
        if settings.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY required to run Ragas with an OpenAI judge")
        llm = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key.get_secret_value(),
        )
    else:
        raise RuntimeError(
            f"Ragas judge LLM only wired for groq / openai; got {p!r}. "
            "Set LLM_PROVIDER=groq (or openai) for the eval invocation."
        )

    return LangchainLLMWrapper(llm)


def _make_judge_embeddings(settings: Settings) -> Any:
    """Build a Ragas-compatible embeddings wrapper using the same local model as the index."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    emb = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    return LangchainEmbeddingsWrapper(emb)


# ---------- aggregation + report ----------

_RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
_RETRIEVAL_METRICS = ("retrieval_hit_at_k", "retrieval_mrr")
_ALL_METRICS = _RETRIEVAL_METRICS + _RAGAS_METRICS


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _means_for(results: list[EvalResult]) -> dict[str, tuple[float, int]]:
    """Return {metric_name: (mean, n_scored)} skipping None values."""
    out: dict[str, tuple[float, int]] = {}
    for m in _ALL_METRICS:
        values = [getattr(r, m) for r in results if getattr(r, m) is not None]
        if values:
            out[m] = (_mean(values), len(values))
    # refusal_rate is meaningful only for out_of_scope.
    oos = [r for r in results if r.question.category == "out_of_scope"]
    if oos:
        out["refusal_rate"] = (
            sum(1 for r in oos if r.refusal_detected) / len(oos),
            len(oos),
        )
    return out


def _fmt(score_n: tuple[float, int] | None) -> str:
    if score_n is None:
        return "—"
    return f"{score_n[0]:.3f} (n={score_n[1]})"


def format_markdown_report(
    results: list[EvalResult],
    *,
    n_worst: int = 5,
    metadata: dict[str, str] | None = None,
) -> str:
    """Render an end-to-end eval markdown report. Output is README-pasteable."""
    lines: list[str] = ["# f1-rag Evaluation Report\n"]

    if metadata:
        for k, v in metadata.items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    n_total = len(results)
    n_skipped_todo = sum(1 for r in results if r.question.gold_answer.strip() == "TODO")
    lines.append(f"**Questions scored:** {n_total}  ")
    if n_skipped_todo:
        lines.append(f"**Questions skipped (TODO gold):** {n_skipped_todo}  ")
    lines.append("")

    # Overall
    lines.append("## Overall scores\n")
    overall = _means_for(results)
    lines.append("| Metric | Score |")
    lines.append("|---|---|")
    for metric in (*_RETRIEVAL_METRICS, *_RAGAS_METRICS, "refusal_rate"):
        lines.append(f"| {metric} | {_fmt(overall.get(metric))} |")
    lines.append("")

    # By category
    lines.append("## By category\n")
    header = "| Category | n | Hit@k | MRR | Faithfulness | Answer Rel. | Ctx Prec. | Ctx Recall | Refusal |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for cat in ALL_CATEGORIES:
        cat_results = [r for r in results if r.question.category == cat]
        if not cat_results:
            continue
        m = _means_for(cat_results)
        row = [
            cat,
            str(len(cat_results)),
            _fmt(m.get("retrieval_hit_at_k")),
            _fmt(m.get("retrieval_mrr")),
            _fmt(m.get("faithfulness")),
            _fmt(m.get("answer_relevancy")),
            _fmt(m.get("context_precision")),
            _fmt(m.get("context_recall")),
            _fmt(m.get("refusal_rate")),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Worst by faithfulness (in-scope only; falls back to MRR for retrieval-only runs)
    scored = [r for r in results if r.faithfulness is not None]
    if scored:
        scored.sort(key=lambda r: r.faithfulness or 0.0)
        title = "faithfulness"
    else:
        scored = [r for r in results if r.retrieval_mrr is not None]
        scored.sort(key=lambda r: r.retrieval_mrr or 0.0)
        title = "MRR"
    if scored:
        lines.append(f"## Worst-scoring examples (bottom {min(n_worst, len(scored))} by {title})\n")
        for r in scored[:n_worst]:
            lines.append(f"### {r.question.id} ({r.question.category})")
            lines.append(f"**Question:** {r.question.question}\n")
            lines.append(f"**Gold answer:** {r.question.gold_answer}\n")
            lines.append(f"**Model answer:** {r.answer_text}\n")
            score_bits: list[str] = []
            if r.retrieval_hit_at_k is not None:
                score_bits.append(f"hit@k={r.retrieval_hit_at_k:.0f}")
            if r.retrieval_mrr is not None:
                score_bits.append(f"MRR={r.retrieval_mrr:.3f}")
            if r.faithfulness is not None:
                score_bits.append(f"faithfulness={r.faithfulness:.3f}")
            if r.answer_relevancy is not None:
                score_bits.append(f"answer_rel={r.answer_relevancy:.3f}")
            if score_bits:
                lines.append("**Scores:** " + ", ".join(score_bits) + "\n")

    lines.append("## Methodology notes\n")
    lines.append(
        "- **Judge LLM ≈ judged LLM:** the same provider runs the pipeline answers and "
        "the Ragas judge. A stronger judge would give more trustworthy faithfulness scores; "
        "treat these numbers as relative (good for tracking change-over-time), not absolute."
    )
    lines.append(
        "- **Out-of-scope category** is scored on refusal-phrase detection only (no gold URLs, "
        "no Ragas faithfulness). Refusal rate of 1.0 means the system declined all out-of-scope "
        "questions; <1.0 means it hallucinated against at least one."
    )
    lines.append(
        "- **Retrieval URLs** are compared after URL-decode + lowercase + trailing-slash strip, "
        "so encoded and human forms of the same Wikipedia article match."
    )

    return "\n".join(lines) + "\n"


def save_results_json(results: list[EvalResult], path: Path) -> None:
    """Persist raw per-question results to JSON for downstream analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump(mode="json") for r in results]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def default_run_metadata(settings: Settings, *, k: int) -> dict[str, str]:
    """Header metadata for the markdown report."""
    return {
        "Run timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Pipeline provider": settings.llm_provider,
        "Pipeline model": settings.resolved_llm_model,
        "Embedding model": settings.embedding_model,
        "Top-k": str(k),
    }
