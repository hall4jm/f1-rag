"""Unit tests for evaluate.py pure-logic helpers.

We test the deterministic, no-dependency pieces: URL normalization,
hit@k, MRR, refusal detection, and the gold-loader. Ragas integration
isn't tested here -- it pulls heavy deps and needs an LLM key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from f1_rag.evaluate import (
    GoldQuestion,
    detect_refusal,
    hit_at_k,
    load_gold,
    mrr,
    normalize_url,
)

# Repo-rooted path to the real gold YAML so tests work no matter where pytest was invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------- URL normalization ----------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://en.wikipedia.org/wiki/2022_S%C3%A3o_Paulo_Grand_Prix",
            "https://en.wikipedia.org/wiki/2022_são_paulo_grand_prix",
        ),
        (
            "https://en.wikipedia.org/wiki/2022_São_Paulo_Grand_Prix/",
            "https://en.wikipedia.org/wiki/2022_são_paulo_grand_prix",
        ),
        (
            "HTTPS://EN.WIKIPEDIA.ORG/wiki/2022_Hungarian_Grand_Prix",
            "https://en.wikipedia.org/wiki/2022_hungarian_grand_prix",
        ),
    ],
)
def test_normalize_url_handles_encoding_case_and_trailing_slash(
    raw: str, expected: str
) -> None:
    assert normalize_url(raw) == expected


# ---------- hit_at_k ----------


def test_hit_at_k_returns_one_when_any_retrieved_url_matches_any_gold() -> None:
    retrieved = [
        "https://en.wikipedia.org/wiki/2020_Hungarian_Grand_Prix",
        "https://en.wikipedia.org/wiki/2022_Hungarian_Grand_Prix",
    ]
    gold = ["https://en.wikipedia.org/wiki/2022_Hungarian_Grand_Prix"]
    assert hit_at_k(retrieved, gold) == 1.0


def test_hit_at_k_returns_zero_when_no_overlap() -> None:
    retrieved = ["https://en.wikipedia.org/wiki/2020_Hungarian_Grand_Prix"]
    gold = ["https://en.wikipedia.org/wiki/2022_Hungarian_Grand_Prix"]
    assert hit_at_k(retrieved, gold) == 0.0


def test_hit_at_k_uses_url_normalized_form() -> None:
    # Encoded URL on one side, decoded on the other -- must still match.
    retrieved = ["https://en.wikipedia.org/wiki/2022_S%C3%A3o_Paulo_Grand_Prix"]
    gold = ["https://en.wikipedia.org/wiki/2022_São_Paulo_Grand_Prix"]
    assert hit_at_k(retrieved, gold) == 1.0


def test_hit_at_k_zero_when_no_gold_urls() -> None:
    # Out-of-scope-shaped input: empty gold => not gradeable as 1.0.
    assert hit_at_k(["https://anywhere"], []) == 0.0


# ---------- MRR ----------


def test_mrr_top_rank_is_one() -> None:
    retrieved = [
        "https://en.wikipedia.org/wiki/2022_Hungarian_Grand_Prix",
        "https://en.wikipedia.org/wiki/2020_Hungarian_Grand_Prix",
    ]
    gold = ["https://en.wikipedia.org/wiki/2022_Hungarian_Grand_Prix"]
    assert mrr(retrieved, gold) == 1.0


def test_mrr_rank_three_is_one_third() -> None:
    retrieved = [
        "https://en.wikipedia.org/wiki/A",
        "https://en.wikipedia.org/wiki/B",
        "https://en.wikipedia.org/wiki/Target",
    ]
    gold = ["https://en.wikipedia.org/wiki/Target"]
    assert mrr(retrieved, gold) == pytest.approx(1 / 3)


def test_mrr_no_match_is_zero() -> None:
    assert mrr(["a", "b"], ["c"]) == 0.0


def test_mrr_zero_when_no_gold() -> None:
    assert mrr(["anything"], []) == 0.0


# ---------- refusal detection ----------


@pytest.mark.parametrize(
    "text",
    [
        "I don't have enough information to answer that confidently from the sources I have.",
        "I don't have information about the 2010 season.",
        "The sources don't contain anything about that.",
        "That question is outside the scope of my sources.",
        "I cannot answer based on the retrieved passages.",
        "I'm unable to find that information in the sources.",
        "No information in the corpus on this topic.",
    ],
)
def test_detect_refusal_matches_common_refusal_phrasings(text: str) -> None:
    assert detect_refusal(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "Max Verstappen won the 2022 Hungarian Grand Prix starting from 10th.",
        "Ferrari's strategy was criticised because the hard tyres failed to warm up.",
        "Mercedes finished third in the 2022 Constructors' Championship.",
    ],
)
def test_detect_refusal_does_not_fire_on_real_answers(text: str) -> None:
    assert detect_refusal(text) is False


# ---------- gold loader ----------


def test_load_gold_parses_the_real_questions_yaml() -> None:
    pytest.importorskip("yaml")  # pyyaml lives in the eval dep group
    gold = load_gold(PROJECT_ROOT / "evals" / "questions.yaml")
    assert len(gold) >= 15
    # IDs unique
    ids = [q.id for q in gold]
    assert len(set(ids)) == len(ids), "duplicate id in gold set"
    # Every category present
    cats = {q.category for q in gold}
    assert "factual" in cats
    assert "strategy" in cats
    assert "comparison" in cats
    assert "multi_race" in cats
    assert "out_of_scope" in cats
    # Out-of-scope questions have empty gold_source_urls
    for q in gold:
        if q.category == "out_of_scope":
            assert q.gold_source_urls == []


def test_load_gold_parses_minimal_yaml(tmp_path: Path) -> None:
    yaml = pytest.importorskip("yaml")  # pyyaml lives in the eval dep group
    yaml_path = tmp_path / "g.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "questions": [
                    {
                        "id": "X1",
                        "category": "factual",
                        "question": "?",
                        "gold_answer": "yes",
                        "gold_source_urls": ["https://example.com/a"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    gold = load_gold(yaml_path)
    assert len(gold) == 1
    assert isinstance(gold[0], GoldQuestion)
    assert gold[0].id == "X1"
