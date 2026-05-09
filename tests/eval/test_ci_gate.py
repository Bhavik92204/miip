"""
CI gate for RAGAS eval scores.

Loads the pre-computed scores from eval_results.json and asserts that
each metric meets its minimum threshold.  The eval runner (run_eval.py)
must have been executed before this file is meaningful — in CI the
Makefile `eval` target handles that sequencing.

Thresholds
----------
context_recall    >= 0.70   (current: 0.91)
answer_relevancy  >= 0.50   (current: 0.56)
faithfulness      SKIPPED   — Groq free-tier enforces a 1 024-token
                              completion cap that truncates the NLI JSON
                              for long answers; 2 of 5 scenarios score
                              N/A.  Re-enable once on a paid plan or when
                              answer length is bounded to ~600 chars.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# ── Load results ──────────────────────────────────────────────────────────────

RESULTS_FILE = Path(__file__).parent / "eval_results.json"

THRESHOLDS = {
    "context_recall":   0.70,
    "answer_relevancy": 0.50,
}


def _load() -> dict:
    if not RESULTS_FILE.exists():
        pytest.fail(
            f"eval_results.json not found at {RESULTS_FILE}.\n"
            "Run the eval first:  python tests/eval/run_eval.py"
        )
    return json.loads(RESULTS_FILE.read_text())


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def results() -> dict:
    return _load()


@pytest.fixture(scope="module")
def mean_scores(results) -> dict:
    return results["mean_scores"]


@pytest.fixture(scope="module")
def per_scenario(results) -> list[dict]:
    return results["per_scenario"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_line(metric: str, actual: float | None, threshold: float) -> str:
    if actual is None:
        return f"  {metric:<22} actual=N/A      threshold={threshold:.2f}  SKIP"
    status = "PASS" if actual >= threshold else "FAIL"
    arrow  = ">=" if actual >= threshold else "< "
    return (
        f"  {metric:<22} actual={actual:.4f}  threshold={threshold:.2f}  "
        f"{actual:.4f} {arrow} {threshold:.2f}  [{status}]"
    )


def _print_summary(mean_scores: dict) -> None:
    print()
    print("=" * 65)
    print("RAGAS CI GATE — score summary")
    print("=" * 65)
    for metric, threshold in THRESHOLDS.items():
        actual = mean_scores.get(metric)
        print(_score_line(metric, actual, threshold))
    faith = mean_scores.get("faithfulness")
    faith_s = f"{faith:.4f}" if faith is not None else "N/A"
    print(
        f"  {'faithfulness':<22} actual={faith_s:<7}  "
        "threshold=SKIPPED (Groq free-tier token cap)"
    )
    print("=" * 65)
    print()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_context_recall_meets_threshold(mean_scores):
    """Mean context_recall across all scored scenarios must be >= 0.70."""
    _print_summary(mean_scores)
    actual = mean_scores.get("context_recall")
    threshold = THRESHOLDS["context_recall"]

    assert actual is not None, (
        "context_recall is None in eval_results.json — re-run the eval."
    )
    assert actual >= threshold, (
        f"context_recall {actual:.4f} is below the required threshold {threshold:.2f}.\n"
        f"Run `python tests/eval/run_eval.py` and investigate low-scoring scenarios."
    )


def test_answer_relevancy_meets_threshold(mean_scores):
    """Mean answer_relevancy across all scored scenarios must be >= 0.50."""
    actual = mean_scores.get("answer_relevancy")
    threshold = THRESHOLDS["answer_relevancy"]

    assert actual is not None, (
        "answer_relevancy is None in eval_results.json — re-run the eval."
    )
    assert actual >= threshold, (
        f"answer_relevancy {actual:.4f} is below the required threshold {threshold:.2f}.\n"
        f"Run `python tests/eval/run_eval.py` and investigate low-scoring scenarios."
    )


@pytest.mark.skip(
    reason=(
        "Faithfulness skipped: Groq free-tier caps completions at 1 024 tokens, "
        "truncating the NLI JSON for long analyses (INC-SYN-001, INC-SYN-002). "
        "Re-enable by setting FAITHFULNESS_THRESHOLD and upgrading the Groq plan "
        "or bounding answer length to ~600 chars before scoring."
    )
)
def test_faithfulness_meets_threshold(mean_scores):
    threshold = 0.60
    actual = mean_scores.get("faithfulness")
    assert actual is not None and actual >= threshold, (
        f"faithfulness {actual} is below threshold {threshold}"
    )


def test_per_scenario_no_total_failure(per_scenario):
    """No scenario should have ALL three metrics as N/A simultaneously."""
    all_na = [
        s["incident_id"]
        for s in per_scenario
        if s.get("context_recall") is None
        and s.get("answer_relevancy") is None
        and s.get("faithfulness") is None
    ]
    assert not all_na, (
        f"These scenarios have no valid scores at all: {all_na}\n"
        "Check that the eval runner completed successfully."
    )


def test_eval_results_covers_all_scenarios(per_scenario):
    """Results file must contain exactly 5 scenario entries."""
    assert len(per_scenario) == 5, (
        f"Expected 5 scenario results, found {len(per_scenario)}.\n"
        "Re-run the eval to regenerate eval_results.json."
    )
