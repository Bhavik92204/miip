"""
RAGAS evaluation runner for miip log_rag_agent.

Usage:
    python tests/eval/run_eval.py

For each scenario in synthetic_incidents.json:
  1. Feeds log_lines into log_rag_agent (stores + retrieves via pgvector, calls Groq)
  2. Collects the generated answer and the raw log lines as retrieved context
  3. Scores each sample with faithfulness, answer_relevancy, context_recall
     using the RAGAS 0.4.x collections metric.score() interface directly
  4. Prints a summary table and saves tests/eval/eval_results.json

Note on RAGAS 0.4.x:  ragas.metrics.collections metrics do NOT work with the
ragas.evaluate() helper (which validates for the old Metric base class).  They
expose score()/ascore() directly and must be called per-sample.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# ── Repo path setup ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
os.environ.setdefault("DATABASE_URL", "postgresql://miip:miip@localhost:5432/miip")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="ragas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

from miip.config import settings
from miip.agents.log_rag_agent import log_rag_agent
from miip.embeddings import embed as _embed
from miip.state import IncidentState

# ── RAGAS 0.4.x collections — called directly, not via evaluate() ─────────────
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_recall import ContextRecall
from ragas.llms import llm_factory
from ragas.embeddings import BaseRagasEmbedding
from openai import AsyncOpenAI

# ── File paths ────────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
INCIDENTS_FILE = EVAL_DIR / "synthetic_incidents.json"
RESULTS_FILE = EVAL_DIR / "eval_results.json"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_recall"]

# Seconds to sleep between Groq API calls to stay under the 6 000 TPM free tier.
# Each scenario triggers ~3 LLM calls (agent + RAGAS faithfulness + context_recall).
_THROTTLE_S = 15


# ── Embeddings wrapper ────────────────────────────────────────────────────────

class _SentenceTransformerEmbedding(BaseRagasEmbedding):
    """Wraps the cached sentence-transformers model already in use by miip."""

    def embed_text(self, text: str) -> list[float]:
        return _embed([text], settings.embedding_model)[0]

    async def aembed_text(self, text: str) -> list[float]:
        return self.embed_text(text)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_state(scenario: dict) -> IncidentState:
    log_text = "\n".join(scenario["log_lines"])
    return {
        "incident_id": scenario["incident_id"],
        "alert": {
            "alert_id": scenario["incident_id"],
            "title": scenario["alert_message"],
            "severity": "critical",
            "service": "unknown",
            "description": scenario["alert_message"] + "\n\n" + log_text,
            "timestamp": scenario["log_lines"][0][:20] if scenario["log_lines"] else "",
            "metadata": {},
        },
        "log_paths": [],
        "screenshot_paths": [],
        "voice_memo_paths": [],
        "route": ["log"],
        "log_analysis": None,
        "vision_analysis": None,
        "asr_transcription": None,
        "triage_summary": None,
        "draft_response": None,
        "severity_override": None,
        "errors": [],
        "completed_agents": [],
    }


def _safe_score(metric, **kwargs) -> float:
    """Call metric.score(), return NaN on any error."""
    try:
        result = metric.score(**kwargs)
        v = float(result.value)
        return v if not math.isnan(v) else float("nan")
    except Exception as exc:
        print(f"    [score error — {type(metric).__name__}]: {exc}")
        return float("nan")


def _fmt(v: float) -> str:
    return f"{v:.4f}" if not math.isnan(v) else "   N/A"


def _nan_to_none(v: float) -> float | None:
    return None if math.isnan(v) else round(v, 4)


# ── Main runner ───────────────────────────────────────────────────────────────

def run_eval() -> None:
    scenarios: list[dict] = json.loads(INCIDENTS_FILE.read_text())
    print(f"Loaded {len(scenarios)} scenarios from {INCIDENTS_FILE.name}\n")

    # ── Build evaluator LLM + embeddings ───────────────────────────────────────
    # openai.AsyncOpenAI aimed at Groq's OpenAI-compatible endpoint is the only
    # client type that ragas llm_factory correctly detects as async (is_async=True).
    # groq.AsyncGroq is not recognised by instructor.from_openai and returns NoneType.
    groq_as_openai = AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    evaluator_llm = llm_factory("llama-3.3-70b-versatile", client=groq_as_openai)
    evaluator_embeddings = _SentenceTransformerEmbedding()

    faithfulness_metric   = Faithfulness(llm=evaluator_llm)
    answer_rel_metric     = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    context_recall_metric = ContextRecall(llm=evaluator_llm)

    # ── Phase 1: run log_rag_agent for every scenario ──────────────────────────
    # Run in order so pgvector accumulates prior incidents; later scenarios
    # benefit from cross-incident similarity retrieval.
    print("=" * 50)
    print("Phase 1 — generating answers via log_rag_agent")
    print("=" * 50)

    agent_outputs: list[dict] = []

    for idx, scenario in enumerate(scenarios, 1):
        inc_id = scenario["incident_id"]
        print(f"\n[{idx}/{len(scenarios)}] {inc_id}")

        result = log_rag_agent(_build_state(scenario))
        answer: str = result.get("log_analysis") or ""

        agent_outputs.append({
            "incident_id": inc_id,
            "question":      scenario["question"],
            "answer":        answer,
            "contexts":      scenario["log_lines"],
            "ground_truth":  scenario["ground_truth"],
        })

        snippet = answer[:110].replace("\n", " ")
        print(f"  answer ({len(answer)} chars): {snippet}...")

        if idx < len(scenarios):
            time.sleep(_THROTTLE_S)  # stay under Groq TPM limit

    # ── Phase 2: score each sample with RAGAS ─────────────────────────────────
    print(f"\n{'=' * 50}")
    print("Phase 2 — RAGAS scoring (3 metrics × 5 samples)")
    print(f"{'=' * 50}\n")

    per_scenario_rows: list[dict] = []

    for idx, out in enumerate(agent_outputs, 1):
        print(f"[{idx}/{len(agent_outputs)}] Scoring {out['incident_id']}...")

        f_score = _safe_score(
            faithfulness_metric,
            user_input=out["question"],
            response=out["answer"],
            retrieved_contexts=out["contexts"],
        )
        time.sleep(_THROTTLE_S)

        ar_score = _safe_score(
            answer_rel_metric,
            user_input=out["question"],
            response=out["answer"],
        )
        time.sleep(_THROTTLE_S)

        cr_score = _safe_score(
            context_recall_metric,
            user_input=out["question"],
            retrieved_contexts=out["contexts"],
            reference=out["ground_truth"],
        )
        if idx < len(agent_outputs):
            time.sleep(_THROTTLE_S)

        print(f"  faithfulness={_fmt(f_score)}  answer_relevancy={_fmt(ar_score)}  context_recall={_fmt(cr_score)}")

        per_scenario_rows.append({
            "incident_id":     out["incident_id"],
            "faithfulness":    f_score,
            "answer_relevancy": ar_score,
            "context_recall":  cr_score,
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    SEP = "=" * 67
    print(f"\n{SEP}")
    print("RAGAS EVALUATION RESULTS")
    print(SEP)
    print(
        f"{'Incident':<20} "
        f"{'Faithfulness':>14} "
        f"{'Ans Relevancy':>15} "
        f"{'Ctx Recall':>12}"
    )
    print("-" * 67)

    totals = {k: [] for k in METRIC_COLS}
    serialisable_rows: list[dict] = []

    for row in per_scenario_rows:
        f  = row["faithfulness"]
        ar = row["answer_relevancy"]
        cr = row["context_recall"]

        print(
            f"{row['incident_id']:<20} "
            f"{_fmt(f):>14} "
            f"{_fmt(ar):>15} "
            f"{_fmt(cr):>12}"
        )

        for col, val in zip(METRIC_COLS, [f, ar, cr]):
            if not math.isnan(val):
                totals[col].append(val)

        serialisable_rows.append({
            "incident_id":      row["incident_id"],
            "faithfulness":     _nan_to_none(f),
            "answer_relevancy": _nan_to_none(ar),
            "context_recall":   _nan_to_none(cr),
        })

    print("-" * 67)
    means = {
        col: (sum(vals) / len(vals) if vals else float("nan"))
        for col, vals in totals.items()
    }
    print(
        f"{'MEAN':<20} "
        f"{_fmt(means['faithfulness']):>14} "
        f"{_fmt(means['answer_relevancy']):>15} "
        f"{_fmt(means['context_recall']):>12}"
    )
    print(SEP)

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "run_timestamp":  datetime.now(timezone.utc).isoformat(),
        "model":          "llama-3.3-70b-versatile",
        "embedding_model": settings.embedding_model,
        "mean_scores": {
            col: (_nan_to_none(means[col]) if means[col] == means[col] else None)
            for col in METRIC_COLS
        },
        "per_scenario": serialisable_rows,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved -> {RESULTS_FILE}")


if __name__ == "__main__":
    run_eval()
