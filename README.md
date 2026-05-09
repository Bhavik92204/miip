# MIIP - Multimodal Incident Intelligence Platform

An LLM-powered SRE on-call co-pilot that triages incidents from logs, screenshots, and voice memos.

## Overview

MIIP accepts an alert payload alongside raw incident artefacts — log files, dashboard screenshots, and voice memos — and routes them through a five-agent LangGraph pipeline. Each agent analyses one modality independently, and a synthesis agent combines all findings into a structured triage report with root cause, severity, impacted services, and recommended actions. The platform exposes a single REST endpoint so it can be called from PagerDuty webhooks, Slack workflows, or CI pipelines.

## Architecture

```
Incoming alert
      |
      v
 +-----------+
 |  router   |  Classifies severity, determines which agents to activate
 +-----------+
      |
      v
 +-------------+
 | log_rag     |  Chunks & embeds log lines into pgvector, retrieves similar
 |   _agent    |  historical incidents, runs Groq LLM triage over context
 +-------------+
      |
      v
 +-------------+
 | vision      |  Sends dashboard screenshots to Groq vision model,
 |   _agent    |  extracts anomalies and metric spikes
 +-------------+
      |
      v
 +-------------+
 | asr         |  Transcribes voice memos via Whisper-base (HuggingFace),
 |   _agent    |  scipy WAV loading -- no ffmpeg dependency
 +-------------+
      |
      v
 +------------------+
 | synthesis_agent  |  Merges all modality analyses via Groq LLM into a
 |                  |  structured markdown incident report
 +------------------+
      |
      v
 POST /ingest response:
   triage_summary, draft_response,
   completed_agents, errors
```

## HuggingFace Tasks Used

| Task | Model |
|---|---|
| Automatic Speech Recognition | `openai/whisper-base` |
| Sentence Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |

## Tech Stack

| Component | Technology |
|---|---|
| API framework | FastAPI + uvicorn |
| Agent orchestration | LangGraph |
| LLM inference | Groq (`llama-3.1-8b-instant`) |
| Speech-to-text | HuggingFace Transformers — Whisper-base |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | pgvector (PostgreSQL 16) |
| ORM | SQLAlchemy 2.0 |
| WAV processing | scipy (no ffmpeg required) |
| Eval framework | RAGAS 0.4.x |
| Package build | Hatchling |
| Linting / types | Ruff, mypy |

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for pgvector)
- A [Groq API key](https://console.groq.com) (free tier works)
- A HuggingFace token with read access

### Install

```bash
git clone https://github.com/Bhavik92204/miip.git
cd miip
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env and fill in:
#   GROQ_API_KEY=gsk_...
#   HF_TOKEN=hf_...
```

### Start pgvector

```bash
docker-compose up -d
```

The container runs PostgreSQL 16 with the `vector` extension on `localhost:5432`. The init script creates the extension automatically on first start.

### Start the server

```bash
uvicorn miip.main:app --reload
```

API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Example request

```bash
curl -s -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "alert": {
      "title": "HikariPool-1 connection timeout",
      "severity": "critical",
      "service": "payments-api"
    },
    "log_paths": ["/var/log/payments/app.log"],
    "screenshot_paths": [],
    "voice_memo_paths": []
  }' | python -m json.tool
```

## RAGAS Eval Results

Evaluated against 5 synthetic SRE incident scenarios using `llama-3.3-70b-versatile` as the judge model.

| Metric | Score | Threshold | Status |
|---|---|---|---|
| context_recall | 0.91 | >= 0.70 | PASS |
| answer_relevancy | 0.56 | >= 0.50 | PASS |
| faithfulness | N/A | skipped | Groq free-tier 1024-token cap truncates NLI JSON |

Run the eval harness:

```bash
make eval
```

The `make eval` target runs `tests/eval/run_eval.py` (generates `eval_results.json`) then the pytest CI gate (`tests/eval/test_ci_gate.py`).

## Project Structure

```
miip/
├── docker-compose.yml          # pgvector service (pgvector/pgvector:pg16)
├── Makefile                    # eval and test targets
├── pyproject.toml
├── scripts/
│   └── init.sql                # CREATE EXTENSION vector
├── src/miip/
│   ├── agents/
│   │   ├── asr_agent.py        # Whisper transcription via scipy WAV loading
│   │   ├── log_rag_agent.py    # pgvector store + cosine retrieval + Groq triage
│   │   ├── router_agent.py     # Alert classification and routing
│   │   ├── synthesis_agent.py  # Groq LLM structured report generation
│   │   └── vision_agent.py     # Groq vision model screenshot analysis
│   ├── api/
│   │   └── routes.py           # POST /ingest endpoint
│   ├── config.py               # Pydantic Settings from .env
│   ├── db.py                   # SQLAlchemy LogChunk model + pgvector
│   ├── embeddings.py           # SentenceTransformer with lru_cache
│   ├── graph.py                # LangGraph 5-node pipeline
│   ├── main.py                 # FastAPI app entry point
│   └── state.py                # IncidentState TypedDict
└── tests/
    └── eval/
        ├── synthetic_incidents.json   # 5 SRE incident scenarios
        ├── run_eval.py                # RAGAS evaluation runner
        ├── test_ci_gate.py            # pytest CI assertions
        └── eval_results.json          # Last recorded scores
```
