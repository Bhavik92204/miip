from __future__ import annotations

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pathlib import Path
from sqlalchemy.orm import Session

from miip.config import settings
from miip.db import LogChunk, get_session
from miip.embeddings import embed
from miip.state import IncidentState

log = structlog.get_logger()

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 5

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " "],
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an SRE on-call assistant. Analyse the log excerpts and identify:\n"
        "1. Error patterns and anomalies\n"
        "2. Affected components / services\n"
        "3. Likely root cause (1-2 sentences)\n"
        "4. Severity assessment\n\n"
        "Respond in structured markdown."
    )),
    ("human", (
        "## Current incident\n{alert_context}\n\n"
        "## Log excerpts (current incident)\n{current_logs}\n\n"
        "## Similar past incidents (retrieved from vector store)\n{similar_logs}\n\n"
        "Provide your analysis:"
    )),
])


def _load_log_text(path_or_text: str) -> str:
    p = Path(path_or_text)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    return path_or_text  # treat as inline log text


def _store_and_retrieve(
    session: Session,
    incident_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> list[str]:
    for text, vec in zip(chunks, embeddings):
        session.add(LogChunk(incident_id=incident_id, chunk_text=text, embedding=vec))
    session.commit()

    # Cosine similarity search against chunks from *other* incidents
    query_vec = embeddings[0]
    results = (
        session.query(LogChunk)
        .filter(LogChunk.incident_id != incident_id)
        .order_by(LogChunk.embedding.cosine_distance(query_vec))
        .limit(TOP_K)
        .all()
    )
    return [r.chunk_text for r in results]


def log_rag_agent(state: IncidentState) -> IncidentState:
    log.info("log_rag_agent invoked", incident_id=state["incident_id"])

    # ── 1. Load raw log text ──────────────────────────────────────────────────
    raw_sources = state.get("log_paths") or []
    raw_texts = [_load_log_text(p) for p in raw_sources]
    if not raw_texts:
        raw_texts = [state["alert"]["description"]]

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    chunks: list[str] = []
    for text in raw_texts:
        chunks.extend(_splitter_split(text))

    if not chunks:
        return {
            **state,
            "log_analysis": "No log content to analyse.",
            "completed_agents": [*state.get("completed_agents", []), "log_rag"],
        }

    # ── 3. Embed ──────────────────────────────────────────────────────────────
    embeddings: list[list[float]] | None = None
    try:
        embeddings = embed(chunks, settings.embedding_model)
        log.info("embeddings generated", n_chunks=len(chunks))
    except Exception as exc:
        log.warning("embedding failed — skipping pgvector", error=str(exc))

    # ── 4. Store current chunks + retrieve similar past chunks ────────────────
    similar_chunks: list[str] = []
    if embeddings and settings.database_url:
        try:
            with get_session(settings.database_url) as session:
                similar_chunks = _store_and_retrieve(
                    session, state["incident_id"], chunks, embeddings
                )
            log.info("pgvector retrieved", n_similar=len(similar_chunks))
        except Exception as exc:
            log.warning("pgvector query failed", error=str(exc))

    # ── 5. LLM triage summary ─────────────────────────────────────────────────
    analysis: str
    if not settings.groq_api_key:
        analysis = (
            "[Groq key not configured — raw excerpts]\n\n"
            + "\n---\n".join(chunks[:5])
        )
    else:
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=settings.groq_api_key,
            )
            alert = state["alert"]
            result = (_PROMPT | llm).invoke({
                "alert_context": (
                    f"Service: {alert['service']}\n"
                    f"Title: {alert['title']}\n"
                    f"Severity: {alert['severity']}\n"
                    f"Description: {alert['description']}"
                ),
                "current_logs": "\n---\n".join(chunks[:10]),
                "similar_logs": (
                    "\n---\n".join(similar_chunks)
                    if similar_chunks
                    else "No similar past incidents found."
                ),
            })
            analysis = result.content
            log.info("LLM analysis complete", incident_id=state["incident_id"])
        except Exception as exc:
            log.warning("LLM call failed", error=str(exc))
            analysis = f"[LLM error: {exc}]\n\nRaw excerpts:\n" + "\n---\n".join(chunks[:5])

    return {
        **state,
        "log_analysis": analysis,
        "completed_agents": [*state.get("completed_agents", []), "log_rag"],
    }


def _splitter_split(text: str) -> list[str]:
    return _SPLITTER.split_text(text)
