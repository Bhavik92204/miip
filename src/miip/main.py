from __future__ import annotations

import logging
import os
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from miip.api.routes import router as api_router
from miip.config import settings

def _configure_tracing() -> None:
    """Export LangSmith env vars from settings so LangChain picks them up."""
    if settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", settings.langchain_tracing_v2)
        os.environ.setdefault("LANGCHAIN_API_KEY",    settings.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT",    settings.langchain_project)


_configure_tracing()

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)

app = FastAPI(
    title="Multimodal Incident Intelligence Platform",
    description=(
        "On-call co-pilot: ingest alerts, logs, dashboard screenshots, "
        "and voice memos for AI-powered triage."
    ),
    version="0.1.0",
)

app.include_router(api_router, prefix="/api/v1")

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
