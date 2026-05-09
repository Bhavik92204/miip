from __future__ import annotations

import logging

import structlog
from fastapi import FastAPI

from miip.api.routes import router as api_router

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


@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok"}
