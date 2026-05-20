from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from miip.graph import graph
from miip.state import AlertPayload, IncidentState

router = APIRouter()


class IngestRequest(BaseModel):
    alert: AlertPayload
    log_paths: list[str] = Field(default_factory=list)
    screenshot_paths: list[str] = Field(default_factory=list)
    voice_memo_paths: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    incident_id: str
    triage_summary: str | None
    draft_response: str | None
    completed_agents: list[str]
    errors: list[str]


@router.post("/ingest", response_model=IngestResponse, status_code=200)
async def ingest(payload: IngestRequest) -> IngestResponse:
    incident_id = str(uuid.uuid4())

    initial_state: IncidentState = {
        "incident_id": incident_id,
        "alert": payload.alert,
        "log_paths": payload.log_paths,
        "screenshot_paths": payload.screenshot_paths,
        "voice_memo_paths": payload.voice_memo_paths,
        "route": [],
        "log_analysis": None,
        "vision_analysis": None,
        "asr_transcription": None,
        "triage_summary": None,
        "draft_response": None,
        "severity_override": None,
        "errors": [],
        "completed_agents": [],
    }

    trace_config = {
        "run_name": f"incident-triage-{incident_id[:8]}",
        "tags": ["miip", "incident-triage", payload.alert.get("severity", "unknown")],  # type: ignore[union-attr]
        "metadata": {
            "incident_id": incident_id,
            "service":     payload.alert.get("service", ""),   # type: ignore[union-attr]
            "severity":    payload.alert.get("severity", ""),  # type: ignore[union-attr]
            "title":       payload.alert.get("title", ""),     # type: ignore[union-attr]
        },
    }

    try:
        final_state: dict[str, Any] = await graph.ainvoke(initial_state, config=trace_config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        incident_id=incident_id,
        triage_summary=final_state.get("triage_summary"),
        draft_response=final_state.get("draft_response"),
        completed_agents=final_state.get("completed_agents", []),
        errors=final_state.get("errors", []),
    )
