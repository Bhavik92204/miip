from __future__ import annotations

from typing import Any, Optional

from typing_extensions import TypedDict


class AlertPayload(TypedDict):
    alert_id: str
    title: str
    severity: str          # critical | high | medium | low
    service: str
    description: str
    timestamp: str
    metadata: dict[str, Any]


class IncidentState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    incident_id: str
    alert: AlertPayload
    log_paths: list[str]           # paths or inline log text
    screenshot_paths: list[str]    # paths to dashboard screenshots
    voice_memo_paths: list[str]    # paths to audio files

    # ── Routing decision ──────────────────────────────────────────────────────
    route: list[str]               # e.g. ["log", "vision", "asr"]

    # ── Per-agent outputs ─────────────────────────────────────────────────────
    log_analysis: Optional[str]
    vision_analysis: Optional[str]
    asr_transcription: Optional[str]

    # ── Final outputs ─────────────────────────────────────────────────────────
    triage_summary: Optional[str]
    draft_response: Optional[str]
    severity_override: Optional[str]

    # ── Execution metadata ────────────────────────────────────────────────────
    errors: list[str]
    completed_agents: list[str]
