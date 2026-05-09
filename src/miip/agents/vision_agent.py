from __future__ import annotations

import base64
import io

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from PIL import Image
from pathlib import Path

from miip.config import settings
from miip.state import IncidentState

log = structlog.get_logger()

# Resize before encoding to keep token count manageable
_MAX_PX = 1024

_SYSTEM_PROMPT = (
    "You are an SRE on-call assistant specialising in dashboard analysis. "
    "Examine the provided screenshot(s) and identify:\n"
    "1. Metric anomalies (spikes, drops, flatlines)\n"
    "2. Error rate or latency changes\n"
    "3. Saturation indicators (CPU, memory, connection pools)\n"
    "4. Affected time window visible in the chart\n"
    "5. Likely impacted services based on what you see\n\n"
    "Respond in structured markdown."
)

_USER_TEMPLATE = (
    "## Incident context\n"
    "Service: {service}\n"
    "Title: {title}\n"
    "Severity: {severity}\n\n"
    "## Dashboard screenshots ({n} attached)\n"
    "Analyse each screenshot and report your findings:"
)


def _encode_image(path: str) -> tuple[str, str]:
    """Load, resize, and base64-encode an image. Returns (b64_string, mime_type)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((_MAX_PX, _MAX_PX), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/png"


def vision_agent(state: IncidentState) -> IncidentState:
    log.info("vision_agent invoked", incident_id=state["incident_id"])

    screenshot_paths = state.get("screenshot_paths") or []

    # ── No screenshots — skip cleanly ─────────────────────────────────────────
    if not screenshot_paths:
        return {
            **state,
            "vision_analysis": None,
            "completed_agents": [*state.get("completed_agents", []), "vision"],
        }

    # ── No API key — return early ─────────────────────────────────────────────
    if not settings.groq_api_key:
        return {
            **state,
            "vision_analysis": (
                f"[Groq key not configured] {len(screenshot_paths)} screenshot(s) received but not analysed."
            ),
            "completed_agents": [*state.get("completed_agents", []), "vision"],
        }

    # ── Load and encode images ────────────────────────────────────────────────
    encoded: list[tuple[str, str]] = []
    for path in screenshot_paths:
        try:
            b64, mime = _encode_image(path)
            encoded.append((b64, mime))
            log.info("image encoded", path=path)
        except Exception as exc:
            log.warning("failed to load image", path=path, error=str(exc))

    if not encoded:
        return {
            **state,
            "vision_analysis": f"[Could not load any of {len(screenshot_paths)} screenshot(s)]",
            "completed_agents": [*state.get("completed_agents", []), "vision"],
        }

    # ── Build multimodal message ──────────────────────────────────────────────
    alert = state["alert"]
    content: list[dict] = [
        {
            "type": "text",
            "text": _USER_TEMPLATE.format(
                service=alert["service"],
                title=alert["title"],
                severity=alert["severity"],
                n=len(encoded),
            ),
        }
    ]
    for b64, mime in encoded:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    # ── Call Groq vision model ────────────────────────────────────────────────
    analysis: str
    try:
        llm = ChatGroq(
            model=settings.vision_llm_model,
            temperature=0,
            api_key=settings.groq_api_key,
        )
        result = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=content),
        ])
        analysis = result.content
        log.info("vision analysis complete", incident_id=state["incident_id"], n_images=len(encoded))
    except Exception as exc:
        log.warning("vision LLM call failed", error=str(exc))
        analysis = f"[Vision analysis error: {exc}]"

    return {
        **state,
        "vision_analysis": analysis,
        "completed_agents": [*state.get("completed_agents", []), "vision"],
    }
