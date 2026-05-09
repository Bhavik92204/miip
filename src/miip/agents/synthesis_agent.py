from __future__ import annotations

import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from miip.config import settings
from miip.state import IncidentState

log = structlog.get_logger()

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert SRE incident commander. Synthesise inputs from multiple analysis agents "
        "and produce a concise, actionable incident triage report.\n\n"
        "Structure your response as markdown with these exact sections:\n"
        "## Root Cause\n"
        "## Impacted Services\n"
        "## Severity Assessment\n"
        "## Recommended Actions\n"
        "## Draft Incident Response\n"
    )),
    ("human", (
        "# Incident {incident_id}\n"
        "**Service:** {service} | **Severity:** {severity} | **Title:** {title}\n\n"
        "{log_section}"
        "{vision_section}"
        "{asr_section}"
        "Synthesise the above into an actionable triage report:"
    )),
])


def _section(heading: str, content: str | None) -> str:
    if not content:
        return ""
    return f"### {heading}\n{content}\n\n"


def synthesis_agent(state: IncidentState) -> IncidentState:
    log.info("synthesis_agent invoked", incident_id=state["incident_id"])

    alert = state["alert"]
    log_analysis = state.get("log_analysis")
    vision_analysis = state.get("vision_analysis")
    asr_transcription = state.get("asr_transcription")

    # Assemble triage_summary from all upstream outputs for storage / fallback display
    sections: list[str] = []
    if log_analysis:
        sections.append(f"**Log Analysis**\n{log_analysis}")
    if vision_analysis:
        sections.append(f"**Vision Analysis**\n{vision_analysis}")
    if asr_transcription:
        sections.append(f"**Voice Transcription**\n{asr_transcription}")

    triage_summary = "\n\n".join(sections) or "[No upstream analysis available.]"

    draft_response: str
    if not settings.groq_api_key:
        log.info("groq key absent — using raw triage summary", incident_id=state["incident_id"])
        draft_response = (
            f"## Incident {state['incident_id']} — {alert['severity'].upper()}\n\n"
            f"**Service:** {alert['service']}\n\n"
            f"**Title:** {alert['title']}\n\n"
            f"{triage_summary}\n\n"
            "**Recommended Action:** [Groq key not configured — manual review required]"
        )
    else:
        try:
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
                api_key=settings.groq_api_key,
            )
            result = (_PROMPT | llm).invoke({
                "incident_id": state["incident_id"],
                "service": alert["service"],
                "severity": alert["severity"],
                "title": alert["title"],
                "log_section": _section("Log Analysis", log_analysis),
                "vision_section": _section("Vision Analysis", vision_analysis),
                "asr_section": _section("Voice Transcription", asr_transcription),
            })
            draft_response = result.content
            log.info("synthesis LLM complete", incident_id=state["incident_id"])
        except Exception as exc:
            log.warning("synthesis LLM failed", error=str(exc))
            draft_response = (
                f"[Synthesis LLM error: {exc}]\n\n"
                f"## Incident {state['incident_id']} — {alert['severity'].upper()}\n\n"
                f"{triage_summary}"
            )

    return {
        **state,
        "triage_summary": triage_summary,
        "draft_response": draft_response,
        "completed_agents": [*state.get("completed_agents", []), "synthesis"],
    }
