from __future__ import annotations

import structlog

from miip.state import IncidentState

log = structlog.get_logger()


def router_agent(state: IncidentState) -> IncidentState:
    """
    Inspects the incoming incident and decides which downstream agents to invoke.
    Populates `route` with the set of agents that have relevant inputs.
    Stub: presence of input paths drives routing; no LLM call yet.
    """
    log.info("router_agent invoked", incident_id=state["incident_id"])

    route: list[str] = []

    if state.get("log_paths"):
        route.append("log")
    if state.get("screenshot_paths"):
        route.append("vision")
    if state.get("voice_memo_paths"):
        route.append("asr")

    if not route:
        route.append("log")

    log.info("routing decision", route=route, incident_id=state["incident_id"])

    return {
        **state,
        "route": route,
        "errors": state.get("errors", []),
        "completed_agents": [*state.get("completed_agents", []), "router"],
    }
