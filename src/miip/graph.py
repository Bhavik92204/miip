from __future__ import annotations

from langgraph.graph import END, StateGraph

from miip.agents.asr_agent import asr_agent
from miip.agents.log_rag_agent import log_rag_agent
from miip.agents.router_agent import router_agent
from miip.agents.synthesis_agent import synthesis_agent
from miip.agents.vision_agent import vision_agent
from miip.state import IncidentState


def _build_graph() -> StateGraph:
    builder: StateGraph = StateGraph(IncidentState)

    builder.add_node("router", router_agent)
    builder.add_node("log_rag", log_rag_agent)
    builder.add_node("vision", vision_agent)
    builder.add_node("asr", asr_agent)
    builder.add_node("synthesis", synthesis_agent)

    builder.set_entry_point("router")

    # Linear skeleton — replace with conditional fan-out once agents are implemented.
    # The router populates state["route"]; use add_conditional_edges to branch on it.
    builder.add_edge("router", "log_rag")
    builder.add_edge("log_rag", "vision")
    builder.add_edge("vision", "asr")
    builder.add_edge("asr", "synthesis")
    builder.add_edge("synthesis", END)

    return builder.compile()


graph = _build_graph()
