from __future__ import annotations

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph

from miip.agents.asr_agent import asr_agent
from miip.agents.log_rag_agent import log_rag_agent
from miip.agents.router_agent import router_agent
from miip.agents.synthesis_agent import synthesis_agent
from miip.agents.vision_agent import vision_agent
from miip.state import IncidentState


def _node(fn, name: str, tags: list[str]):
    return RunnableLambda(fn).with_config(run_name=name, tags=["miip", *tags])


def _build_graph() -> StateGraph:
    builder: StateGraph = StateGraph(IncidentState)

    builder.add_node("router",    _node(router_agent,    "router",    ["routing"]))
    builder.add_node("log_rag",   _node(log_rag_agent,   "log_rag",   ["rag", "llm"]))
    builder.add_node("vision",    _node(vision_agent,    "vision",    ["multimodal"]))
    builder.add_node("asr",       _node(asr_agent,       "asr",       ["speech"]))
    builder.add_node("synthesis", _node(synthesis_agent, "synthesis", ["llm"]))

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
