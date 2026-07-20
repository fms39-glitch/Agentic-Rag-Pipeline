from typing import Literal

from langgraph.graph import END, START, StateGraph

from nodes.fallback import FALLBACK_DISCLAIMER, fallback_node, is_insufficient_answer
from nodes.generate import generate_node
from nodes.grade import grade_node
from nodes.retrieve import retrieve_node
from nodes.rewrite import rewrite_node
from nodes.verify import verify_node
from state import DEFAULT_MAX_RETRIES, AgentState

UNVERIFIED_DISCLAIMER = (
    "\n\n[Disclaimer: This answer could not be fully verified against the source documents.]"
)


def finish_fallback(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": state["draft_answer"] + FALLBACK_DISCLAIMER,
        "status": "llm_fallback",
    }


def finish_success(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": state["draft_answer"],
        "status": "success",
    }


def finish_unverified(state: AgentState) -> AgentState:
    return {
        **state,
        "final_answer": state["draft_answer"] + UNVERIFIED_DISCLAIMER,
        "status": "unverified",
    }


def route_after_grade(
    state: AgentState,
) -> Literal["generate", "rewrite", "fallback"]:
    if state["graded_chunks"]:
        return "generate"
    if state["retry_count"] < state["max_retries"]:
        return "rewrite"
    return "fallback"


def route_after_generate(
    state: AgentState,
) -> Literal["verify", "fallback"]:
    if is_insufficient_answer(state["draft_answer"]):
        return "fallback"
    return "verify"


def route_after_verify(
    state: AgentState,
) -> Literal["finish_success", "generate", "finish_unverified"]:
    if state["is_grounded"]:
        return "finish_success"
    if state["retry_count"] < state["max_retries"]:
        return "generate"
    return "finish_unverified"


def _build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("finish_fallback", finish_fallback)
    builder.add_node("finish_success", finish_success)
    builder.add_node("finish_unverified", finish_unverified)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade)
    builder.add_edge("rewrite", "retrieve")
    builder.add_conditional_edges("generate", route_after_generate)
    builder.add_conditional_edges("verify", route_after_verify)
    builder.add_edge("fallback", "finish_fallback")

    builder.add_edge("finish_fallback", END)
    builder.add_edge("finish_success", END)
    builder.add_edge("finish_unverified", END)

    return builder.compile()


_graph = _build_graph()


def run_query(question: str) -> dict:
    initial_state: AgentState = {
        "question": question,
        "retrieved_chunks": [],
        "graded_chunks": [],
        "draft_answer": "",
        "is_grounded": False,
        "retry_count": 0,
        "max_retries": DEFAULT_MAX_RETRIES,
        "final_answer": "",
        "status": "",
    }

    final_state = _graph.invoke(initial_state)

    return {
        "answer": final_state["final_answer"],
        "status": final_state["status"],
        "question": final_state["question"],
        "draft_answer": final_state["draft_answer"],
        "retrieved_chunks": final_state["retrieved_chunks"],
        "graded_chunks": final_state["graded_chunks"],
        "retry_count": final_state["retry_count"],
        "is_grounded": final_state["is_grounded"],
    }
