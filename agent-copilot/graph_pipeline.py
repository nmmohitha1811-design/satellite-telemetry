from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from detector_agent import detect_anomaly
from retrieval_agent import retrieve
from report_agent import write_report

SIMILARITY_THRESHOLD = 0.35


class PipelineState(TypedDict):
    channel_id: Optional[str]
    spacecraft: Optional[str]
    anomaly_class: Optional[str]
    num_true_anomalies: Optional[int]
    detection_summary: Optional[str]

    retrieved_chunks: Optional[List[dict]]
    best_retrieval_score: Optional[float]

    draft_report: Optional[str]
    final_report: Optional[str]
    is_grounded: Optional[bool]

def detector_node(state: PipelineState) -> PipelineState:
    result = detect_anomaly(state.get("channel_id"))
    return {**state, **result}


def retrieval_node(state: PipelineState) -> PipelineState:
    query, result = retrieve(state)
    print(f"[Retrieval Agent] Query: \"{query}\"")
    return {**state, **result}


def report_node(state: PipelineState) -> PipelineState:
    draft = write_report(state, state["retrieved_chunks"])
    return {**state, "draft_report": draft}


def guardrail_node(state: PipelineState) -> PipelineState:
    score = state["best_retrieval_score"]
    is_grounded = score >= SIMILARITY_THRESHOLD

    if is_grounded:
        final_report = state["draft_report"]
    else:
        final_report = (
            "INSUFFICIENT GROUNDING: No sufficiently relevant source material "
            f"was found (best match score: {score:.3f}, threshold: {SIMILARITY_THRESHOLD}). "
            "This anomaly should be escalated to a human operator for manual review."
        )

    return {**state, "is_grounded": is_grounded, "final_report": final_report}


def refuse_node(state: PipelineState) -> PipelineState:
    """Used when retrieval score is too low — skip the LLM call entirely."""
    score = state["best_retrieval_score"]
    final_report = (
        "INSUFFICIENT GROUNDING: No sufficiently relevant source material "
        f"was found (best match score: {score:.3f}, threshold: {SIMILARITY_THRESHOLD}). "
        "Skipping report generation. This anomaly should be escalated to a human operator."
    )
    return {**state, "is_grounded": False, "draft_report": None, "final_report": final_report}


def route_after_retrieval(state: PipelineState) -> str:
    """Conditional edge: decide whether it's worth calling the LLM at all."""
    if state["best_retrieval_score"] >= SIMILARITY_THRESHOLD:
        return "report"
    else:
        return "refuse"
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("detect", detector_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("report", report_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("refuse", refuse_node)

    graph.set_entry_point("detect")
    graph.add_edge("detect", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieval, {
        "report": "report",
        "refuse": "refuse",
    })
    graph.add_edge("report", "guardrail")
    graph.add_edge("guardrail", END)
    graph.add_edge("refuse", END)

    return graph.compile()


if __name__ == "__main__":
    pipeline = build_graph()

    print("=" * 60)
    print("Running pipeline (default: first caught anomaly)")
    print("=" * 60)

    result = pipeline.invoke({})

    print(f"\nChannel: {result['channel_id']} ({result['spacecraft']})")
    print(f"Detection: {result['detection_summary']}")
    print(f"Retrieval score: {result['best_retrieval_score']:.3f}")
    print(f"Grounded: {result['is_grounded']}")
    print(f"\n--- FINAL REPORT ---\n{result['final_report']}")