SIMILARITY_THRESHOLD = 0.35


def apply_guardrail(retrieval_state, draft_report):
    """
    Decide whether the draft report is grounded enough to present,
    based on the measured similarity threshold from Layer 2 testing.
    """
    score = retrieval_state["best_retrieval_score"]
    is_grounded = score >= SIMILARITY_THRESHOLD

    if is_grounded:
        final_report = draft_report
    else:
        final_report = (
            "INSUFFICIENT GROUNDING: No sufficiently relevant source material "
            f"was found (best match score: {score:.3f}, threshold: {SIMILARITY_THRESHOLD}). "
            "This anomaly should be escalated to a human operator for manual review "
            "rather than relying on an unverified automated explanation."
        )

    return {
        "is_grounded": is_grounded,
        "final_report": final_report,
    }


if __name__ == "__main__":
    from detector_agent import detect_anomaly
    from retrieval_agent import retrieve
    from report_agent import write_report

    detector_state = detect_anomaly()
    query, retrieval_state = retrieve(detector_state)
    draft = write_report(detector_state, retrieval_state["retrieved_chunks"])

    guardrail_result = apply_guardrail(retrieval_state, draft)

    print("Detection:", detector_state["detection_summary"])
    print(f"Retrieval score: {retrieval_state['best_retrieval_score']:.3f}")
    print(f"Grounded: {guardrail_result['is_grounded']}")
    print("\n--- FINAL REPORT ---")
    print(guardrail_result["final_report"])