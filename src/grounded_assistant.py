"""Answer fixed Marshal questions strictly from selected saved evidence."""


GROUNDED_QUESTIONS = (
    "Why was this case surfaced?",
    "What evidence is available?",
    "What must a human review?",
    "What did reviewers record?",
    "What is withheld?",
)
GROUNDED_ASSISTANT_DISCLAIMER = (
    "This is a fixed-question evidence explainer, not a generative decision-maker. "
    "Every answer is composed only from the selected saved case and recorded human review material."
)


def _source_is_available(source):
    """Treat only a real file or a positive available-count as available."""
    status = str(source["status"])
    if status == "Available":
        return True
    count_and_label = status.split(maxsplit=1)
    return (
        len(count_and_label) == 2
        and count_and_label[0].isdigit()
        and int(count_and_label[0]) > 0
        and count_and_label[1] == "available"
    )


def build_grounded_case_answer(question, case, ladder, correlation, case_rows, ledger_rows):
    """Return a constrained, cited answer without adding facts beyond saved records."""
    if question not in GROUNDED_QUESTIONS:
        raise ValueError("Choose one of the supported evidence questions.")
    sources = ["Selected case", "Decision ladder", "Evidence correlation"]
    vehicle_label = ", ".join(map(str, case["track_ids"]))
    if question == "Why was this case surfaced?":
        answer = (
            f"Vehicles {vehicle_label} have {len(case_rows)} saved evidence record(s) from "
            f"{case['start_time_seconds']:.2f}s to {case['end_time_seconds']:.2f}s. "
            f"Recorded evidence types: {', '.join(ladder['source_evidence_types']) or 'none'}. "
            f"The current review recommendation is {ladder['status']}."
        )
    elif question == "What evidence is available?":
        available = [item["source"] for item in correlation if _source_is_available(item)]
        answer = "Available sources: " + (", ".join(available) if available else "none recorded for this case") + "."
    elif question == "What must a human review?":
        pending = [item["evidence_source"] for item in ladder["checks"] if item["status"] in {"Pending", "Unavailable", "Withheld"}]
        answer = ladder["recommended_action"] + " Outstanding or withheld checks: " + (", ".join(pending) if pending else "none recorded") + "."
    elif question == "What did reviewers record?":
        if ledger_rows:
            answer = f"{len(ledger_rows)} saved human review record(s) are available in the decision ledger. The latest is {ledger_rows[-1]['record_type']}: {ledger_rows[-1]['decision_or_action']}."
            sources.append("Decision ledger")
        else:
            answer = "No saved human review, race-control action, correction, or independent second opinion is recorded for this run."
    else:
        withheld = [item["evidence_source"] for item in ladder["checks"] if item["status"] in {"Unavailable", "Withheld"}]
        unavailable_sources = [item["source"] for item in correlation if "unavailable" in str(item["status"]).lower() or item["status"] == "Not attached"]
        answer = "Withheld or unavailable: " + (", ".join(withheld + unavailable_sources) if withheld or unavailable_sources else "no selected-case item is currently marked withheld") + "."
    return {"question": question, "answer": answer, "sources": sources, "disclaimer": GROUNDED_ASSISTANT_DISCLAIMER}
