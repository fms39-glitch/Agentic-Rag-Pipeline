from llm import chat_completion
from state import AgentState, RetrievedChunk

MAX_TOKENS = 128


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No context provided.)"

    sections: list[str] = []
    for chunk in chunks:
        label = f"[{chunk['source_file']}, p.{chunk['page_number']}]"
        sections.append(f"{label}\n{chunk['content']}")
    return "\n\n".join(sections)


def _build_prompt(draft_answer: str, context: str) -> str:
    return f"""You are verifying whether a draft answer is grounded in source material.

Check whether every factual claim in the draft answer is actually supported by the provided chunks.

Reply with only YES if every factual claim is supported.

Reply with NO followed by one sentence explaining any unsupported claim if any claim is not supported by the chunks.

Draft answer:
{draft_answer}

Source chunks:
{context}
"""


def verify_node(state: AgentState) -> AgentState:
    context = _format_context(state["graded_chunks"])
    prompt = _build_prompt(state["draft_answer"], context)

    print("Verifying grounding...")
    response = chat_completion(
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    verification = (response.choices[0].message.content or "").strip()
    first_line = verification.split("\n", 1)[0].strip().upper()
    is_grounded = first_line.startswith("YES")

    if is_grounded:
        print("Verification: grounded (YES)")
    else:
        print(f"Verification: not grounded (NO) — {verification}")

    updated: AgentState = {**state, "is_grounded": is_grounded}
    if not is_grounded:
        updated["retry_count"] = state["retry_count"] + 1

    return updated
