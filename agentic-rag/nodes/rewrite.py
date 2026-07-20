from llm import chat_completion
from state import AgentState

MAX_TOKENS = 256


def rewrite_node(state: AgentState) -> AgentState:
    if state["graded_chunks"]:
        return state

    question = state["question"]
    prompt = (
        "Rewrite the following question into a clearer or differently-phrased version "
        "that might retrieve better results from a document search system. "
        "Expand acronyms, add specificity, and keep the same intent. "
        "Reply with only the rewritten question, nothing else.\n\n"
        f"Question: {question}"
    )

    print("Rewriting question for better retrieval...")
    response = chat_completion(
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    rewritten = (response.choices[0].message.content or question).strip()

    print(f"Rewrote question (retry {state['retry_count'] + 1}): {rewritten}")

    return {
        **state,
        "question": rewritten,
        "retry_count": state["retry_count"] + 1,
    }
