from llm import chat_completion
from state import AgentState

MAX_TOKENS = 1024
INSUFFICIENT_PHRASE = "i don't have enough information"
FALLBACK_DISCLAIMER = (
    "\n\n---\n*This answer is from the AI model's general knowledge, "
    "not from your uploaded documents.*"
)


def is_insufficient_answer(text: str) -> bool:
    return INSUFFICIENT_PHRASE in text.lower()


def fallback_node(state: AgentState) -> AgentState:
    prompt = (
        "Answer the following question using your general knowledge.\n"
        "Be helpful, accurate, and concise. If you are uncertain, say so.\n\n"
        f"Question: {state['question']}"
    )

    print("No answer in documents — using general LLM knowledge...")
    response = chat_completion(
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = (response.choices[0].message.content or "").strip()
    print("Fallback answer generated.")

    return {**state, "draft_answer": answer}
