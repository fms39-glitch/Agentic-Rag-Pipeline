import time

from llm import GRADE_CALL_DELAY_SEC, chat_completion
from state import AgentState, RetrievedChunk


def _grade_chunk(question: str, content: str) -> bool:
    prompt = (
        f"Given the question '{question}', is the following chunk relevant enough "
        f"to help answer it? Reply with only YES or NO.\n\nChunk: {content}"
    )
    response = chat_completion(
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = (response.choices[0].message.content or "").strip().upper()
    return answer.startswith("YES")


def grade_node(state: AgentState) -> AgentState:
    question = state["question"]
    retrieved_chunks = state["retrieved_chunks"]
    graded_chunks: list[RetrievedChunk] = []

    print(f"Grading {len(retrieved_chunks)} chunks...")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        if _grade_chunk(question, chunk["content"]):
            graded_chunks.append(chunk)
        print(f"  Graded chunk {i}/{len(retrieved_chunks)}", end="\r", flush=True)
        if i < len(retrieved_chunks):
            time.sleep(GRADE_CALL_DELAY_SEC)

    print(f"Graded {len(graded_chunks)}/{len(retrieved_chunks)} chunks as relevant")

    return {**state, "graded_chunks": graded_chunks}
