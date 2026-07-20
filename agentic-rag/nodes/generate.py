from llm import chat_completion
from state import AgentState, RetrievedChunk

MAX_TOKENS = 1024


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(No context provided.)"

    sections: list[str] = []
    for chunk in chunks:
        label = f"[{chunk['source_file']}, p.{chunk['page_number']}]"
        sections.append(f"{label}\n{chunk['content']}")
    return "\n\n".join(sections)


def _build_prompt(question: str, context: str) -> str:
    return f"""You are a knowledge assistant. Answer the question using ONLY the context below.

Rules:
1. Use ONLY the provided context. Do not use outside knowledge.
2. Cite the source file and page inline for every factual claim, using this format: [filename.pdf, p.N]
3. If the context does not contain enough information to answer the question, respond with exactly:
   "I don't have enough information to answer this"
   Do not guess or invent information.

Question:
{question}

Context:
{context}
"""


def generate_node(state: AgentState) -> AgentState:
    context = _format_context(state["graded_chunks"])
    prompt = _build_prompt(state["question"], context)

    print("Generating answer...")
    response = chat_completion(
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    draft_answer = (response.choices[0].message.content or "").strip()
    print("Answer generated.")

    return {**state, "draft_answer": draft_answer}
