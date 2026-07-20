from typing import TypedDict

DEFAULT_MAX_RETRIES = 3


class RetrievedChunk(TypedDict):
    content: str
    source_file: str
    page_number: int
    similarity: float


class AgentState(TypedDict):
    """State passed between nodes in the agentic RAG graph.

    question: The user's query, possibly rewritten across retries.
    retrieved_chunks: Raw vector-search results from Supabase, each with
        content, source metadata, and a cosine similarity score.
    graded_chunks: Subset of retrieved_chunks that the grader node judged
        relevant to the question; used as context for generation.
    draft_answer: The LLM's current answer attempt, refined across retries.
    is_grounded: Whether the verifier node confirmed the draft is supported
        by the graded chunks (True = pass, False = trigger rewrite/retry).
    retry_count: How many retry cycles have run so far.
    max_retries: Upper bound on retries before the graph exits; initialize to 3.
    final_answer: The answer returned to the caller after a terminal node runs.
    status: Outcome label — success, not_enough_info, or unverified.
    """

    question: str
    retrieved_chunks: list[RetrievedChunk]
    graded_chunks: list[RetrievedChunk]
    draft_answer: str
    is_grounded: bool
    retry_count: int
    max_retries: int
    final_answer: str
    status: str
