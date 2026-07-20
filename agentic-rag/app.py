from openai import RateLimitError

from graph import run_query
from llm import get_model


def _grounding_label(status: str, is_grounded: bool) -> str:
    if status == "not_enough_info":
        return "N/A (no answer generated)"
    return "Yes" if is_grounded else "No"


def _print_summary(result: dict) -> None:
    print("\n--- Internal summary ---")
    print(f"Chunks retrieved: {len(result['retrieved_chunks'])}")
    print(f"Chunks kept after grading: {len(result['graded_chunks'])}")
    print(f"Retries: {result['retry_count']}")
    print(f"Grounding passed: {_grounding_label(result['status'], result['is_grounded'])}")
    print(f"Status: {result['status']}")


def main() -> None:
    print("Agentic RAG CLI")
    print(f"LLM model: {get_model()}")
    print("Ask a question about your documents. Type 'quit' or 'exit' to leave.\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        print("\nThinking...\n")
        try:
            result = run_query(question)
        except RateLimitError:
            print(
                "NVIDIA API rate limit hit (429 Too Many Requests). "
                "Wait a minute and try again, or ask fewer questions in a row."
            )
            print()
            continue

        print("--- Answer ---")
        print(result["answer"])
        _print_summary(result)
        print()


if __name__ == "__main__":
    main()
