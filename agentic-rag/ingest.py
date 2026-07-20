import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from supabase import create_client

DATA_DIR = Path(__file__).parent / "data"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE_TARGET = 700
OVERLAP_RATIO = 0.15


def load_documents(data_dir: Path) -> list[tuple[str, int, str]]:
    """Return (source_file, page_number, text) for every page/section."""
    pages: list[tuple[str, int, str]] = []

    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            reader = PdfReader(path)
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((path.name, page_number, text))
        else:
            text = path.read_text(encoding="utf-8")
            if text.strip():
                pages.append((path.name, 1, text))

    return pages


def chunk_text(text: str, tokenizer, chunk_size: int, overlap_ratio: float) -> list[str]:
    """Split text into token-sized chunks with overlap."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    overlap = int(chunk_size * overlap_ratio)
    step = max(chunk_size - overlap, 1)
    chunks: list[str] = []

    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        if not chunk_tokens:
            break
        decoded = tokenizer.decode(chunk_tokens, skip_special_tokens=True).strip()
        if decoded:
            chunks.append(decoded)
        if start + chunk_size >= len(tokens):
            break

    return chunks


def main() -> None:
    load_dotenv()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        raise SystemExit("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

    if not DATA_DIR.is_dir():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    documents = load_documents(DATA_DIR)
    if not documents:
        raise SystemExit(f"No supported files found in {DATA_DIR}")

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    supabase = create_client(supabase_url, supabase_key)

    chunk_size = min(CHUNK_SIZE_TARGET, model.max_seq_length)
    overlap = int(chunk_size * OVERLAP_RATIO)
    print(
        f"Loading {MODEL_NAME} | chunk_size={chunk_size} tokens | "
        f"overlap={overlap} tokens ({OVERLAP_RATIO:.0%})"
    )

    ingested = 0
    chunk_counters: dict[str, int] = {}

    for source_file, page_number, text in documents:
        chunks = chunk_text(text, tokenizer, chunk_size, OVERLAP_RATIO)
        for chunk in chunks:
            chunk_index = chunk_counters.get(source_file, 0)
            embedding = model.encode(chunk, normalize_embeddings=True).tolist()

            supabase.table("document_chunks").insert(
                {
                    "content": chunk,
                    "embedding": embedding,
                    "source_file": source_file,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            ).execute()

            chunk_counters[source_file] = chunk_index + 1
            ingested += 1
            print(f"Ingested {ingested} chunks", end="\r", flush=True)

    print(f"\nDone. Ingested {ingested} chunks from {len(chunk_counters)} file(s).")


if __name__ == "__main__":
    main()
