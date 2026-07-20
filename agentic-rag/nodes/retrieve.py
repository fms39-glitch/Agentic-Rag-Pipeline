import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import Client, create_client

from state import AgentState, RetrievedChunk

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MATCH_COUNT = 6

_model: SentenceTransformer | None = None
_supabase: Client | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")
        _supabase = create_client(url, key)
    return _supabase


def retrieve_node(state: AgentState) -> AgentState:
    print("Retrieving chunks...")
    model = _get_model()
    supabase = _get_supabase()

    query_text = QUERY_PREFIX + state["question"]
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    response = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": MATCH_COUNT,
        },
    ).execute()

    retrieved_chunks: list[RetrievedChunk] = [
        {
            "content": row["content"],
            "source_file": row["source_file"],
            "page_number": row["page_number"],
            "similarity": float(row["similarity"]),
        }
        for row in response.data or []
    ]

    print(f"Retrieved {len(retrieved_chunks)} chunks")

    return {**state, "retrieved_chunks": retrieved_chunks}
