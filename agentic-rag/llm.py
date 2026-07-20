import os
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
REQUEST_TIMEOUT = 60.0
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2.0
GRADE_CALL_DELAY_SEC = 0.5

_client: OpenAI | None = None
_model: str | None = None


def get_model() -> str:
    global _model
    if _model is None:
        load_dotenv()
        _model = os.getenv("NVIDIA_MODEL", DEFAULT_MODEL)
    return _model


def get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise RuntimeError("Missing NVIDIA_API_KEY in .env")
        _client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=api_key,
            timeout=REQUEST_TIMEOUT,
        )
    return _client


def chat_completion(*, messages: list[dict], max_tokens: int):
    client = get_client()
    model = get_model()
    backoff = INITIAL_BACKOFF_SEC

    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
        except RateLimitError:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"Rate limited — waiting {backoff:.0f}s before retry ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(backoff)
            backoff *= 2
