import os

from openai import OpenAI

_NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
_DEFAULT_MODEL = "BAAI/bge-en-icl"
_BATCH_SIZE = 32


def is_configured() -> bool:
    return bool(os.getenv("NEBIUS_API_KEY", "").strip())


def _get_client() -> OpenAI:
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. "
            "Add it to your .env file: NEBIUS_API_KEY=your_key_here"
        )
    return OpenAI(base_url=_NEBIUS_BASE_URL, api_key=api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    import openai

    client = _get_client()
    model = os.getenv("NEBIUS_EMBED_MODEL", _DEFAULT_MODEL)

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        try:
            response = client.embeddings.create(model=model, input=batch)
        except openai.AuthenticationError:
            raise RuntimeError(
                "Nebius API key is invalid or expired (got 401). "
                "Check NEBIUS_API_KEY in your .env file and make sure it is a valid key "
                "from https://studio.nebius.com/."
            )
        except openai.APIConnectionError as exc:
            raise RuntimeError(f"Could not reach Nebius API: {exc}") from exc
        all_embeddings.extend(item.embedding for item in response.data)

    return all_embeddings
