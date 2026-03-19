"""Text chunking with token-aware splitting for legal documents."""

import logging
import uuid

import tiktoken

logger = logging.getLogger(__name__)

_encoding = tiktoken.get_encoding("cl100k_base")


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    source_id: str = "",
    domaine: str = "",
) -> list[dict]:
    """Split text into token-aware chunks with overlap.

    Args:
        text: The text to chunk.
        chunk_size: Maximum tokens per chunk.
        overlap: Token overlap between consecutive chunks.
        source_id: ID of the source document.
        domaine: Legal domain of the document.

    Returns:
        List of chunk dicts with keys: chunk_id, source_id, domaine, chunk_index, text.
    """
    if not text or not text.strip():
        return []

    tokens = _encoding.encode(text)

    if len(tokens) <= chunk_size:
        return [
            {
                "chunk_id": str(uuid.uuid4()),
                "source_id": source_id,
                "domaine": domaine,
                "chunk_index": 0,
                "text": text.strip(),
            }
        ]

    chunks: list[dict] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_decoded = _encoding.decode(chunk_tokens).strip()

        if chunk_text_decoded:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "source_id": source_id,
                "domaine": domaine,
                "chunk_index": chunk_index,
                "text": chunk_text_decoded,
            })
            chunk_index += 1

        if end >= len(tokens):
            break

        start = end - overlap

    return chunks
