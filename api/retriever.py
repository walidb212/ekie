"""RAG retrieval from Qdrant vector store using Gemini embeddings."""

import logging
import os
import time

import numpy as np
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger("ekie.retriever")

COLLECTION_NAME = "legal_fr"
EMBED_MODEL = "gemini-embedding-2-preview"
VECTOR_SIZE = 768
MIN_PERTINENCE = 0.72
SEARCH_LIMIT_MULTIPLIER = 3


def _normalize(v: list[float]) -> list[float]:
    """L2-normalize a vector."""
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def embed_query(question: str, gemini_client: genai.Client | None = None) -> list[float]:
    """Embed a query string and return the normalized vector. Can be called in parallel with classification."""
    if gemini_client is None:
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    logger.info("Embedding query with %s (dims=%d)...", EMBED_MODEL, VECTOR_SIZE)
    t0 = time.perf_counter()
    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=VECTOR_SIZE,
        ),
    )
    query_vector = _normalize(result.embeddings[0].values)
    embed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("Query embedded in %dms", embed_ms)
    return query_vector


def retrieve_legal_context(
    question: str,
    domaine: str,
    confiance: float = 0.0,
    n: int = 3,
    qdrant_client: QdrantClient | None = None,
    gemini_client: genai.Client | None = None,
    query_vector: list[float] | None = None,
    domaine_secondaire: str | None = None,
) -> list[dict]:
    """Retrieve the top-N most relevant legal documents for a question.

    Args:
        question: The user's legal question.
        domaine: Classified legal domain.
        confiance: Classification confidence score.
        n: Number of results to return.
        qdrant_client: Optional pre-configured Qdrant client.
        gemini_client: Optional pre-configured Gemini client.

    Returns:
        List of dicts with keys: source, extrait, pertinence, texte.
    """
    if qdrant_client is None:
        qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )

    # Use pre-computed vector or embed now
    if query_vector is None:
        query_vector = embed_query(question, gemini_client)
    else:
        logger.info("Using pre-computed query vector (parallel mode)")

    # Build filter: only filter by domaine if classification confidence is high
    query_filter = None
    if confiance > 0.7 and domaine != "autre":
        if domaine_secondaire:
            # Multi-domain: search in both domains
            query_filter = Filter(
                should=[
                    FieldCondition(key="domaine", match=MatchValue(value=domaine)),
                    FieldCondition(key="domaine", match=MatchValue(value=domaine_secondaire)),
                ]
            )
            logger.info("Qdrant filter: domaine='%s' OR '%s' (confiance=%.0f%%)", domaine, domaine_secondaire, confiance * 100)
        else:
            query_filter = Filter(
                must=[FieldCondition(key="domaine", match=MatchValue(value=domaine))]
            )
            logger.info("Qdrant filter: domaine='%s' (confiance=%.0f%% > 70%%)", domaine, confiance * 100)
    else:
        logger.info("Qdrant filter: NONE (confiance=%.0f%% <= 70%% or domaine='autre')", confiance * 100)

    # Query Qdrant
    search_limit = max(n, n * SEARCH_LIMIT_MULTIPLIER)
    logger.info(
        "Querying Qdrant collection '%s' for top %d candidates...",
        COLLECTION_NAME,
        search_limit,
    )
    t0 = time.perf_counter()
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=search_limit,
        with_payload=True,
    )
    qdrant_ms = int((time.perf_counter() - t0) * 1000)

    documents = []
    for point in search_result.points:
        payload = point.payload or {}
        texte = payload.get("text", "")
        documents.append({
            "source": payload.get("source", "Unknown"),
            "extrait": texte[:500] if texte else "",
            "pertinence": round(point.score, 4),
            "texte": texte,
        })

    raw_count = len(documents)
    documents = [doc for doc in documents if doc["pertinence"] >= MIN_PERTINENCE]
    dropped_count = raw_count - len(documents)
    if dropped_count:
        logger.info(
            "Dropped %d low-pertinence docs (< %.2f)",
            dropped_count,
            MIN_PERTINENCE,
        )

    documents = documents[:n]

    logger.info(
        "Qdrant returned %d relevant results in %dms (%d raw)",
        len(documents),
        qdrant_ms,
        raw_count,
    )
    for i, doc in enumerate(documents, 1):
        logger.info(
            "  #%d  score=%.4f  src=%s  (%d chars)",
            i, doc["pertinence"], doc["source"], len(doc["texte"]),
        )

    return documents
