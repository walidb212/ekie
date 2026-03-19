"""Embed legal documents with Gemini and store vectors in Qdrant."""

import asyncio
import logging
import os
import time
import uuid

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from tqdm import tqdm

from ingestion.auth_piste import PISTEAuth
from ingestion.chunker import chunk_text
from ingestion.fetch_judilibre import fetch_all_domaines
from ingestion.fetch_legifrance import fetch_all_codes

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legal_fr"
VECTOR_SIZE = 768
EMBED_MODEL = "gemini-embedding-2-preview"
BATCH_SIZE = 100


def normalize_vector(v: list[float]) -> list[float]:
    """L2-normalize a vector. Required for gemini-embedding-2-preview at reduced dims."""
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the legal_fr collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if recreate and COLLECTION_NAME in collections:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)
        collections.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="domaine",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Created collection '%s' with domaine index", COLLECTION_NAME)
    else:
        logger.info("Collection '%s' already exists", COLLECTION_NAME)


def embed_batch(gemini_client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using Gemini and return normalized vectors."""
    result = gemini_client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=VECTOR_SIZE,
        ),
    )
    return [normalize_vector(emb.values) for emb in result.embeddings]


def upsert_batch(
    qdrant: QdrantClient,
    chunks: list[dict],
    vectors: list[list[float]],
) -> None:
    """Upsert a batch of points into Qdrant."""
    points = []
    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk["text"],
                    "source_id": chunk["source_id"],
                    "domaine": chunk["domaine"],
                    "chunk_index": chunk["chunk_index"],
                    "source": chunk.get("source", ""),
                    "titre": chunk.get("titre", ""),
                },
            )
        )
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


async def run_ingestion() -> None:
    """Main ingestion pipeline: fetch, chunk, embed, store."""
    load_dotenv()

    auth = PISTEAuth()
    gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    qdrant = get_qdrant_client()
    ensure_collection(qdrant, recreate=True)

    # Fetch all legal data
    logger.info("=== Fetching Légifrance articles ===")
    articles = await fetch_all_codes(auth)

    logger.info("=== Fetching Judilibre decisions ===")
    decisions = await fetch_all_domaines(auth)

    await auth.close()

    # Chunk all documents
    logger.info("=== Chunking documents ===")
    all_chunks: list[dict] = []

    for article in tqdm(articles, desc="Chunking articles"):
        chunks = chunk_text(
            text=article["texte"],
            source_id=article["id"],
            domaine=article["domaine"],
        )
        for chunk in chunks:
            chunk["source"] = f"{article['code']} - {article['titre']}"
            chunk["titre"] = article["titre"]
        all_chunks.extend(chunks)

    for decision in tqdm(decisions, desc="Chunking decisions"):
        text = decision.get("texte_court", "") or decision.get("sommaire", "")
        chunks = chunk_text(
            text=text,
            source_id=decision["id"],
            domaine=decision["domaine"],
        )
        for chunk in chunks:
            chunk["source"] = f"Judilibre - {decision['juridiction']} - {decision['date']}"
            chunk["titre"] = decision.get("sommaire", "")[:100]
        all_chunks.extend(chunks)

    logger.info("Total chunks to embed: %d", len(all_chunks))

    # Embed and store in batches
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info("=== Embedding with Gemini (%d batches) ===", total_batches)
    stored = 0
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), desc="Embedding batches"):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        for retry in range(3):
            try:
                vectors = embed_batch(gemini_client, texts)
                upsert_batch(qdrant, batch, vectors)
                stored += len(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 15 * (retry + 1)
                    logger.warning("Rate limited, waiting %ds (retry %d/3)...", wait, retry + 1)
                    time.sleep(wait)
                else:
                    logger.exception("Error embedding batch %d", i // BATCH_SIZE)
                    break

    collection_info = qdrant.get_collection(COLLECTION_NAME)
    logger.info(
        "Ingestion complete. Collection '%s' has %d points (%d stored).",
        COLLECTION_NAME,
        collection_info.points_count,
        stored,
    )


def main() -> None:
    """Entry point for python -m ingestion.embed_and_store."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_ingestion())


if __name__ == "__main__":
    main()
