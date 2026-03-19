"""FastAPI application for the Ekie Legal Router."""

import logging
import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware
from slowapi.util import get_remote_address

from google import genai
from mistralai.client import Mistral

import asyncio
import concurrent.futures

from api.brief_generator import generate_brief
from api.classifier import classify_question
from api.conversation import process_message
from api.models import APIResponse, ChatRequest, ChatResponse, QuestionInput
from api.retriever import embed_query, retrieve_legal_context

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — human-readable for CLI, rich detail
# ---------------------------------------------------------------------------
LOG_FORMAT = (
    "\033[90m%(asctime)s\033[0m "
    "%(levelname)-8s "
    "\033[36m%(name)-28s\033[0m "
    "%(message)s"
)
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
# Silence noisy HTTP-level logs from httpx unless WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("ekie.api")

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["100/hour"])


def _banner(text: str) -> str:
    """Return a visible CLI banner line."""
    return f"\n{'='*60}\n  {text}\n{'='*60}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup shared clients."""
    logger.info(_banner("EKIE LEGAL ROUTER — Starting up"))

    # Qdrant
    qdrant_url = os.environ["QDRANT_URL"]
    logger.info("Connecting to Qdrant at %s ...", qdrant_url)
    app.state.qdrant = QdrantClient(
        url=qdrant_url,
        api_key=os.environ["QDRANT_API_KEY"],
    )
    try:
        collections = app.state.qdrant.get_collections()
        names = [c.name for c in collections.collections]
        logger.info("Qdrant OK — collections: %s", names)
        if "legal_fr" in names:
            info = app.state.qdrant.get_collection("legal_fr")
            logger.info("Collection 'legal_fr': %d vectors, %d dims", info.points_count, info.config.params.vectors.size)
        else:
            logger.warning("Collection 'legal_fr' NOT FOUND — run ingestion first!")
    except Exception:
        logger.exception("Qdrant connection FAILED")

    # Gemini
    logger.info("Initializing Gemini client (model: gemini-2.5-flash + embedding-2)...")
    app.state.gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    logger.info("Gemini client ready")

    # Mistral
    logger.info("Initializing Mistral client (model: mistral-small-latest)...")
    app.state.mistral = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    logger.info("Mistral client ready")

    logger.info(_banner("READY — Listening on http://0.0.0.0:%s") , os.environ.get("PORT", "8080"))

    yield

    logger.info(_banner("Shutting down"))
    app.state.qdrant.close()


app = FastAPI(
    title="Ekie Legal Router",
    description="AI-powered legal brief generator for French law",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIASGIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """Add X-Process-Time header to all responses."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{elapsed:.0f}ms"
    logger.info("%s %s — %dms", request.method, request.url.path, elapsed)
    return response


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    qdrant_status = "disconnected"
    try:
        request.app.state.qdrant.get_collections()
        qdrant_status = "connected"
    except Exception:
        pass
    return {"status": "ok", "qdrant": qdrant_status}


@app.post("/analyze", response_model=APIResponse)
@limiter.limit("10/minute")
async def analyze_question(question_input: QuestionInput, request: Request):
    """Analyze a legal question and generate a structured brief.

    Pipeline: classify → retrieve → generate brief.
    """
    pipeline_start = time.perf_counter()
    question = question_input.question

    logger.info(_banner("NEW REQUEST"))
    logger.info("Question (%d chars): \"%s\"", len(question), question[:120])

    try:
        # ── Step 1+2: Classification + Embedding in parallel ──────
        logger.info("─── STEP 1/3 : Classification + Embedding (parallel) ───")
        t0 = time.perf_counter()

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            classify_future = loop.run_in_executor(
                pool, classify_question, question, request.app.state.gemini,
            )
            embed_future = loop.run_in_executor(
                pool, embed_query, question, request.app.state.gemini,
            )
            classification, query_vector = await asyncio.gather(classify_future, embed_future)

        t1 = time.perf_counter()
        logger.info("  Domaine     : %s", classification["domaine"])
        if classification.get("domaine_secondaire"):
            logger.info("  Domaine 2   : %s", classification["domaine_secondaire"])
        logger.info("  Sous-domaine: %s", classification["sous_domaine"])
        logger.info("  Confiance   : %.0f%%", classification["confiance"] * 100)
        logger.info("  Durée       : %dms (parallel)", (t1 - t0) * 1000)

        # ── Step 2: Qdrant search ─────────────────────────────────
        logger.info("─── STEP 2/3 : Retrieval (Qdrant search) ───")
        t0 = time.perf_counter()

        context_docs = retrieve_legal_context(
            question=question,
            domaine=classification["domaine"],
            confiance=classification["confiance"],
            n=3,
            qdrant_client=request.app.state.qdrant,
            query_vector=query_vector,
            domaine_secondaire=classification.get("domaine_secondaire"),
        )

        t1 = time.perf_counter()
        logger.info("  Documents trouvés: %d", len(context_docs))
        for i, doc in enumerate(context_docs, 1):
            logger.info("  [%d] score=%.4f | %s", i, doc["pertinence"], doc["source"])
            logger.info("      \"%s...\"", doc["extrait"][:80])
        logger.info("  Durée: %dms", (t1 - t0) * 1000)

        # ── Step 3: Brief Generation ──────────────────────────────
        logger.info("─── STEP 3/3 : Brief Generation (Mistral Small) ───")
        t0 = time.perf_counter()

        brief = generate_brief(
            question=question,
            domaine=classification["domaine"],
            sous_domaine=classification["sous_domaine"],
            context_docs=context_docs,
            mistral_client=request.app.state.mistral,
        )

        t1 = time.perf_counter()
        logger.info("  Urgence         : %s", brief.urgence)
        logger.info("  Points clés     : %d", len(brief.points_cles))
        logger.info("  Références      : %d", len(brief.references_legales))
        logger.info("  Délais          : %s", brief.delais_importants or "aucun")
        logger.info("  Recommandation  : \"%s...\"", brief.recommandation_immediate[:80])
        logger.info("  Durée           : %dms", (t1 - t0) * 1000)

        # ── Summary ──────────────────────────────────────────────
        total_ms = int((time.perf_counter() - pipeline_start) * 1000)
        logger.info(_banner(f"DONE — {total_ms}ms total"))
        logger.info("  Classification : %dms", (t1 - pipeline_start) * 1000 - (t1 - t0) * 1000)
        logger.info("  Pipeline total : %dms", total_ms)

        return APIResponse(
            success=True,
            brief=brief,
            processing_time_ms=total_ms,
        )

    except Exception as e:
        total_ms = int((time.perf_counter() - pipeline_start) * 1000)
        logger.exception("Pipeline FAILED after %dms: %s", total_ms, e)
        return JSONResponse(
            status_code=500,
            content=APIResponse(
                success=False,
                processing_time_ms=total_ms,
                error=str(e),
            ).model_dump(),
        )


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(chat_request: ChatRequest, request: Request):
    """Conversational endpoint — guides the user then generates a brief."""
    start = time.perf_counter()

    logger.info(_banner("CHAT MESSAGE"))
    logger.info("Conversation: %s | Message: \"%s\"", chat_request.conversation_id or "NEW", chat_request.message[:100])

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            process_message,
            chat_request.conversation_id,
            chat_request.message,
            request.app.state.gemini,
            request.app.state.qdrant,
            request.app.state.mistral,
        )

        total_ms = int((time.perf_counter() - start) * 1000)

        if result["action"] == "brief":
            logger.info("→ Brief generated in %dms", total_ms)
            return ChatResponse(
                conversation_id=result["conversation_id"],
                action="brief",
                brief=result["brief"],
                domaine=result["domaine"],
                confiance=result["confiance"],
                processing_time_ms=total_ms,
            )
        else:
            logger.info("→ Question asked in %dms: \"%s\"", total_ms, result["message"][:80])
            return ChatResponse(
                conversation_id=result["conversation_id"],
                action="question",
                message=result["message"],
                domaine=result["domaine"],
                confiance=result["confiance"],
                processing_time_ms=total_ms,
            )

    except Exception as e:
        total_ms = int((time.perf_counter() - start) * 1000)
        logger.exception("Chat FAILED after %dms: %s", total_ms, e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "processing_time_ms": total_ms},
        )
