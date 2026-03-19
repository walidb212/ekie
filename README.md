# Ekie Legal Router — AI-Powered Legal Brief Generator

**Live demo** : _Coming soon_
**API docs** : _Coming soon_ `/docs`

## What it does

Transforms a plain-language legal question from an employee into a structured brief ready for a specialized lawyer — in under 2 seconds.

```
Employee Question
       │
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Classifier  │────▶│  Retriever   │────▶│   Brief Gen  │
│ Gemini Flash │     │ Qdrant + RAG │     │ Mistral Small│
│  (~50ms)     │     │   (~100ms)   │     │  (~800ms)    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                 │
                                                 ▼
                                          BriefAvocat JSON
                                          ready for lawyer
```

## Stack

| Component | Technology |
|---|---|
| **Classification** | Gemini 2.5 Flash (zero-shot, ~50ms) |
| **Embeddings** | Gemini Embedding 2 (768 dims) |
| **Vector DB** | Qdrant Cloud (~30k chunks from Legifrance + Judilibre) |
| **Generation** | Mistral Small (~800ms) |
| **API** | FastAPI on GCP Cloud Run (europe-west1) |
| **Frontend** | Vanilla HTML/CSS/JS on Cloudflare Pages |
| **Data sources** | API PISTE (Legifrance + Judilibre) |

## Production considerations

- **Fine-tuning path**: CamemBERT/ModernBERT for on-premise classification
- **Full RGPD compliance**: Vertex AI migration path documented
- **Semantic cache**: Redis-based cache for repeated questions (documented, not implemented)
- **Cost per request**: ~0.001 EUR

## Local setup

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- API keys (see `.env.example`)

### Quick start

```bash
# Clone and configure
cp .env.example .env
# Edit .env with your API keys

# Start services (Qdrant + API)
docker-compose up --build

# API available at http://localhost:8080
# Swagger docs at http://localhost:8080/docs
```

### Run without Docker

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

## Ingestion

Populate the Qdrant vector store with legal data from Legifrance and Judilibre:

```bash
# Requires PISTE_CLIENT_ID, PISTE_CLIENT_SECRET, GEMINI_API_KEY,
# QDRANT_URL, QDRANT_API_KEY in .env
python -m ingestion.embed_and_store
```

This fetches articles from 5 legal codes (Travail, Civil, Consommation, CGI, Penal) and jurisprudence decisions from Judilibre, chunks them, embeds with Gemini, and stores in Qdrant.

## Tests

```bash
pytest tests/ -v
```

## Deployment

Automated via Cloud Build on push to `main`:

```bash
gcloud builds submit --config cloudbuild.yaml
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (Qdrant connectivity) |
| `POST` | `/analyze` | Analyze a legal question → BriefAvocat |
| `GET` | `/docs` | Swagger UI |

### POST /analyze

```json
{
  "question": "Mon employeur refuse de me payer mes heures supplementaires"
}
```

Response:
```json
{
  "success": true,
  "brief": {
    "domaine": "travail",
    "sous_domaine": "remuneration",
    "urgence": "haute",
    "resume_situation": "...",
    "points_cles": ["..."],
    "questions_a_creuser": ["..."],
    "references_legales": [{"source": "...", "extrait": "...", "pertinence": 0.95}],
    "delais_importants": "3 ans (prescription)",
    "recommandation_immediate": "..."
  },
  "processing_time_ms": 1200
}
```

## License

Internal project — All rights reserved.
