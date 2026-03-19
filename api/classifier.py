"""Few-shot legal domain classifier using Gemini Flash with multi-domain detection."""

import json
import logging
import os
import time

from google import genai
from google.genai import types

logger = logging.getLogger("ekie.classifier")

VALID_DOMAINES = {"travail", "famille", "immobilier", "consommation", "fiscal", "penal", "societe", "autre"}

CLASSIFICATION_PROMPT = """Tu es un classificateur juridique français expert.

Exemples :
- "Mon employeur m'a licencié" → travail
- "Mon propriétaire refuse ma caution" → immobilier
- "Je divorce et la garde des enfants" → famille
- "Garantie refusée par le vendeur" → consommation
- "Redressement fiscal, impôts" → fiscal
- "J'ai été agressé" → penal
- "Mon voisin me menace" → penal
- "J'ai subi une arnaque" → penal
- "On m'a volé, vol, escroquerie" → penal
- "Contravention, amende, plainte" → penal

Domaines : travail, famille, immobilier, consommation, fiscal, penal, societe, autre

Si la question touche PLUSIEURS domaines, indique le domaine principal ET le secondaire.
Si un seul domaine, mets domaine_secondaire à null.

Réponds UNIQUEMENT en JSON valide :
{{"domaine": "travail", "domaine_secondaire": "penal", "sous_domaine": "harcèlement", "confiance": 0.90}}

Question : {question}

JSON :"""


def classify_question(
    question: str,
    gemini_client: genai.Client | None = None,
) -> dict:
    """Classify a legal question into one or two domains using Gemini Flash.

    Args:
        question: The user's legal question in French.
        gemini_client: Optional pre-configured Gemini client.

    Returns:
        Dict with keys: domaine, domaine_secondaire, sous_domaine, confiance.
    """
    if gemini_client is None:
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = CLASSIFICATION_PROMPT.format(question=question)
    logger.debug("Prompt sent to Gemini (%d chars)", len(prompt))

    try:
        t0 = time.perf_counter()
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=300,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        api_ms = int((time.perf_counter() - t0) * 1000)

        raw_text = response.text.strip()
        logger.debug("Gemini raw response (%dms): %s", api_ms, raw_text[:200])

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]
            raw_text = raw_text.rsplit("```", 1)[0].strip()

        result = json.loads(raw_text)

        domaine = result.get("domaine", "autre").lower()
        if domaine not in VALID_DOMAINES:
            domaine = "autre"

        domaine_secondaire = result.get("domaine_secondaire")
        if domaine_secondaire:
            domaine_secondaire = domaine_secondaire.lower()
            if domaine_secondaire not in VALID_DOMAINES or domaine_secondaire == domaine:
                domaine_secondaire = None

        classification = {
            "domaine": domaine,
            "domaine_secondaire": domaine_secondaire,
            "sous_domaine": result.get("sous_domaine", "général"),
            "confiance": float(result.get("confiance", 0.5)),
        }

        if domaine_secondaire:
            logger.info(
                "Gemini classified in %dms → %s + %s / %s (%.0f%% confidence)",
                api_ms, domaine, domaine_secondaire,
                classification["sous_domaine"],
                classification["confiance"] * 100,
            )
        else:
            logger.info(
                "Gemini classified in %dms → %s / %s (%.0f%% confidence)",
                api_ms, domaine,
                classification["sous_domaine"],
                classification["confiance"] * 100,
            )
        return classification

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning("Classification parse error: %s — falling back to 'autre'", e)
        return {
            "domaine": "autre",
            "domaine_secondaire": None,
            "sous_domaine": "général",
            "confiance": 0.0,
        }
    except Exception:
        logger.exception("Gemini API call failed")
        return {
            "domaine": "autre",
            "domaine_secondaire": None,
            "sous_domaine": "général",
            "confiance": 0.0,
        }
