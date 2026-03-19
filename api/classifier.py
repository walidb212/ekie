"""Few-shot legal domain classifier using Gemini Flash with edge-case heuristics."""

import json
import logging
import os
import re
import time
import unicodedata

from google import genai
from google.genai import types

logger = logging.getLogger("ekie.classifier")

VALID_DOMAINES = {
    "travail",
    "famille",
    "immobilier",
    "consommation",
    "fiscal",
    "penal",
    "societe",
    "autre",
}
FAMILY_RELATION_TOKENS = {
    "conjoint",
    "conjointe",
    "divorce",
    "epoux",
    "epouse",
    "ex",
    "mari",
    "pacs",
    "pacse",
    "rupture",
    "separation",
}
PROPERTY_DISPUTE_TOKENS = {
    "achat",
    "appart",
    "appartement",
    "bien",
    "indivision",
    "logement",
    "maison",
}
EMPLOYMENT_FRAUD_TOKENS = {
    "falsifie",
    "falsifier",
    "fausse",
    "fausses",
    "faux",
    "modifie",
    "modifier",
    "truque",
    "truquer",
}
PAYSLIP_TOKENS = {
    "fiche",
    "fiches",
    "heures",
    "paie",
    "salaire",
    "supplementaires",
}

CLASSIFICATION_PROMPT = """Tu es un classificateur juridique francais expert.

Exemples :
- "Mon employeur m'a licencie" -> travail
- "Mon proprietaire refuse ma caution" -> immobilier
- "Je divorce et la garde des enfants" -> famille
- "Garantie refusee par le vendeur" -> consommation
- "Redressement fiscal, impots" -> fiscal
- "J'ai ete agresse" -> penal
- "Mon voisin me menace" -> penal
- "J'ai subi une arnaque" -> penal
- "On m'a vole, vol, escroquerie" -> penal
- "Contravention, amende, plainte" -> penal
- "Mon ex veut me reprendre l'appartement achete pendant notre PACS" -> famille + immobilier
- "Mon patron falsifie mes fiches de paie pour effacer des heures supplementaires" -> travail + penal

Domaines : travail, famille, immobilier, consommation, fiscal, penal, societe, autre

Si la question touche PLUSIEURS domaines, indique le domaine principal ET le secondaire.
Si un seul domaine, mets domaine_secondaire a null.

Reponds UNIQUEMENT en JSON valide :
{{"domaine": "travail", "domaine_secondaire": "penal", "sous_domaine": "harcelement", "confiance": 0.90}}

Question : {question}

JSON :"""


def _normalize_question(question: str) -> str:
    """Normalize text for deterministic post-processing heuristics."""
    normalized = unicodedata.normalize("NFKD", question).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _post_process_classification(question: str, classification: dict) -> dict:
    """Apply lightweight heuristics for recurring product edge cases."""
    normalized = _normalize_question(question)
    tokens = set(normalized.split())

    if tokens & FAMILY_RELATION_TOKENS and tokens & PROPERTY_DISPUTE_TOKENS:
        classification["domaine"] = "famille"
        classification["domaine_secondaire"] = "immobilier"
        if "pacs" in tokens or "pacse" in tokens:
            classification["sous_domaine"] = "indivision post-pacs"
        elif classification["sous_domaine"] in {"general", "général"}:
            classification["sous_domaine"] = "indivision"
        classification["confiance"] = max(float(classification["confiance"]), 0.9)

    if tokens & EMPLOYMENT_FRAUD_TOKENS and tokens & PAYSLIP_TOKENS:
        classification["domaine"] = "travail"
        classification["domaine_secondaire"] = "penal"
        classification["sous_domaine"] = "faux et usage de faux"
        classification["confiance"] = max(float(classification["confiance"]), 0.9)

    return classification


def classify_question(
    question: str,
    gemini_client: genai.Client | None = None,
) -> dict:
    """Classify a legal question into one or two domains using Gemini Flash."""
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
            "sous_domaine": result.get("sous_domaine", "general"),
            "confiance": float(result.get("confiance", 0.5)),
        }
        classification = _post_process_classification(question, classification)

        if classification["domaine_secondaire"]:
            logger.info(
                "Gemini classified in %dms -> %s + %s / %s (%.0f%% confidence)",
                api_ms,
                classification["domaine"],
                classification["domaine_secondaire"],
                classification["sous_domaine"],
                classification["confiance"] * 100,
            )
        else:
            logger.info(
                "Gemini classified in %dms -> %s / %s (%.0f%% confidence)",
                api_ms,
                classification["domaine"],
                classification["sous_domaine"],
                classification["confiance"] * 100,
            )
        return classification

    except (json.JSONDecodeError, KeyError, ValueError) as error:
        logger.warning("Classification parse error: %s -> falling back to 'autre'", error)
        return {
            "domaine": "autre",
            "domaine_secondaire": None,
            "sous_domaine": "general",
            "confiance": 0.0,
        }
    except Exception:
        logger.exception("Gemini API call failed")
        return {
            "domaine": "autre",
            "domaine_secondaire": None,
            "sous_domaine": "general",
            "confiance": 0.0,
        }
