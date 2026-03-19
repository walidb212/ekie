"""Generate structured legal briefs using Mistral Small."""

import json
import logging
import os
import time

from mistralai.client import Mistral

from api.models import ArticleRef, BriefAvocat

logger = logging.getLogger("ekie.brief")

SYSTEM_PROMPT = """Tu es un assistant juridique expert français. Tu génères des briefs structurés pour des avocats.

Règles :
- Sois précis et actionnable
- Cite les articles récupérés dans references_legales
- urgence = "haute" si délai légal imminent détecté
- urgence = "critique" si prescription ou forclusion imminente
- questions_a_creuser = ce que l'avocat doit vérifier en priorité
- Réponds UNIQUEMENT en JSON valide"""

USER_PROMPT_TEMPLATE = """QUESTION DU SALARIÉ : {question}
DOMAINE : {domaine} / {sous_domaine}

RÉFÉRENCES LÉGALES PERTINENTES :
{context_formatted}

Réponds UNIQUEMENT en JSON valide correspondant exactement à ce schéma :
{{
  "domaine": "string",
  "sous_domaine": "string",
  "urgence": "faible|normale|haute|critique",
  "resume_situation": "string",
  "points_cles": ["string", "string", "string"],
  "questions_a_creuser": ["string", "string", "string"],
  "references_legales": [
    {{"source": "string", "extrait": "string (200 chars max)", "pertinence": 0.95}}
  ],
  "delais_importants": "string ou null",
  "recommandation_immediate": "string"
}}"""


def _format_context(context_docs: list[dict]) -> str:
    """Format retrieved documents for the prompt."""
    parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("source", "Source inconnue")
        extrait = doc.get("extrait", "")[:400]
        parts.append(f"[{i}] {source}\n{extrait}")
    return "\n\n".join(parts) if parts else "Aucune référence trouvée."


def generate_brief(
    question: str,
    domaine: str,
    sous_domaine: str,
    context_docs: list[dict],
    mistral_client: Mistral | None = None,
) -> BriefAvocat:
    """Generate a structured legal brief using Mistral Small.

    Args:
        question: The user's legal question.
        domaine: Classified legal domain.
        sous_domaine: Classified sub-domain.
        context_docs: Retrieved legal context documents.
        mistral_client: Optional pre-configured Mistral client.

    Returns:
        A validated BriefAvocat Pydantic model.

    Raises:
        ValueError: If brief generation or validation fails after retries.
    """
    if mistral_client is None:
        mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    user_prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        domaine=domaine,
        sous_domaine=sous_domaine,
        context_formatted=_format_context(context_docs),
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    prompt_tokens = len(SYSTEM_PROMPT) + len(user_prompt)
    logger.info("Sending to Mistral Small (prompt ~%d chars, %d context docs)...", prompt_tokens, len(context_docs))

    last_error: Exception | None = None

    for attempt in range(2):
        try:
            t0 = time.perf_counter()
            response = mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=messages,
            )
            api_ms = int((time.perf_counter() - t0) * 1000)

            raw_text = response.choices[0].message.content.strip()
            logger.info("Mistral responded in %dms (%d chars)", api_ms, len(raw_text))
            logger.debug("Mistral raw: %s", raw_text[:300])

            # Strip markdown code fences if present
            if raw_text.startswith("```"):
                raw_text = raw_text.split("\n", 1)[-1]
                raw_text = raw_text.rsplit("```", 1)[0].strip()
                logger.debug("Stripped markdown fences")

            data = json.loads(raw_text)

            # Ensure references_legales are ArticleRef-compatible
            if "references_legales" in data:
                refs = []
                for ref in data["references_legales"]:
                    refs.append({
                        "source": ref.get("source", ""),
                        "extrait": str(ref.get("extrait", ""))[:500],
                        "pertinence": float(ref.get("pertinence", 0.0)),
                    })
                data["references_legales"] = refs

            brief = BriefAvocat.model_validate(data)
            logger.info(
                "Brief validated OK (attempt %d) — urgence=%s, %d points, %d refs",
                attempt + 1,
                brief.urgence,
                len(brief.points_cles),
                len(brief.references_legales),
            )
            return brief

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning("Attempt %d/%d — JSON parse error: %s", attempt + 1, 2, e)
            logger.warning("Raw text was: %s", raw_text[:200] if 'raw_text' in dir() else "N/A")
        except Exception as e:
            last_error = e
            logger.warning("Attempt %d/%d — Error: %s", attempt + 1, 2, e)

    raise ValueError(f"Failed to generate valid brief after 2 attempts: {last_error}")
