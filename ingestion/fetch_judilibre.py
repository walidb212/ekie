"""Fetch jurisprudence decisions from Judilibre via PISTE API."""

import asyncio
import logging
import os
from typing import Any

import httpx
import tiktoken

from ingestion.auth_piste import PISTEAuth

logger = logging.getLogger(__name__)

JUDILIBRE_BASE = os.environ.get(
    "JUDILIBRE_BASE_URL",
    "https://api.piste.gouv.fr/cassation/judilibre/v1.0",
)
SEARCH_URL = f"{JUDILIBRE_BASE}/search"
DECISION_URL = f"{JUDILIBRE_BASE}/decision"

DOMAINES = {
    "travail": "droit du travail licenciement contrat salaire",
    "famille": "divorce garde enfant pension alimentaire",
    "immobilier": "bail locataire propriétaire loyer caution expulsion",
    "consommation": "consommateur achat garantie remboursement",
    "fiscal": "impôt fiscal taxe redressement",
    "penal": "infraction peine condamnation amende",
}

_encoding = tiktoken.get_encoding("cl100k_base")
MAX_TEXT_TOKENS = 800


def _truncate_text(text: str, max_tokens: int = MAX_TEXT_TOKENS) -> str:
    """Truncate text to max_tokens using tiktoken."""
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _encoding.decode(tokens[:max_tokens])


async def search_decisions(
    auth: PISTEAuth,
    domaine: str,
    max_decisions: int = 500,
) -> list[dict[str, Any]]:
    """Search Judilibre for decisions in a specific legal domain.

    Args:
        auth: PISTEAuth instance.
        domaine: Legal domain key (travail, famille, etc.).
        max_decisions: Maximum decisions to fetch.

    Returns:
        List of decision dicts with keys: id, domaine, juridiction, date, sommaire, texte_court.
    """
    if domaine not in DOMAINES:
        raise ValueError(f"Unknown domaine: {domaine}. Must be one of {list(DOMAINES)}")

    keywords = DOMAINES[domaine]
    decisions: list[dict[str, Any]] = []
    page = 0
    page_size = 50

    async with httpx.AsyncClient(timeout=60.0) as client:
        while len(decisions) < max_decisions:
            headers = await auth.get_auth_headers()

            params = {
                "query": keywords,
                "operator": "or",
                "page_size": page_size,
                "page": page,
                "sort": "score",
                "order": "desc",
                "resolve_references": "true",
            }

            try:
                response = await client.get(SEARCH_URL, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                logger.error("Judilibre search error %d: %s", e.response.status_code, e.response.text[:200])
                break
            except httpx.RequestError as e:
                logger.error("Judilibre request error: %s", e)
                break

            results = data.get("results", [])
            if not results:
                logger.info("No more Judilibre results for '%s' at page %d", domaine, page)
                break

            for result in results:
                decision_id = result.get("id", "")
                sommaire = result.get("summary", "") or ""
                texte_zones = result.get("zones", {})
                texte_parts = []
                for zone_name in ("introduction", "expose", "moyens", "motivations", "dispositif"):
                    zone_text = texte_zones.get(zone_name, "")
                    if zone_text:
                        texte_parts.append(zone_text)
                raw_text = sommaire if not texte_parts else " ".join(texte_parts)

                decisions.append({
                    "id": decision_id,
                    "domaine": domaine,
                    "juridiction": result.get("jurisdiction", ""),
                    "date": result.get("decision_date", ""),
                    "sommaire": _truncate_text(sommaire, 400) if sommaire else "",
                    "texte_court": _truncate_text(raw_text),
                })

                if len(decisions) >= max_decisions:
                    break

            total = data.get("total", 0)
            logger.info(
                "Domaine %s: page %d, fetched %d decisions (total available: %d)",
                domaine, page, len(decisions), total,
            )

            if (page + 1) * page_size >= total or (page + 1) * page_size >= 10000:
                break

            page += 1
            await asyncio.sleep(0.1)

    return decisions[:max_decisions]


async def fetch_all_domaines(auth: PISTEAuth, max_per_domaine: int = 500) -> list[dict[str, Any]]:
    """Fetch decisions from all legal domains.

    Args:
        auth: PISTEAuth instance.
        max_per_domaine: Max decisions per domain.

    Returns:
        Combined list of decisions from all domains.
    """
    all_decisions: list[dict[str, Any]] = []
    for domaine in DOMAINES:
        logger.info("Fetching Judilibre decisions for '%s'...", domaine)
        decisions = await search_decisions(auth, domaine, max_decisions=max_per_domaine)
        all_decisions.extend(decisions)
        logger.info("Fetched %d decisions for '%s'", len(decisions), domaine)
        await asyncio.sleep(0.5)

    logger.info("Total Judilibre decisions fetched: %d", len(all_decisions))
    return all_decisions
