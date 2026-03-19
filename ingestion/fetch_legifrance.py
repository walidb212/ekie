"""Fetch articles from Légifrance via PISTE API."""

import asyncio
import logging
import os
from typing import Any

import httpx

from ingestion.auth_piste import PISTEAuth

logger = logging.getLogger(__name__)

LEGIFRANCE_BASE = os.environ.get(
    "LEGIFRANCE_BASE_URL",
    "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app",
)
SEARCH_URL = f"{LEGIFRANCE_BASE}/search"
CONSULT_CODE_URL = f"{LEGIFRANCE_BASE}/consult/code"

# Mapping: code name -> (textId for /consult/code, domaine label)
CODES = {
    "TRAVAIL": ("LEGITEXT000006072050", "travail"),
    "CIVIL": ("LEGITEXT000006070721", "famille"),
    "CONSO": ("LEGITEXT000006069565", "consommation"),
    "CGI": ("LEGITEXT000006069577", "fiscal"),
    "PENAL": ("LEGITEXT000006070719", "penal"),
}


def _build_search_body(
    search_term: str,
    page_number: int = 1,
    page_size: int = 100,
) -> dict[str, Any]:
    """Build the SearchRequestDTO body for Légifrance search."""
    return {
        "fond": "CODE_DATE",
        "recherche": {
            "champs": [
                {
                    "typeChamp": "ALL",
                    "operateur": "ET",
                    "criteres": [
                        {
                            "valeur": search_term,
                            "operateur": "ET",
                            "typeRecherche": "TOUS_LES_MOTS_DANS_UN_CHAMP",
                        }
                    ],
                }
            ],
            "operateur": "ET",
            "typePagination": "DEFAUT",
            "pageNumber": page_number,
            "pageSize": page_size,
            "sort": "PERTINENCE",
        },
    }


async def search_articles(
    auth: PISTEAuth,
    code_name: str,
    max_articles: int = 3000,
) -> list[dict[str, Any]]:
    """Search articles in a specific legal code via Légifrance API.

    Args:
        auth: PISTEAuth instance for authentication.
        code_name: One of TRAVAIL, CIVIL, CONSO, CGI, PENAL.
        max_articles: Maximum number of articles to fetch per code.

    Returns:
        List of article dicts with keys: id, code, domaine, titre, texte, date_vigueur.
    """
    if code_name not in CODES:
        raise ValueError(f"Unknown code: {code_name}. Must be one of {list(CODES)}")

    text_id, domaine = CODES[code_name]
    articles: list[dict[str, Any]] = []
    page_number = 1
    page_size = 100

    async with httpx.AsyncClient(timeout=60.0) as client:
        while len(articles) < max_articles:
            headers = await auth.get_auth_headers()
            headers["Content-Type"] = "application/json"

            body = _build_search_body(
                search_term=code_name.lower(),
                page_number=page_number,
                page_size=page_size,
            )

            try:
                response = await client.post(SEARCH_URL, json=body, headers=headers)
                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                logger.error("Légifrance search error %d: %s", e.response.status_code, e.response.text[:200])
                break
            except httpx.RequestError as e:
                logger.error("Légifrance request error: %s", e)
                break

            results = data.get("results", [])
            if not results:
                logger.info("No more results for %s at page %d", code_name, page_number)
                break

            for result in results:
                titles = result.get("titles", [])
                title = titles[0].get("title", "") if titles else ""

                sections = result.get("sections", [])
                for section in sections:
                    extracts = section.get("extracts", [])
                    for extract in extracts:
                        article_id = extract.get("id", "")
                        article_title = extract.get("title", title)
                        article_num = extract.get("num", "")
                        values = extract.get("values", [])
                        texte = " ".join(values) if values else ""

                        if texte.strip():
                            articles.append({
                                "id": article_id,
                                "code": code_name,
                                "domaine": domaine,
                                "titre": f"{article_title} {article_num}".strip(),
                                "texte": texte,
                                "date_vigueur": extract.get("dateDebut", ""),
                            })

                if len(articles) >= max_articles:
                    break

            total = data.get("totalResultNumber", 0)
            logger.info(
                "Code %s: page %d, fetched %d articles (total available: %d)",
                code_name, page_number, len(articles), total,
            )

            if page_number * page_size >= total:
                break

            page_number += 1
            await asyncio.sleep(0.1)

    return articles[:max_articles]


async def fetch_all_codes(auth: PISTEAuth, max_per_code: int = 3000) -> list[dict[str, Any]]:
    """Fetch articles from all 5 legal codes.

    Args:
        auth: PISTEAuth instance.
        max_per_code: Max articles per code.

    Returns:
        Combined list of articles from all codes.
    """
    all_articles: list[dict[str, Any]] = []
    for code_name in CODES:
        logger.info("Fetching articles from %s...", code_name)
        articles = await search_articles(auth, code_name, max_articles=max_per_code)
        all_articles.extend(articles)
        logger.info("Fetched %d articles from %s", len(articles), code_name)
        await asyncio.sleep(0.5)

    logger.info("Total articles fetched: %d", len(all_articles))
    return all_articles
