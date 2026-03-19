"""Seed Qdrant with sample French legal data for demo/testing.

Use this when PISTE API access is not yet configured.
Run: python -m ingestion.seed_sample_data
"""

import logging
import os
import uuid

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams
from tqdm import tqdm

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legal_fr"
VECTOR_SIZE = 768
EMBED_MODEL = "gemini-embedding-2-preview"

SAMPLE_DOCUMENTS = [
    # --- TRAVAIL ---
    {
        "source": "Code du travail - Art. L1231-1",
        "domaine": "travail",
        "titre": "Rupture du contrat de travail",
        "text": "Le contrat de travail à durée indéterminée peut être rompu à l'initiative de l'employeur ou du salarié, ou d'un commun accord, dans les conditions prévues par les dispositions du présent titre. Ces dispositions ne sont pas applicables pendant la période d'essai.",
    },
    {
        "source": "Code du travail - Art. L1232-1",
        "domaine": "travail",
        "titre": "Licenciement pour motif personnel",
        "text": "Tout licenciement pour motif personnel est motivé dans les conditions définies par le présent chapitre. Il est justifié par une cause réelle et sérieuse.",
    },
    {
        "source": "Code du travail - Art. L1234-1",
        "domaine": "travail",
        "titre": "Préavis de licenciement",
        "text": "Lorsque le licenciement n'est pas motivé par une faute grave, le salarié a droit à un préavis dont la durée est déterminée par la loi, la convention ou l'accord collectif de travail ou, à défaut, par les usages pratiqués dans la localité et la profession. Le préavis est de un mois pour une ancienneté de six mois à deux ans, et de deux mois pour une ancienneté de deux ans ou plus.",
    },
    {
        "source": "Code du travail - Art. L1234-9",
        "domaine": "travail",
        "titre": "Indemnité de licenciement",
        "text": "Le salarié titulaire d'un contrat de travail à durée indéterminée, licencié alors qu'il compte 8 mois d'ancienneté ininterrompus au service du même employeur, a droit, sauf en cas de faute grave, à une indemnité de licenciement. Les modalités de calcul de cette indemnité sont fonction de la rémunération brute dont le salarié bénéficiait antérieurement à la rupture du contrat de travail.",
    },
    {
        "source": "Code du travail - Art. L3121-27",
        "domaine": "travail",
        "titre": "Durée légale du travail",
        "text": "La durée légale de travail effectif des salariés à temps complet est fixée à trente-cinq heures par semaine. Toute heure accomplie au-delà de la durée légale hebdomadaire est une heure supplémentaire qui ouvre droit à une majoration salariale ou, le cas échéant, à un repos compensateur équivalent.",
    },
    {
        "source": "Code du travail - Art. L1237-11",
        "domaine": "travail",
        "titre": "Rupture conventionnelle",
        "text": "L'employeur et le salarié peuvent convenir en commun des conditions de la rupture du contrat de travail qui les lie. La rupture conventionnelle, exclusive du licenciement ou de la démission, ne peut être imposée par l'une ou l'autre des parties. Elle résulte d'une convention signée par les parties au contrat.",
    },
    # --- IMMOBILIER ---
    {
        "source": "Loi n°89-462 - Art. 22",
        "domaine": "immobilier",
        "titre": "Dépôt de garantie (caution)",
        "text": "Lorsqu'un dépôt de garantie est prévu par le contrat de location pour garantir l'exécution de ses obligations locatives par le locataire, il ne peut être supérieur à un mois de loyer en principal pour les locations nues. Il doit être restitué dans un délai maximal de un mois à compter de la remise des clés par le locataire lorsque l'état des lieux de sortie est conforme à l'état des lieux d'entrée, déduction faite des sommes restant dues au bailleur.",
    },
    {
        "source": "Loi n°89-462 - Art. 6",
        "domaine": "immobilier",
        "titre": "Obligations du bailleur",
        "text": "Le bailleur est tenu de remettre au locataire un logement décent ne laissant pas apparaître de risques manifestes pouvant porter atteinte à la sécurité physique ou à la santé, exempt de toute infestation d'espèces nuisibles et parasites, répondant à un critère de performance énergétique minimale et doté des éléments le rendant conforme à l'usage d'habitation.",
    },
    {
        "source": "Loi n°89-462 - Art. 15",
        "domaine": "immobilier",
        "titre": "Congé donné par le bailleur",
        "text": "Lorsque le bailleur donne congé à son locataire, ce congé doit être justifié soit par sa décision de reprendre ou de vendre le logement, soit par un motif légitime et sérieux, notamment l'inexécution par le locataire de l'une des obligations lui incombant. Le délai de préavis est de six mois pour les locations nues.",
    },
    # --- CONSOMMATION ---
    {
        "source": "Code de la consommation - Art. L217-4",
        "domaine": "consommation",
        "titre": "Garantie légale de conformité",
        "text": "Le vendeur livre un bien conforme au contrat ainsi qu'aux critères énoncés à l'article L. 217-5. Il répond des défauts de conformité existant lors de la délivrance du bien. Il répond également des défauts de conformité résultant de l'emballage, des instructions de montage, ou de l'installation lorsque celle-ci a été mise à sa charge par le contrat ou a été réalisée sous sa responsabilité.",
    },
    {
        "source": "Code de la consommation - Art. L221-18",
        "domaine": "consommation",
        "titre": "Droit de rétractation",
        "text": "Le consommateur dispose d'un délai de quatorze jours pour exercer son droit de rétractation d'un contrat conclu à distance, à la suite d'un démarchage téléphonique ou hors établissement, sans avoir à motiver sa décision ni à supporter d'autres coûts que ceux prévus aux articles L. 221-23 à L. 221-25.",
    },
    # --- FISCAL ---
    {
        "source": "CGI - Art. 1 A",
        "domaine": "fiscal",
        "titre": "Impôt sur le revenu",
        "text": "Il est établi un impôt annuel unique sur le revenu des personnes physiques désigné sous le nom d'impôt sur le revenu. Cet impôt frappe le revenu net global du contribuable déterminé conformément aux dispositions des articles 156 à 168 A.",
    },
    {
        "source": "CGI - Art. 150 U",
        "domaine": "fiscal",
        "titre": "Plus-values immobilières",
        "text": "Les plus-values réalisées par les personnes physiques lors de la cession à titre onéreux de biens immobiliers bâtis ou non bâtis ou de droits relatifs à ces biens, sont passibles de l'impôt sur le revenu. La plus-value brute est égale à la différence entre le prix de cession et le prix d'acquisition par le cédant.",
    },
    # --- PENAL ---
    {
        "source": "Code pénal - Art. 311-1",
        "domaine": "penal",
        "titre": "Vol",
        "text": "Le vol est la soustraction frauduleuse de la chose d'autrui. Il est puni de trois ans d'emprisonnement et de 45 000 euros d'amende.",
    },
    {
        "source": "Code pénal - Art. 222-13",
        "domaine": "penal",
        "titre": "Violences ayant entraîné une ITT",
        "text": "Les violences ayant entraîné une incapacité de travail inférieure ou égale à huit jours ou n'ayant entraîné aucune incapacité de travail sont punies de trois ans d'emprisonnement et de 45 000 euros d'amende lorsqu'elles sont commises sur un mineur de quinze ans ou sur une personne vulnérable.",
    },
    # --- FAMILLE ---
    {
        "source": "Code civil - Art. 229",
        "domaine": "famille",
        "titre": "Cas de divorce",
        "text": "Le divorce peut être prononcé en cas : soit de consentement mutuel, soit d'acceptation du principe de la rupture du mariage, soit d'altération définitive du lien conjugal, soit de faute.",
    },
    {
        "source": "Code civil - Art. 373-2",
        "domaine": "famille",
        "titre": "Autorité parentale après séparation",
        "text": "La séparation des parents est sans incidence sur les règles de dévolution de l'exercice de l'autorité parentale. Chacun des père et mère doit maintenir des relations personnelles avec l'enfant et respecter les liens de celui-ci avec l'autre parent.",
    },
    {
        "source": "Code civil - Art. 270",
        "domaine": "famille",
        "titre": "Prestation compensatoire",
        "text": "Le divorce met fin au devoir de secours entre époux. L'un des époux peut être tenu de verser à l'autre une prestation destinée à compenser, autant qu'il est possible, la disparité que la rupture du mariage crée dans les conditions de vie respectives.",
    },
]


def normalize_vector(v: list[float]) -> list[float]:
    """L2-normalize a vector."""
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    load_dotenv()

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    # Recreate collection
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in collections:
        qdrant.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection '%s'", COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Created collection '%s'", COLLECTION_NAME)

    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="domaine",
        field_schema=PayloadSchemaType.KEYWORD,
    )
    logger.info("Created payload index for 'domaine'")

    # Embed and store
    texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    logger.info("Embedding %d sample documents...", len(texts))

    all_vectors = []
    batch_size = 20
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        result = gemini.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=VECTOR_SIZE,
            ),
        )
        for emb in result.embeddings:
            all_vectors.append(normalize_vector(emb.values))

    # Upsert to Qdrant
    points = []
    for doc, vector in zip(SAMPLE_DOCUMENTS, all_vectors):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": doc["text"],
                    "source": doc["source"],
                    "domaine": doc["domaine"],
                    "titre": doc["titre"],
                    "source_id": doc["source"],
                    "chunk_index": 0,
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

    info = qdrant.get_collection(COLLECTION_NAME)
    logger.info("Seeded %d documents into '%s'", info.points_count, COLLECTION_NAME)
    logger.info("Done! You can now test the API with: uvicorn api.main:app --port 8080")


if __name__ == "__main__":
    main()
