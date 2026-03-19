"""Batch test — run all 20 test questions and save results to JSON."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

from api.classifier import classify_question
from api.retriever import embed_query, retrieve_legal_context
from api.brief_generator import generate_brief

QUESTIONS = [
    # Droit du travail
    ("travail", "Mon employeur m'a licencié par SMS sans aucune lettre officielle, est-ce légal ?"),
    ("travail", "Je suis en arrêt maladie depuis 3 semaines et mon patron me demande de répondre aux emails professionnels, peut-il faire ça ?"),
    ("travail", "Mon entreprise m'a mis en chômage partiel mais continue de me faire travailler les mêmes heures, que puis-je faire ?"),
    ("travail", "J'ai signé une rupture conventionnelle il y a 10 jours mais je veux changer d'avis, c'est encore possible ?"),
    # Immobilier
    ("immobilier", "Mon propriétaire refuse de me rendre ma caution depuis 3 mois en disant qu'il y a des dégâts mais sans preuve, que faire ?"),
    ("immobilier", "Mon propriétaire veut rentrer dans mon appartement sans me prévenir pour faire des travaux, il en a le droit ?"),
    ("immobilier", "J'ai reçu un commandement de quitter les lieux mais j'ai deux enfants en bas âge, peut-on m'expulser ?"),
    ("immobilier", "Mon chauffage est en panne depuis novembre et mon propriétaire ne répond plus à mes messages, quels sont mes recours ?"),
    # Famille
    ("famille", "Je divorce et mon ex refuse de me laisser voir mes enfants le weekend alors que j'ai un droit de visite officiel, que faire ?"),
    ("famille", "Mon ex ne paie plus la pension alimentaire depuis 4 mois, comment la récupérer rapidement ?"),
    ("famille", "Ma mère est décédée sans testament et mon frère veut vendre la maison familiale sans mon accord, est-ce possible ?"),
    # Consommation
    ("consommation", "J'ai acheté un téléphone il y a 8 mois qui ne s'allume plus, le vendeur refuse la garantie en disant que c'est ma faute, que faire ?"),
    ("consommation", "Un site e-commerce a débité ma carte deux fois pour la même commande et ne répond plus à mes mails depuis 2 semaines"),
    ("consommation", "J'ai signé un contrat avec un prestataire internet et ils ne respectent pas les débits promis, puis-je résilier sans frais ?"),
    # Fiscal
    ("fiscal", "J'ai reçu un avis de redressement fiscal pour 2021 mais j'avais bien déclaré tous mes revenus, comment contester ?"),
    ("fiscal", "Je suis auto-entrepreneur et l'URSSAF me réclame des cotisations que j'ai déjà payées, que faire ?"),
    # Pénal
    ("penal", "J'ai été victime d'une arnaque en ligne à 800 euros, vaut-il la peine de porter plainte et comment ?"),
    ("penal", "Mon voisin m'a agressé verbalement et menacé devant témoins, est-ce que ça constitue une infraction pénale ?"),
    # Ambigus
    ("ambigu", "J'ai un problème avec mon patron concernant mon logement de fonction suite à mon licenciement"),
    ("ambigu", "Mon ex-associé dans notre SCI familiale refuse de vendre alors que j'ai besoin d'argent suite à mon divorce"),
]


def run_question(idx: int, expected: str, question: str) -> dict:
    """Run a single question through the full pipeline."""
    print(f"\n{'='*60}")
    print(f"  [{idx+1:02d}/20] {question[:70]}...")
    print(f"  Expected domain: {expected}")
    print(f"{'='*60}")

    result = {
        "id": idx + 1,
        "question": question,
        "expected_domaine": expected,
    }

    pipeline_start = time.perf_counter()

    # Step 1: Classification + Embedding (parallel)
    print("  ── Classify+Embed...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            cls_fut = pool.submit(classify_question, question)
            emb_fut = pool.submit(embed_query, question)
            classification = cls_fut.result()
            query_vector = emb_fut.result()
    except Exception as e:
        print(f"FAILED: {e}")
        result["error"] = f"Classification failed: {e}"
        return result
    cls_ms = int((time.perf_counter() - t0) * 1000)

    match = "✅" if classification["domaine"] == expected or expected == "ambigu" else "❌"
    print(f"{match} {classification['domaine']} / {classification['sous_domaine']} ({classification['confiance']*100:.0f}%) [{cls_ms}ms]")

    result["classification"] = {
        "domaine": classification["domaine"],
        "sous_domaine": classification["sous_domaine"],
        "confiance": classification["confiance"],
        "duration_ms": cls_ms,
        "match": classification["domaine"] == expected or expected == "ambigu",
    }

    # Step 2: Retrieval (Qdrant only, vector already computed)
    print("  ── Retrieval...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        docs = retrieve_legal_context(
            question=question,
            domaine=classification["domaine"],
            confiance=classification["confiance"],
            query_vector=query_vector,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        result["error"] = f"Retrieval failed: {e}"
        return result
    ret_ms = int((time.perf_counter() - t0) * 1000)

    sources = [f"{d['source']} ({d['pertinence']:.2%})" for d in docs]
    print(f"✅ {len(docs)} docs [{ret_ms}ms]")
    for s in sources:
        print(f"       • {s}")

    result["retrieval"] = {
        "count": len(docs),
        "sources": [{"source": d["source"], "pertinence": d["pertinence"]} for d in docs],
        "duration_ms": ret_ms,
    }

    # Step 3: Brief
    print("  ── Brief generation...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        brief = generate_brief(
            question=question,
            domaine=classification["domaine"],
            sous_domaine=classification["sous_domaine"],
            context_docs=docs,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        result["error"] = f"Brief generation failed: {e}"
        return result
    brief_ms = int((time.perf_counter() - t0) * 1000)

    total_ms = int((time.perf_counter() - pipeline_start) * 1000)

    urgence_icons = {"faible": "🟢", "normale": "🔵", "haute": "🟠", "critique": "🔴"}
    icon = urgence_icons.get(brief.urgence, "⚪")
    print(f"✅ {icon} {brief.urgence} [{brief_ms}ms]")
    print(f"       Recommandation: \"{brief.recommandation_immediate[:80]}...\"")
    print(f"       Total: {total_ms}ms")

    result["brief"] = brief.model_dump()
    result["brief"]["generation_duration_ms"] = brief_ms
    result["total_duration_ms"] = total_ms

    return result


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("\n" + "=" * 60)
    print("  EKIE LEGAL ROUTER — BATCH TEST (20 questions)")
    print("=" * 60)

    results = []
    total_start = time.perf_counter()

    for idx, (expected, question) in enumerate(QUESTIONS):
        result = run_question(idx, expected, question)
        results.append(result)

    total_time = time.perf_counter() - total_start

    # Summary
    print("\n\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    correct = sum(1 for r in results if r.get("classification", {}).get("match", False))
    errors = sum(1 for r in results if "error" in r)
    avg_ms = sum(r.get("total_duration_ms", 0) for r in results) / len(results)

    print(f"  Classification accuracy : {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    print(f"  Errors                  : {errors}")
    print(f"  Avg pipeline time       : {avg_ms:.0f}ms")
    print(f"  Total batch time        : {total_time:.1f}s")
    print()

    # Breakdown by expected domain
    from collections import Counter
    domain_results = {}
    for r in results:
        exp = r["expected_domaine"]
        got = r.get("classification", {}).get("domaine", "error")
        if exp not in domain_results:
            domain_results[exp] = {"correct": 0, "total": 0, "misclassified": []}
        domain_results[exp]["total"] += 1
        if r.get("classification", {}).get("match", False):
            domain_results[exp]["correct"] += 1
        else:
            domain_results[exp]["misclassified"].append(f"Q{r['id']}: got '{got}'")

    for dom, stats in domain_results.items():
        status = "✅" if stats["correct"] == stats["total"] else "⚠️"
        print(f"  {status} {dom:15s} : {stats['correct']}/{stats['total']}")
        for miss in stats["misclassified"]:
            print(f"       ❌ {miss}")

    # Save to JSON
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "total_questions": len(results),
            "classification_accuracy": f"{correct}/{len(results)}",
            "accuracy_pct": round(correct / len(results) * 100, 1),
            "errors": errors,
            "avg_pipeline_ms": round(avg_ms),
            "total_batch_seconds": round(total_time, 1),
        },
        "results": results,
    }

    output_path = "test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n  📁 Results saved to {output_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
