"""CLI pour tester le pipeline Ekie directement sans serveur."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

from api.classifier import classify_question
from api.retriever import embed_query, retrieve_legal_context
from api.brief_generator import generate_brief


def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("\n💬 Posez votre question juridique : ")

    if len(question.strip()) < 10:
        print("❌ Question trop courte (10 caractères minimum)")
        return

    print(f"\n{'='*60}")
    print(f"  Question : {question}")
    print(f"{'='*60}")

    pipeline_start = time.perf_counter()

    # Step 1 — Classification + Embedding in parallel
    print("\n─── STEP 1/3 : Classification + Embedding (parallel) ───")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        classify_fut = pool.submit(classify_question, question)
        embed_fut = pool.submit(embed_query, question)
        classification = classify_fut.result()
        query_vector = embed_fut.result()
    t1 = time.perf_counter()
    print(f"  ✅ Domaine      : {classification['domaine']}")
    print(f"     Sous-domaine : {classification['sous_domaine']}")
    print(f"     Confiance    : {classification['confiance']*100:.0f}%")
    print(f"     Durée        : {(t1-t0)*1000:.0f}ms (parallel)")

    # Step 2
    print("\n─── STEP 2/3 : Retrieval (Qdrant search) ───")
    t0 = time.perf_counter()
    docs = retrieve_legal_context(
        question=question,
        domaine=classification["domaine"],
        confiance=classification["confiance"],
        query_vector=query_vector,
    )
    t1 = time.perf_counter()
    print(f"  ✅ {len(docs)} documents trouvés ({(t1-t0)*1000:.0f}ms)")
    for i, doc in enumerate(docs, 1):
        print(f"     [{i}] {doc['pertinence']:.2%} — {doc['source']}")
        print(f"         \"{doc['extrait'][:80]}...\"")

    # Step 3
    print("\n─── STEP 3/3 : Brief Generation (Mistral Small) ───")
    t0 = time.perf_counter()
    brief = generate_brief(
        question=question,
        domaine=classification["domaine"],
        sous_domaine=classification["sous_domaine"],
        context_docs=docs,
    )
    t1 = time.perf_counter()
    print(f"  ✅ Brief généré ({(t1-t0)*1000:.0f}ms)")

    total_ms = (time.perf_counter() - pipeline_start) * 1000

    # Display brief
    print(f"\n{'='*60}")
    print(f"  BRIEF AVOCAT — {total_ms:.0f}ms")
    print(f"{'='*60}")

    urgence_icons = {"faible": "🟢", "normale": "🔵", "haute": "🟠", "critique": "🔴"}
    print(f"\n  {urgence_icons.get(brief.urgence, '⚪')} Urgence : {brief.urgence.upper()}")
    print(f"  📂 Domaine : {brief.domaine} / {brief.sous_domaine}")

    print(f"\n  📋 Résumé :")
    print(f"     {brief.resume_situation}")

    print(f"\n  🔑 Points clés :")
    for pt in brief.points_cles:
        print(f"     • {pt}")

    print(f"\n  📚 Références légales :")
    for ref in brief.references_legales:
        print(f"     [{ref.pertinence:.0%}] {ref.source}")
        print(f"          \"{ref.extrait}\"")

    print(f"\n  ❓ Questions à creuser :")
    for q in brief.questions_a_creuser:
        print(f"     • {q}")

    if brief.delais_importants:
        print(f"\n  ⏰ Délais : {brief.delais_importants}")

    print(f"\n  ⚡ Recommandation immédiate :")
    print(f"     {brief.recommandation_immediate}")

    print(f"\n{'='*60}")
    print(f"  Pipeline total : {total_ms:.0f}ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
