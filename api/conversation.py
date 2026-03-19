"""Conversational engine — guides the user through clarifying questions before generating a brief."""

import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

from api.classifier import classify_question
from api.retriever import embed_query, retrieve_legal_context
from api.brief_generator import generate_brief
from api.models import ChatMessage

logger = logging.getLogger("ekie.conversation")

# In-memory conversation store (swap for Redis in prod)
_conversations: dict[str, dict] = {}

SUMMARY_PROMPT = """Résume cette conversation juridique en UNE phrase claire qui capture le problème du salarié, les faits et le contexte.

Conversation :
{history}

Résumé en une phrase :"""

NEXT_QUESTION_PROMPT = """Tu es un assistant juridique français. Tu poses des questions de clarification courtes et précises pour comprendre la situation juridique du salarié.

Conversation :
{history}

Domaine détecté : {domaine} / {sous_domaine}

Questions déjà posées (NE PAS RÉPÉTER) :
{asked}

Règles :
- Pose UNE SEULE question courte et précise
- Ne répète jamais une question déjà posée
- Concentre-toi sur : les faits, les dates, les preuves, les démarches déjà faites
- Sois empathique mais professionnel
- Réponds directement avec la question, sans introduction

Question :"""

GREETING = "Bonjour, je suis l'assistant juridique Ekie. Décrivez votre situation, je vous poserai quelques questions avant de générer un brief pour votre avocat."


def get_conversation(conversation_id: str | None) -> tuple[str, dict]:
    """Get or create a conversation state."""
    if conversation_id and conversation_id in _conversations:
        return conversation_id, _conversations[conversation_id]

    cid = str(uuid.uuid4())[:8]
    state = {
        "messages": [],
        "domaine": None,
        "domaine_secondaire": None,
        "sous_domaine": None,
        "confiance": 0.0,
        "questions_asked": [],
    }
    _conversations[cid] = state
    return cid, state


def _format_history(messages: list[dict]) -> str:
    """Format messages for prompts."""
    lines = []
    for msg in messages:
        role = "Salarié" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role} : {msg['content']}")
    return "\n".join(lines)


def _summarize(messages: list[dict], gemini_client: genai.Client) -> str:
    """Summarize conversation into a single sentence for classification + retrieval."""
    history = _format_history(messages)
    prompt = SUMMARY_PROMPT.format(history=history)

    t0 = time.perf_counter()
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=200,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    summary = response.text.strip()
    logger.info("Summary generated in %dms: \"%s\"", (time.perf_counter() - t0) * 1000, summary[:100])
    return summary


def _generate_next_question(
    messages: list[dict],
    domaine: str,
    sous_domaine: str,
    questions_asked: list[str],
    gemini_client: genai.Client,
) -> str:
    """Generate the next clarifying question."""
    history = _format_history(messages)
    asked = "\n".join(f"- {q}" for q in questions_asked) if questions_asked else "Aucune"

    prompt = NEXT_QUESTION_PROMPT.format(
        history=history,
        domaine=domaine or "non détecté",
        sous_domaine=sous_domaine or "non détecté",
        asked=asked,
    )

    t0 = time.perf_counter()
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=150,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    question = response.text.strip()
    logger.info("Next question generated in %dms: \"%s\"", (time.perf_counter() - t0) * 1000, question[:100])
    return question


def _should_generate_brief(state: dict) -> bool:
    """Decide if we have enough info to generate the brief."""
    user_messages = [m for m in state["messages"] if m["role"] == "user"]
    total_user_chars = sum(len(m["content"]) for m in user_messages)

    # Generate brief if:
    # 1. High confidence + at least 3 user messages (after 2 bot questions)
    # 2. Or user provided a very detailed first message (>200 chars) with high confidence
    # 3. Or we've asked 4+ questions already (don't over-ask)
    if state["confiance"] >= 0.85 and len(user_messages) >= 3:
        logger.info("Brief ready: confiance=%.0f%% >= 85%% and %d user messages", state["confiance"] * 100, len(user_messages))
        return True
    if len(user_messages) == 1 and total_user_chars > 200 and state["confiance"] >= 0.85:
        logger.info("Brief ready: detailed first message (%d chars) with high confidence", total_user_chars)
        return True
    if len(state["questions_asked"]) >= 4:
        logger.info("Brief ready: asked %d questions already (max reached)", len(state["questions_asked"]))
        return True

    logger.info(
        "Not ready yet: confiance=%.0f%%, user_msgs=%d, chars=%d, questions_asked=%d",
        state["confiance"] * 100, len(user_messages), total_user_chars, len(state["questions_asked"]),
    )
    return False


def process_message(
    conversation_id: str | None,
    user_message: str,
    gemini_client: genai.Client,
    qdrant_client=None,
    mistral_client=None,
) -> dict:
    """Process a user message in the conversation flow.

    Returns dict with: conversation_id, action, message/brief, domaine, confiance.
    """
    cid, state = get_conversation(conversation_id)

    # Add user message
    state["messages"].append({"role": "user", "content": user_message})
    logger.info("─── Conversation %s — message #%d ───", cid, len(state["messages"]))
    logger.info("User: \"%s\"", user_message[:100])

    # Summarize conversation for classification
    if len(state["messages"]) == 1:
        summary = user_message
    else:
        summary = _summarize(state["messages"], gemini_client)

    # Classify + embed in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        cls_fut = pool.submit(classify_question, summary, gemini_client)
        emb_fut = pool.submit(embed_query, summary, gemini_client)
        classification = cls_fut.result()
        query_vector = emb_fut.result()

    state["domaine"] = classification["domaine"]
    state["domaine_secondaire"] = classification.get("domaine_secondaire")
    state["sous_domaine"] = classification["sous_domaine"]
    state["confiance"] = classification["confiance"]

    logger.info(
        "Classification: %s%s / %s (%.0f%%)",
        state["domaine"],
        f" + {state['domaine_secondaire']}" if state["domaine_secondaire"] else "",
        state["sous_domaine"],
        state["confiance"] * 100,
    )

    # Decide: brief or next question
    if _should_generate_brief(state):
        logger.info("→ Generating BRIEF")

        context_docs = retrieve_legal_context(
            question=summary,
            domaine=state["domaine"],
            confiance=state["confiance"],
            n=3,
            qdrant_client=qdrant_client,
            query_vector=query_vector,
            domaine_secondaire=state["domaine_secondaire"],
        )

        brief = generate_brief(
            question=summary,
            domaine=state["domaine"],
            sous_domaine=state["sous_domaine"],
            context_docs=context_docs,
            mistral_client=mistral_client,
        )

        return {
            "conversation_id": cid,
            "action": "brief",
            "brief": brief,
            "domaine": state["domaine"],
            "confiance": state["confiance"],
        }

    else:
        logger.info("→ Generating next QUESTION")

        next_q = _generate_next_question(
            state["messages"],
            state["domaine"],
            state["sous_domaine"],
            state["questions_asked"],
            gemini_client,
        )

        state["questions_asked"].append(next_q)
        state["messages"].append({"role": "assistant", "content": next_q})

        return {
            "conversation_id": cid,
            "action": "question",
            "message": next_q,
            "domaine": state["domaine"],
            "confiance": state["confiance"],
        }
