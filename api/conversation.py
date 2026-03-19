"""Conversational engine that asks clarifying questions before generating a brief."""

import logging
import re
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

from api.brief_generator import generate_brief
from api.classifier import classify_question
from api.retriever import embed_query, retrieve_legal_context

logger = logging.getLogger("ekie.conversation")

# In-memory conversation store (swap for Redis in prod)
_conversations: dict[str, dict] = {}

MIN_USER_MESSAGES_FOR_BRIEF = 4
DETAILED_FIRST_MESSAGE_CHARS = 200
MIN_USER_MESSAGES_FOR_EARLY_BRIEF = 3
MIN_QUESTIONS_FOR_EARLY_BRIEF = 2
MIN_TOTAL_CHARS_FOR_EARLY_BRIEF = 80

NON_SUBSTANTIVE_REPLY = (
    "J'ai besoin d'un peu plus de contexte. Decris brievement le probleme "
    "juridique : ce qu'il s'est passe, avec qui, et si possible quand."
)

GREETING_PATTERNS = (
    re.compile(r"^h[eai]?l{1,2}o+$"),
    re.compile(r"^hey+$"),
    re.compile(r"^yo+$"),
    re.compile(r"^bonjou+r+$"),
    re.compile(r"^salu+t+$"),
    re.compile(r"^coucou+$"),
)

GREETING_TOKENS = {
    "bjr",
    "bonjour",
    "bonsoir",
    "cc",
    "coucou",
    "hello",
    "hey",
    "salut",
    "yo",
}
FILLER_TOKENS = {
    "aide",
    "help",
    "merci",
    "ok",
    "okok",
    "oki",
    "please",
    "stp",
    "svp",
    "test",
}
LEGAL_SIGNAL_TOKENS = {
    "agression",
    "amende",
    "arnaque",
    "bijou",
    "bijoux",
    "caution",
    "contrat",
    "divorce",
    "employeur",
    "expulsion",
    "facture",
    "garde",
    "harcelement",
    "impot",
    "licencie",
    "loyer",
    "plainte",
    "preavis",
    "salaire",
    "succession",
    "travail",
    "vol",
    "vole",
    "volee",
}

SUMMARY_PROMPT = """Resume cette conversation juridique en UNE phrase claire qui capture le probleme du salarie, les faits et le contexte.

Conversation :
{history}

Resume en une phrase :"""

NEXT_QUESTION_PROMPT = """Tu es un assistant juridique francais. Tu poses des questions de clarification courtes et precises pour comprendre la situation juridique du salarie.

Conversation :
{history}

Domaine detecte : {domaine} / {sous_domaine}

Questions deja posees (NE PAS REPETER) :
{asked}

Regles :
- Pose UNE SEULE question courte et precise
- Ne repete jamais une question deja posee
- Concentre-toi sur : les faits, les dates, les preuves, les demarches deja faites
- Sois empathique mais professionnel
- Reponds directement avec la question, sans introduction

Question :"""

GREETING = (
    "Bonjour, je suis l'assistant juridique Ekie. Decrivez votre situation, "
    "je vous poserai quelques questions avant de generer un brief pour votre avocat."
)


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
        role = "Salarie" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role} : {msg['content']}")
    return "\n".join(lines)


def _normalize_text(text: str) -> str:
    """Normalize text for lightweight message-quality heuristics."""
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def _is_greeting_token(token: str) -> bool:
    """Detect simple greeting variants, including elongated spellings."""
    return token in GREETING_TOKENS or any(pattern.fullmatch(token) for pattern in GREETING_PATTERNS)


def _is_meaningful_user_message(message: str) -> bool:
    """Ignore greetings and filler messages so they do not count toward the brief."""
    normalized = _normalize_text(message)
    if not normalized:
        return False

    tokens = normalized.split()
    if len(tokens) <= 3 and all(_is_greeting_token(token) or token in FILLER_TOKENS for token in tokens):
        return False

    if len(normalized) < 12 and not any(token in LEGAL_SIGNAL_TOKENS for token in tokens):
        return False

    if len(tokens) == 1 and len(tokens[0]) <= 3 and tokens[0] not in LEGAL_SIGNAL_TOKENS:
        return False

    return True


def _is_useful_follow_up_answer(message: str) -> bool:
    """Allow short follow-up answers once the assistant has asked a real question."""
    normalized = _normalize_text(message)
    if not normalized:
        return False

    tokens = normalized.split()
    if all(_is_greeting_token(token) or token in FILLER_TOKENS for token in tokens):
        return False

    return True


def _summarize(messages: list[dict], gemini_client: genai.Client) -> str:
    """Summarize conversation into a single sentence for classification and retrieval."""
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
        domaine=domaine or "non detecte",
        sous_domaine=sous_domaine or "non detecte",
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
    logger.info(
        "Next question generated in %dms: \"%s\"",
        (time.perf_counter() - t0) * 1000,
        question[:100],
    )
    return question


def _should_generate_brief(state: dict) -> bool:
    """Decide if we have enough info to generate the brief."""
    user_messages = [message for message in state["messages"] if message["role"] == "user"]
    total_user_chars = sum(len(message["content"]) for message in user_messages)

    # Keep one more question in the loop by default so the brief has enough factual detail.
    if state["confiance"] >= 0.85 and len(user_messages) >= MIN_USER_MESSAGES_FOR_BRIEF:
        logger.info(
            "Brief ready: confiance=%.0f%% >= 85%% and %d substantive user messages",
            state["confiance"] * 100,
            len(user_messages),
        )
        return True
    if (
        state["confiance"] >= 0.85
        and len(user_messages) >= MIN_USER_MESSAGES_FOR_EARLY_BRIEF
        and len(state["questions_asked"]) >= MIN_QUESTIONS_FOR_EARLY_BRIEF
        and total_user_chars >= MIN_TOTAL_CHARS_FOR_EARLY_BRIEF
    ):
        logger.info(
            "Brief ready early: confiance=%.0f%%, user_msgs=%d, chars=%d, questions_asked=%d",
            state["confiance"] * 100,
            len(user_messages),
            total_user_chars,
            len(state["questions_asked"]),
        )
        return True
    if (
        len(user_messages) == 1
        and total_user_chars > DETAILED_FIRST_MESSAGE_CHARS
        and state["confiance"] >= 0.85
    ):
        logger.info("Brief ready: detailed first message (%d chars) with high confidence", total_user_chars)
        return True
    if len(state["questions_asked"]) >= 4:
        logger.info("Brief ready: asked %d questions already (max reached)", len(state["questions_asked"]))
        return True

    logger.info(
        "Not ready yet: confiance=%.0f%%, user_msgs=%d, chars=%d, questions_asked=%d",
        state["confiance"] * 100,
        len(user_messages),
        total_user_chars,
        len(state["questions_asked"]),
    )
    return False


def process_message(
    conversation_id: str | None,
    user_message: str,
    gemini_client: genai.Client,
    qdrant_client=None,
    mistral_client=None,
) -> dict:
    """Process a user message in the conversation flow."""
    cid, state = get_conversation(conversation_id)
    cleaned_message = user_message.strip()
    has_follow_up_context = bool(state["questions_asked"])

    if not _is_meaningful_user_message(cleaned_message) and not (
        has_follow_up_context and _is_useful_follow_up_answer(cleaned_message)
    ):
        logger.info("Ignoring non-substantive user message in conversation %s: %r", cid, user_message)
        return {
            "conversation_id": cid,
            "action": "question",
            "message": NON_SUBSTANTIVE_REPLY,
            "domaine": state["domaine"],
            "confiance": state["confiance"],
        }

    state["messages"].append({"role": "user", "content": cleaned_message})
    logger.info("--- Conversation %s - message #%d ---", cid, len(state["messages"]))
    logger.info("User: \"%s\"", cleaned_message[:100])

    if len(state["messages"]) == 1:
        summary = cleaned_message
    else:
        summary = _summarize(state["messages"], gemini_client)

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

    if _should_generate_brief(state):
        logger.info("-> Generating BRIEF")

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

    logger.info("-> Generating next QUESTION")

    next_question = _generate_next_question(
        state["messages"],
        state["domaine"],
        state["sous_domaine"],
        state["questions_asked"],
        gemini_client,
    )

    state["questions_asked"].append(next_question)
    state["messages"].append({"role": "assistant", "content": next_question})

    return {
        "conversation_id": cid,
        "action": "question",
        "message": next_question,
        "domaine": state["domaine"],
        "confiance": state["confiance"],
    }
