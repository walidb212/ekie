"""Pydantic v2 schemas for the Ekie Legal Router API."""

from typing import Literal

from pydantic import BaseModel, Field


class QuestionInput(BaseModel):
    question: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        examples=["Mon employeur refuse de me payer mes heures supplémentaires"],
    )


class ArticleRef(BaseModel):
    source: str = Field(..., description="e.g. Code du travail - Art. L1234-1")
    extrait: str = Field(..., max_length=500, description="Relevant extract")
    pertinence: float = Field(..., ge=0.0, le=1.0, description="Similarity score")


class BriefAvocat(BaseModel):
    domaine: str
    sous_domaine: str
    urgence: Literal["faible", "normale", "haute", "critique"]
    resume_situation: str
    points_cles: list[str] = Field(..., min_length=1, max_length=5)
    questions_a_creuser: list[str] = Field(..., min_length=1, max_length=5)
    references_legales: list[ArticleRef]
    delais_importants: str | None = None
    recommandation_immediate: str


class APIResponse(BaseModel):
    success: bool
    brief: BriefAvocat | None = None
    processing_time_ms: int
    error: str | None = None


# ── Conversation models ──────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(..., min_length=1, max_length=1000)


class ChatResponse(BaseModel):
    conversation_id: str
    action: Literal["question", "brief"]
    message: str | None = None
    brief: BriefAvocat | None = None
    domaine: str | None = None
    confiance: float | None = None
    processing_time_ms: int
