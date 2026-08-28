"""
Follow-up questions about a medicine already identified.

Stateless with respect to the medicine: the client sends the scan/search result
it is asking about, so the server never has to guess which of several recent
scans a question refers to. That was a real ambiguity in the system this
replaces, where "is it safe?" could attach to the wrong upload.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..auth import AuthUser, get_current_user_opt
from ..deps import get_state

router = APIRouter()

MAX_QUESTION = 400
MAX_HISTORY = 12


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)
    subject: str = Field(min_length=1, max_length=200)
    """Which medicine the question is about — the brand name from the scan or
    search result the user is looking at.

    The server re-resolves this rather than accepting a fact sheet from the
    client. That costs one retrieval (~250 ms) and buys two things: the sheet
    is complete, carrying the price working and the Jan Aushadhi alternatives
    that a trimmed client payload would have dropped, and every figure the
    answer is grounded against was produced server-side this request."""

    history: list[ChatMessage] = Field(default_factory=list, max_length=MAX_HISTORY)


@router.post("/chat")
def chat(
    request: ChatRequest,
    user: AuthUser | None = Depends(get_current_user_opt),
) -> dict:
    """Answer a question about one medicine.

    The response always states which path produced it. `grounded: true` means
    every claim traces to the retrieved fact sheet and the contextual grounding
    filter verified it. `grounded: false` means the databases did not cover the
    question and the model answered from its own training — labelled, screened
    against the denied-topic policy, and carrying a disclaimer.
    """
    state = get_state()
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail={"error": "service not ready", "problems": state.startup_errors},
        )
    if state.chat_answerer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Follow-up questions need Amazon Bedrock, which is not configured on "
                "this deployment. Scanning, pricing, alternatives and interaction "
                "checks are unaffected."
            ),
        )

    result = state.orchestrator.analyse_text(request.subject, explain=False)
    turn = state.chat_answerer.answer(
        request.question,
        result,
        history=[m.model_dump() for m in request.history],
        subject=request.subject,
    )
    return {"question": request.question, "answer": turn.to_dict()}
