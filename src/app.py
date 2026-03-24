import os
from typing import Any, Dict, List

import ollama
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.prompts import ChatPromptTemplate
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from pydantic import BaseModel, Field

from monitoring.metrics import (
    llm_tokens_completion,
    llm_tokens_prompt,
    llm_tokens_total,
)
from tracing.langfuse_tracing import (
    create_chat_trace,
    finalize_span,
    flush_client,
    score_response_length,
)

load_dotenv()

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma:4b")

app = FastAPI(title="LLMSecOps Workshop", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(..., description="User question.")
    context: str = Field(
        default="",
        description="Optional grounding context for the model.",
    )


class ChatResponse(BaseModel):
    answer: str
    model: str


def _build_messages(question: str, context: str) -> List[Dict[str, str]]:
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant. Use the context when it helps; "
                "if context is empty, answer from general knowledge.\n\nContext:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    rendered = template.format_messages(context=context or "(none)", question=question)
    out: List[Dict[str, str]] = []
    for m in rendered:
        role = m.type if m.type != "human" else "user"
        if role == "ai":
            role = "assistant"
        out.append({"role": role, "content": m.content})
    return out


def _response_payload(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    return {
        "message": {"content": getattr(resp.message, "content", "")},
        "prompt_eval_count": getattr(resp, "prompt_eval_count", None),
        "eval_count": getattr(resp, "eval_count", None),
    }


def _record_token_metrics(model: str, payload: Dict[str, Any]) -> None:
    prompt_tokens = payload.get("prompt_eval_count") or 0
    completion_tokens = payload.get("eval_count") or 0
    try:
        p = int(prompt_tokens)
        c = int(completion_tokens)
    except (TypeError, ValueError):
        return
    if p > 0:
        llm_tokens_prompt.labels(model=model).inc(p)
    if c > 0:
        llm_tokens_completion.labels(model=model).inc(c)
    if p + c > 0:
        llm_tokens_total.labels(model=model).inc(p + c)


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    messages = _build_messages(req.question, req.context)
    trace, span, lf_client = create_chat_trace()
    if span is not None:
        try:
            span.update(
                input={"question": req.question, "context": req.context},
                metadata={"model": DEFAULT_MODEL},
            )
        except Exception:
            pass

    try:
        raw = ollama.chat(model=DEFAULT_MODEL, messages=messages)
    except Exception as exc:
        finalize_span(span, "")
        flush_client(lf_client)
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc}") from exc

    payload = _response_payload(raw)
    answer = ""
    msg = payload.get("message") or {}
    if isinstance(msg, dict):
        answer = str(msg.get("content", ""))
    _record_token_metrics(DEFAULT_MODEL, payload)
    finalize_span(span, answer)
    score_response_length(trace, answer)
    flush_client(lf_client)

    return ChatResponse(answer=answer, model=DEFAULT_MODEL)
