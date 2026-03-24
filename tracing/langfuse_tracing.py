import os
from typing import Any, Optional, Tuple


def _langfuse_client():
    public = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret = os.getenv("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        return None
    from langfuse import Langfuse

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    return Langfuse(public_key=public, secret_key=secret, host=host)


def create_chat_trace() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """
    Create a Langfuse trace and span for the chat endpoint.
    Returns (trace, span, client) or (None, None, None) when Langfuse is not configured.
    """
    client = _langfuse_client()
    if client is None:
        return None, None, None
    trace = client.trace(name="chat-endpoint")
    span = trace.span(name="openai-completion")
    return trace, span, client


def finalize_span(span: Optional[Any], output: str) -> None:
    if span is None:
        return
    try:
        if hasattr(span, "update"):
            span.update(output=output)
    except Exception:
        pass
    try:
        if hasattr(span, "end"):
            span.end()
    except Exception:
        pass


def score_response_length(trace: Optional[Any], text: str) -> None:
    if trace is None:
        return
    value = min(float(len(text)) / 4096.0, 1.0)
    try:
        trace.score(name="response_length", value=value)
    except Exception:
        pass


def flush_client(client: Optional[Any]) -> None:
    if client is None:
        return
    try:
        client.flush()
    except Exception:
        pass
