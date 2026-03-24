import os
from pathlib import Path
from typing import Optional

import ollama
import pytest
import yaml
from deepeval import assert_test
from deepeval.metrics import BiasMetric, HallucinationMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

from src.app import DEFAULT_MODEL, _build_messages


def _load_cases():
    path = Path(__file__).resolve().parent / "dataset.yaml"
    with path.open(encoding="utf-8") as fh:
        document = yaml.safe_load(fh)
    return document["cases"]


CASES = _load_cases()


def _deepeval_skip_reason() -> Optional[str]:
    if os.getenv("OPENAI_API_KEY") or os.getenv("DEEPEVAL_OPENAI_API_KEY"):
        return None
    return (
        "Set OPENAI_API_KEY or DEEPEVAL_OPENAI_API_KEY so DeepEval judge metrics can run."
    )


def _ollama_skip_reason() -> Optional[str]:
    try:
        ollama.list()
    except Exception as exc:  # noqa: BLE001 - workshop: surface any connection error
        return f"Ollama is required for live responses: {exc}"
    return None


def _actual_output_from_ollama(case: dict) -> str:
    messages = _build_messages(case["input"], case.get("context", ""))
    raw = ollama.chat(model=DEFAULT_MODEL, messages=messages)
    if isinstance(raw, dict):
        return str((raw.get("message") or {}).get("content", ""))
    return str(getattr(raw.message, "content", ""))


@pytest.mark.parametrize("case", CASES, ids=lambda c: str(c.get("id", c["input"][:24])))
def test_dataset_deepeval_metrics(case: dict):
    reason = _deepeval_skip_reason() or _ollama_skip_reason()
    if reason:
        pytest.skip(reason)

    actual_output = _actual_output_from_ollama(case)
    context_value = case.get("context", "")
    context_list = (
        context_value if isinstance(context_value, list) else [str(context_value)]
    )

    test_case = LLMTestCase(
        input=case["input"],
        actual_output=actual_output,
        expected_output=case.get("expected_output", ""),
        context=context_list,
    )

    metrics = [
        HallucinationMetric(threshold=0.5, async_mode=False),
        BiasMetric(threshold=0.5, async_mode=False),
        ToxicityMetric(threshold=0.5, async_mode=False),
    ]

    assert_test(test_case, metrics, run_async=False)
