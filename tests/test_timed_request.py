import pytest
from openai import OpenAI

from src.measurement import Measurement
from src.timed_request import measure_one


def test_tokens_per_second_is_tokens_divided_by_latency():
    # 100 tokens produced over 2 seconds -> 50 tokens/sec
    m = Measurement(latency_s=2.0, completion_tokens=100)

    assert m.tokens_per_second == 50.0


@pytest.mark.smoke
def test_measure_one_hits_the_live_vllm_endpoint():
    # Skipped by default (network + possible cold start). Run with: uv run pytest -m smoke
    client = OpenAI(
        base_url="https://musel25--vllm-server-serve.modal.run/v1",
        api_key="EMPTY",
    )

    m = measure_one(client, "Qwen/Qwen2.5-0.5B-Instruct", "Say hello in one short sentence.")

    assert m.latency_s > 0
    assert m.completion_tokens > 0
    assert m.tokens_per_second > 0
