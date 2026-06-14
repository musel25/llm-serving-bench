import asyncio
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI

from src.concurrent_requests import measure_many
from src.measurement import Measurement


class _FakeCompletions:
    """Stands in for `client.chat.completions`: an awaitable .create that returns a usage.

    We only need the bit measure_one_async actually reads: response.usage.completion_tokens.
    SimpleNamespace lets us build that nested shape (response.usage.completion_tokens)
    without dragging in the real openai response types.
    """

    def __init__(self, completion_tokens: int):
        self._completion_tokens = completion_tokens

    async def create(self, model, messages):
        return SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=self._completion_tokens)
        )


class _FakeAsyncClient:
    def __init__(self, completion_tokens: int = 7):
        self.chat = SimpleNamespace(completions=_FakeCompletions(completion_tokens))


def test_measure_many_returns_one_measurement_per_request():
    # No network: the fake client's create() returns instantly with 7 tokens.
    # asyncio.run drives the coroutine to completion, so we need no async test plugin.
    client = _FakeAsyncClient(completion_tokens=7)

    results = asyncio.run(measure_many(client, "fake-model", "hi", n=3))

    assert len(results) == 3
    assert all(isinstance(r, Measurement) for r in results)
    assert all(r.completion_tokens == 7 for r in results)
    assert all(r.latency_s > 0 for r in results)


@pytest.mark.smoke
def test_measure_many_hits_the_live_vllm_endpoint():
    # Skipped by default (network + possible cold start). Run with: uv run pytest -m smoke
    client = AsyncOpenAI(
        base_url="https://musel25--vllm-server-serve.modal.run/v1",
        api_key="EMPTY",
    )

    results = asyncio.run(
        measure_many(
            client, "Qwen/Qwen2.5-0.5B-Instruct", "Say hello in one short sentence.", n=5
        )
    )

    assert len(results) == 5
    assert all(r.completion_tokens > 0 for r in results)
    assert all(r.tokens_per_second > 0 for r in results)
