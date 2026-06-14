import asyncio
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI

from src.qps_sweep import SweepPoint, sweep
from src.run_summary import RunSummary


class _InstantCompletions:
    """Awaitable .create that returns instantly with a fixed token count (no network)."""

    def __init__(self, completion_tokens: int = 5):
        self._completion_tokens = completion_tokens

    async def create(self, model, messages):
        return SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=self._completion_tokens)
        )


class _InstantClient:
    def __init__(self, completion_tokens: int = 5):
        self.chat = SimpleNamespace(completions=_InstantCompletions(completion_tokens))


class _ConcurrencyTrackingCompletions:
    """Records how many .create calls are in flight at once.

    Each call bumps a counter, parks on a real await (so sibling calls admitted by
    the semaphore pile up alongside it), then drops the counter. asyncio is single
    threaded, so the counter is only touched around the await — no real data race.
    max_in_flight is the high-water mark the semaphore should be holding down.
    """

    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0

    async def create(self, model, messages):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)  # hold the slot so others can accumulate
        self.in_flight -= 1
        return SimpleNamespace(usage=SimpleNamespace(completion_tokens=5))


class _ConcurrencyTrackingClient:
    def __init__(self):
        self.completions = _ConcurrencyTrackingCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_sweep_returns_one_point_per_level():
    client = _InstantClient(completion_tokens=5)

    points = asyncio.run(
        sweep(
            client,
            "fake-model",
            "hi",
            concurrency_levels=[1, 2, 4],
            requests_per_level=8,
        )
    )

    assert len(points) == 3
    assert all(isinstance(p, SweepPoint) for p in points)
    assert [p.concurrency for p in points] == [1, 2, 4]
    assert all(isinstance(p.summary, RunSummary) for p in points)
    assert all(p.achieved_qps > 0 for p in points)


def test_each_level_sends_requests_per_level_requests():
    # requests_per_level fixes the sample size at EVERY level, independent of concurrency.
    client = _InstantClient()

    points = asyncio.run(
        sweep(client, "fake-model", "hi", concurrency_levels=[1, 8], requests_per_level=10)
    )

    assert all(p.summary.request_count == 10 for p in points)


def test_sweep_caps_in_flight_requests_at_the_concurrency_level():
    # The core new behavior: with concurrency=3 and 12 requests, never more than 3 in flight.
    client = _ConcurrencyTrackingClient()

    asyncio.run(
        sweep(client, "fake-model", "hi", concurrency_levels=[3], requests_per_level=12)
    )

    assert client.completions.max_in_flight == 3


@pytest.mark.smoke
def test_sweep_hits_the_live_vllm_endpoint():
    # Skipped by default (network + possible cold start). Run with: uv run pytest -m smoke
    client = AsyncOpenAI(
        base_url="https://musel25--vllm-server-serve.modal.run/v1",
        api_key="EMPTY",
    )

    points = asyncio.run(
        sweep(
            client,
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Say hello in one short sentence.",
            concurrency_levels=[1, 4],
            requests_per_level=8,
        )
    )

    assert [p.concurrency for p in points] == [1, 4]
    assert all(p.achieved_qps > 0 for p in points)
    assert all(p.summary.total_completion_tokens > 0 for p in points)
