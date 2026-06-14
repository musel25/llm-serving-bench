"""Fire many timed requests at once and collect one Measurement each.

Slice 2. Slice 1 timed a single request with the SYNC openai client; this module times
many at once with the ASYNC client so their network waits overlap instead of summing.
See temp.py for the from-scratch asyncio walkthrough that motivates this shape — the key
result being STEP 4: the sync client's blocking .create() would freeze the event loop and
serialise the waits, so concurrency requires AsyncOpenAI's awaitable .create().

Run it live against the deployed vLLM endpoint:
    uv run python -m src.concurrent_requests
"""

import asyncio
from time import perf_counter

from openai import AsyncOpenAI

from src.measurement import Measurement


async def measure_one_async(client: AsyncOpenAI, model: str, prompt: str) -> Measurement:
    """Async sibling of measure_one: await one request, timing the full round trip.

    The `await` is what makes concurrency possible: while this coroutine is parked
    waiting for the server, the event loop runs the other in-flight requests.
    """
    start = perf_counter()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_s = perf_counter() - start

    return Measurement(
        latency_s=latency_s,
        completion_tokens=response.usage.completion_tokens,
    )


async def measure_many(
    client: AsyncOpenAI, model: str, prompt: str, n: int
) -> list[Measurement]:
    """Fire n identical requests concurrently; return one Measurement per request, in order.

    asyncio.gather schedules all n coroutines on the loop at once and waits for them all,
    so the wall-clock total tracks the SLOWEST request, not the sum. gather preserves order,
    so result[i] is request i.
    """
    return await asyncio.gather(
        *(measure_one_async(client, model, prompt) for _ in range(n))
    )


if __name__ == "__main__":
    # Wiring lives here, separate from the logic above (same split as slice 1).
    client = AsyncOpenAI(
        base_url="https://musel25--vllm-server-serve.modal.run/v1",
        api_key="EMPTY",
    )
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Say hello in one short sentence."
    n = 10

    start = perf_counter()
    measurements = asyncio.run(measure_many(client, model, prompt, n))
    wall_s = perf_counter() - start

    slowest_s = max(m.latency_s for m in measurements)
    sum_of_latencies_s = sum(m.latency_s for m in measurements)
    total_tokens = sum(m.completion_tokens for m in measurements)

    print(f"fired {n} requests concurrently")
    print(f"wall-clock total : {wall_s:.3f} s")
    print(f"slowest single   : {slowest_s:.3f} s   <- wall should be close to THIS")
    print(f"sum of latencies : {sum_of_latencies_s:.3f} s   <- what doing them one-by-one would cost")
    print(f"total out tokens : {total_tokens}")
    print(f"throughput       : {total_tokens / wall_s:.1f} tokens/sec (aggregate)")
