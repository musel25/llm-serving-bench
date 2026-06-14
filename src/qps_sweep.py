"""Drive the endpoint at rising load and record one summary per level — the benchmark.

Slice 4. Slices 2+3 fire ONE batch at ONE concurrency and summarize it. This module
runs that across a list of concurrency levels (1, 2, 4, ... 32) so we can see the curve:
how p50/p99 latency and throughput change as offered load climbs.

The load model is CLOSED-LOOP: we fix the number of in-flight requests (concurrency) and
let throughput fall out of it — as opposed to open-loop, where you fix a request rate and
let concurrency float (which can pile up unbounded once the server saturates). Within a
level we hold concurrency steady with an asyncio.Semaphore, pushing a fixed total of
requests through so every level gets the SAME, large enough sample for trustworthy
percentiles. Levels run one after another so they don't interfere.

This module is pure orchestration: it composes measure_one_async (slice 2, times one
request) and summarize (slice 3, computes the stats) and owns neither job itself.

Run it live against the deployed vLLM endpoint:
    uv run python -m src.qps_sweep
"""

import asyncio
from dataclasses import dataclass
from time import perf_counter

from openai import AsyncOpenAI

from src.concurrent_requests import measure_one_async
from src.measurement import Measurement
from src.run_summary import RunSummary, summarize


@dataclass(frozen=True)
class SweepPoint:
    """One load level's result: the offered concurrency and what came out of it."""

    concurrency: int  # offered load: how many requests we held in flight at once
    achieved_qps: float  # requests completed per second (requests / wall-clock)
    summary: RunSummary  # p50/p99 latency + token throughput for this level


async def _measure_at_concurrency(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    concurrency: int,
    total_requests: int,
) -> tuple[list[Measurement], float]:
    """Send total_requests, holding at most `concurrency` of them in flight at once.

    The Semaphore is a permit counter: `async with in_flight` waits for a free permit
    before letting a request hit the server and releases it when the request returns.
    So we launch all total_requests coroutines, but only `concurrency` can be past the
    gate at any instant — sustained load held at C, not one burst of C.
    """
    in_flight = asyncio.Semaphore(concurrency)

    async def one_capped() -> Measurement:
        async with in_flight:
            return await measure_one_async(client, model, prompt)

    start = perf_counter()
    measurements = await asyncio.gather(*(one_capped() for _ in range(total_requests)))
    wall_clock_s = perf_counter() - start
    return measurements, wall_clock_s


async def sweep(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    concurrency_levels: list[int],
    requests_per_level: int,
) -> list[SweepPoint]:
    """Run the load at each concurrency level in turn; return one SweepPoint per level.

    Levels run sequentially: a higher level must not steal capacity from a lower one,
    or the curve would be meaningless. requests_per_level fixes the sample size at every
    level so percentiles are comparable across the curve.
    """
    points = []
    for concurrency in concurrency_levels:
        measurements, wall_clock_s = await _measure_at_concurrency(
            client, model, prompt, concurrency, requests_per_level
        )
        points.append(
            SweepPoint(
                concurrency=concurrency,
                achieved_qps=len(measurements) / wall_clock_s,
                summary=summarize(measurements, wall_clock_s),
            )
        )
    return points


if __name__ == "__main__":
    # Wiring lives here, separate from the logic above (same split as slices 1-2).
    client = AsyncOpenAI(
        base_url="https://musel25--vllm-server-serve.modal.run/v1",
        api_key="EMPTY",
    )
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Say hello in one short sentence."
    concurrency_levels = [1, 2, 4, 8, 16, 32]  # 32 = the container's max_inputs
    requests_per_level = 40

    # Warm up first: the endpoint scales to zero after 5 idle min, and the first request
    # then pays a ~3.5 min cold start. One throwaway call absorbs that so it doesn't land
    # inside (and wreck) the concurrency=1 row.
    print("warming up (one request; may take ~3.5 min if the endpoint is cold)...")
    asyncio.run(measure_one_async(client, model, prompt))
    print("warm. running sweep:\n")

    points = asyncio.run(
        sweep(client, model, prompt, concurrency_levels, requests_per_level)
    )

    header = f"{'conc':>4}  {'qps':>7}  {'p50 (s)':>9}  {'p99 (s)':>9}  {'tok/s':>8}"
    print(header)
    print("-" * len(header))
    for p in points:
        print(
            f"{p.concurrency:>4}  "
            f"{p.achieved_qps:>7.2f}  "
            f"{p.summary.p50_latency_s:>9.3f}  "
            f"{p.summary.p99_latency_s:>9.3f}  "
            f"{p.summary.throughput_tokens_per_second:>8.1f}"
        )
    print(f"\n({requests_per_level} requests per level)")
