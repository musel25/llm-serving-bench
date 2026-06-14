"""Condense a batch of Measurements into a run's headline statistics.

Slice 3. measure_many (slice 2) returns a raw list of Measurements; this module turns
that list into the numbers a benchmark actually reports: p50/p99 latency and aggregate
throughput. It is pure arithmetic — no network, no openai, no asyncio — so it is fully
unit-testable without touching the endpoint.

Two ideas worth keeping straight:
  - LATENCY is per-request "how long did one call take" -> we report p50 (typical) and
    p99 (tail: 99% of calls were at least this fast; the worst 1% were slower).
  - THROUGHPUT is whole-system "how many tokens/sec did the batch sustain". Because the
    requests overlapped, this needs the batch's real WALL-CLOCK time, which only the
    caller knows — so summarize() takes it as an argument rather than guessing from the
    per-request latencies (whose sum would massively overcount).
"""

import math
from dataclasses import dataclass

from src.measurement import Measurement


def percentile(values: list[float], p: float) -> float:
    """The p-th percentile by the nearest-rank method (0 < p <= 100).

    Sort ascending, then take the value at rank ceil(p/100 * N). "p99 latency" is then
    literally: line the latencies up slowest-last, and pick the one 99% of the way up.
    """
    if not values:
        raise ValueError("cannot take a percentile of no values")
    if not 0 < p <= 100:
        raise ValueError(f"percentile p must be in (0, 100], got {p}")

    ordered = sorted(values)
    rank = math.ceil(p / 100 * len(ordered))  # 1-based rank
    return ordered[rank - 1]


@dataclass(frozen=True)
class RunSummary:
    """The headline numbers for one batch of requests."""

    request_count: int
    p50_latency_s: float  # typical per-request latency
    p99_latency_s: float  # tail per-request latency (worst ~1%)
    total_completion_tokens: int  # output tokens produced across the batch
    throughput_tokens_per_second: float  # whole-system: total tokens / wall-clock


def summarize(measurements: list[Measurement], wall_clock_s: float) -> RunSummary:
    """Reduce a batch of Measurements to a RunSummary.

    wall_clock_s is the real elapsed time the caller measured for the whole batch; it is
    what makes throughput a system metric rather than a per-request one.
    """
    if not measurements:
        raise ValueError("cannot summarize an empty batch of measurements")

    latencies = [m.latency_s for m in measurements]
    total_tokens = sum(m.completion_tokens for m in measurements)

    return RunSummary(
        request_count=len(measurements),
        p50_latency_s=percentile(latencies, 50),
        p99_latency_s=percentile(latencies, 99),
        total_completion_tokens=total_tokens,
        throughput_tokens_per_second=total_tokens / wall_clock_s,
    )
