"""Turn sustained throughput into dollars per million output tokens — the cost headline.

Slice 5. The sweep (slice 4) measures tokens/sec at each load level; the GPU is rented by
the hour at a flat rate whether it's busy or idle. This module converts the two into the
benchmark's headline cost number: USD per 1,000,000 generated tokens.

Pure arithmetic — no sweep, no network, no matplotlib. The insight it exposes: at low
concurrency the GPU is mostly idle but still billed, so each token is expensive; as
batching saturates the GPU the same hourly cost spreads over far more tokens, so the cost
per million tokens falls. That drop is the whole point of the benchmark.
"""

SECONDS_PER_HOUR = 3600
TOKENS_PER_MILLION = 1_000_000


def dollars_per_million_tokens(
    throughput_tokens_per_second: float, gpu_dollars_per_hour: float
) -> float:
    """USD to generate one million output tokens at this sustained throughput.

    Derivation: the GPU costs gpu_dollars_per_hour regardless of load. In one hour at this
    throughput it emits throughput * 3600 tokens, so each token costs
    gpu_dollars_per_hour / (throughput * 3600); scale up to a million.
    """
    if throughput_tokens_per_second <= 0:
        raise ValueError(
            f"throughput must be positive, got {throughput_tokens_per_second}"
        )
    if gpu_dollars_per_hour < 0:
        raise ValueError(f"price must be non-negative, got {gpu_dollars_per_hour}")

    tokens_per_hour = throughput_tokens_per_second * SECONDS_PER_HOUR
    return gpu_dollars_per_hour / tokens_per_hour * TOKENS_PER_MILLION
