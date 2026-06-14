"""Measure one chat request against an OpenAI-compatible endpoint.

This is the atom the whole benchmark is built from: send a single request, time
it end to end, and report how fast tokens came back. Later slices fire many of
these at once (slice 2) and aggregate them into percentiles (slice 3).

The logic here is endpoint-agnostic on purpose: the caller builds the OpenAI
client (with its base_url / api_key) and passes it in, so this module never
knows or cares which engine is behind the URL. That keeps it unit-testable and
lets slice 2 reuse one client across many concurrent requests.

Run it live against the deployed vLLM endpoint:
    uv run python -m src.timed_request
"""

from time import perf_counter

from openai import OpenAI

from src.measurement import Measurement


def measure_one(client: OpenAI, model: str, prompt: str) -> Measurement:
    """Send one chat request, timing the full round trip with a monotonic clock."""
    start = perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    latency_s = perf_counter() - start

    return Measurement(
        latency_s=latency_s,
        completion_tokens=response.usage.completion_tokens,
    )


if __name__ == "__main__":
    # Wiring lives here, separate from the logic above. The vLLM server was
    # started with no --api-key, so any non-empty string is accepted; the
    # openai client just requires the field to be set.
    client = OpenAI(
        base_url="https://musel25--vllm-server-serve.modal.run/v1",
        api_key="EMPTY",
    )
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt = "Say hello in one short sentence."

    m = measure_one(client, model, prompt)

    print(f"latency        : {m.latency_s:.3f} s")
    print(f"completion_toks: {m.completion_tokens}")
    print(f"tokens/sec     : {m.tokens_per_second:.1f}")
