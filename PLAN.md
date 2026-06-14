# Plan — llm-serving-bench

> Single source of truth for state. /map writes it, /slice reads and updates it,
> /spike logs decisions here, /recap syncs it. Keep it under one screen: when a
> slice is done, its line compresses to one outcome sentence + commit ref.

## Goal

Build a **local async load generator** (the `openai` library + `asyncio`) that drives the
already-deployed vLLM endpoint and reports **tokens/sec, p50/p99 latency, and
cost-per-million-tokens under varying QPS**. This repo is the *measuring half*; the *serving
half* already exists in the sibling repo `a2a-scratch` and is treated as a given.
"Good enough for now" = one honest QPS sweep against the live endpoint that I can explain
end to end, plus a short writeup.

## Now

mode: BUILD · current: slice 4 (QPS sweep) · updated: 2026-06-14

## The endpoint we measure (lives in `a2a-scratch`, not here)

- URL: `https://musel25--vllm-server-serve.modal.run` (OpenAI-compatible)
- Model: `Qwen/Qwen2.5-0.5B-Instruct` · GPU: Modal **T4** (~$0.59/hr) · vLLM 0.21.0
- Container batches up to 32 concurrent requests (`@modal.concurrent(max_inputs=32)`)
- Scales down after 5 idle min → next request pays a **~3.5 min cold start** (warm up first)

## Slices (one per session, `/slice <name>`)

- [x] 1. **one request, timed** — DONE (8a9957f): sync `openai` call → `measure_one`
      returns end-to-end latency + `completion_tokens`, with a `tokens_per_second`
      property. Proven against the live endpoint; unit test green.
- [x] 2. **fire N at once (asyncio)** — DONE (df88ba4): `AsyncOpenAI` + `asyncio.gather`
      in `src/concurrent_requests.py` fires N concurrent requests, returns one Measurement
      each (in order). Proven live: wall-clock ≈ slowest request, not the sum. (`Measurement`
      extracted to its own module en route.)
- [x] 3. **aggregate → p50/p99 + throughput** — DONE (09328b9): `src/run_summary.py`
      reduces a batch to a `RunSummary` (p50/p99 latency via nearest-rank `percentile`,
      aggregate throughput = total tokens ÷ wall-clock). Pure/no-network; 6 unit tests.
      Live runner now prints the summary.
- [ ] 4. **QPS sweep** ← NEXT — drive rising load (concurrency 1→32) and record the metric
      curve. Learn: how latency degrades as QPS climbs — the actual benchmark. Warm up first.
- [ ] 5. **cost/M-tokens + writeup** — throughput × T4 price → $/1M tokens; honest README.

## Not building yet

- Streaming / time-to-first-token (TTFT) — a refinement; revisit as slice 6+ once
  end-to-end latency works. Keeps the asyncio slice from carrying two new ideas at once.
- Multiple models, SGLang, GPU comparison, plots, results database.
- Terminal chatbot — that's `a2a-scratch`'s slice 3, a different project.

## Decisions

- 2026-06-13 — **build a custom load generator**, not `vllm bench serve`. The point of this
  repo is to learn asyncio + the openai client + how the metrics are computed, not to read
  someone else's. (`a2a-scratch`'s plan used `vllm bench serve`; that's the other path.)
- 2026-06-13 — this repo holds **client code only**; it targets the endpoint deployed from
  `a2a-scratch`. Local machine has no NVIDIA GPU, so all serving stays on Modal.
- 2026-06-14 — **nearest-rank percentile, hand-rolled** (not `statistics.quantiles`/numpy):
  the point is to understand p50/p99, and nearest-rank ("sort, pick rank ⌈p/100·N⌉") is
  unambiguous and dependency-free. Tradeoff: coarse at small N.
- 2026-06-14 — **throughput takes `wall_clock_s` as an argument**, not derived from the
  measurements: requests overlap, so `sum(latencies)` overcounts. System throughput needs
  the batch's real elapsed time, which only the caller knows.

## What's fake / unproven

- Slices 1–3 proven. p99 is coarse at small N (nearest-rank → p99 of 10 ≈ the max);
  only meaningful once slice 4 fires larger batches.
- `summarize` doesn't guard `wall_clock_s <= 0` (caller always passes a perf_counter delta);
  add a guard if it ever becomes external input (slice 4/5).
- Reproducibility caveat: the endpoint lives in another repo. A fresh clone of this repo
  can't stand up serving on its own (acceptable for solo learning; note it in the writeup).
- Endpoint may be scaled down (cold) — slice 1 must expect a ~3.5 min first response.

## Open questions / spike candidates

- asyncio mental model (beginner) → settle with a 10-line `asyncio.gather`/`sleep` spike at
  the start of slice 2, before any network is involved.
- Open-loop (fixed req/sec) vs closed-loop (fixed concurrency) load — decide at slice 4.
