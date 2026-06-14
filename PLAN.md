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

mode: BUILD · current: slice 2 (fire N at once, asyncio) · updated: 2026-06-14

## The endpoint we measure (lives in `a2a-scratch`, not here)

- URL: `https://musel25--vllm-server-serve.modal.run` (OpenAI-compatible)
- Model: `Qwen/Qwen2.5-0.5B-Instruct` · GPU: Modal **T4** (~$0.59/hr) · vLLM 0.21.0
- Container batches up to 32 concurrent requests (`@modal.concurrent(max_inputs=32)`)
- Scales down after 5 idle min → next request pays a **~3.5 min cold start** (warm up first)

## Slices (one per session, `/slice <name>`)

- [x] 1. **one request, timed** — DONE (8a9957f): sync `openai` call → `measure_one`
      returns end-to-end latency + `completion_tokens`, with a `tokens_per_second`
      property. Proven against the live endpoint; unit test green.
- [ ] 2. **fire N at once (asyncio)** ← NEXT — `AsyncOpenAI` + `asyncio.gather` over ~10
      concurrent requests; collect each latency. Learn: `async`/`await`, why I/O-bound
      concurrency works here. (Opens with a no-network `asyncio.sleep` spike.)
- [ ] 3. **aggregate → p50/p99 + throughput** — list of (latency, tokens) → percentiles
      and total tokens/sec. Learn: what p50/p99 mean & how computed; latency vs throughput.
- [ ] 4. **QPS sweep** — drive rising load (concurrency 1→32) and record the metric curve.
      Learn: how latency degrades as QPS climbs — the actual benchmark. Warm up first.
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

## What's fake / unproven

- Slice 1 (`src/timed_request.py`) is proven. Nothing else built yet.
- Reproducibility caveat: the endpoint lives in another repo. A fresh clone of this repo
  can't stand up serving on its own (acceptable for solo learning; note it in the writeup).
- Endpoint may be scaled down (cold) — slice 1 must expect a ~3.5 min first response.

## Open questions / spike candidates

- asyncio mental model (beginner) → settle with a 10-line `asyncio.gather`/`sleep` spike at
  the start of slice 2, before any network is involved.
- Open-loop (fixed req/sec) vs closed-loop (fixed concurrency) load — decide at slice 4.
