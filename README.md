# llm-serving-bench

Benchmarking **vLLM** and **SGLang** inference serving for **Llama-3 8B / Mistral-7B** on
**Modal** GPU containers, behind an OpenAI-compatible endpoint.

A local load generator (using the `openai` library) drives the endpoint at varying **QPS**
and reports:

- **tokens/sec** (throughput)
- **p50 / p99 latency** (including time-to-first-token)
- **cost-per-million-tokens** (derived from Modal GPU billing)

The aim is a reproducible comparison of serving features — prefix caching, continuous
batching, speculative decoding — under load.

## Status

Early setup. No benchmarks run yet. See `PLAN.md` for the current goal and slice queue.
Nothing here is proven until `PLAN.md` and `git log` say so.

## Architecture (planned)

```
  local machine                         Modal (rented GPU container)
┌─────────────────┐   HTTP / OpenAI    ┌──────────────────────────────┐
│ load generator  │ ─────────────────► │ vLLM (or SGLang) serving the   │
│ (openai client) │ ◄───────────────── │ model on an OpenAI-compatible  │
│ records timings │                    │ /v1/chat/completions endpoint  │
└─────────────────┘                    └──────────────────────────────┘
```

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for environment and dependencies.

```bash
uv venv          # create the virtual environment
uv sync          # install dependencies from pyproject.toml / uv.lock
```

## Running

No runnable entry point yet. Commands will be documented here as slices land.

## Tests

```bash
uv run pytest    # (no tests yet)
```

## How this project is built

Development follows a learning-first contract (`CLAUDE.md`): small vertical slices, each
designed, tested, explained, and committed. State lives in `PLAN.md` and `git log`, not in
chat.
