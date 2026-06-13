# Plan — llm-serving-bench

> Single source of truth for state. /map writes it, /slice reads and updates it,
> /spike logs decisions here, /recap syncs it. Keep it under one screen: when a
> slice is done, its line compresses to one outcome sentence + commit ref.

## Goal
Benchmark vLLM (and later SGLang) serving Llama-3 8B / Mistral-7B on Modal GPU containers
behind an OpenAI-compatible endpoint, driven by a local load generator (the `openai`
library), reporting tokens/sec, p50/p99 latency, and cost-per-million-tokens under varying
QPS. "Good enough for now" = one model on vLLM, one real Modal endpoint, one honest
latency/throughput report I can explain end to end.

## Now
mode: BUILD · current: project setup, then /map to design the slice queue · updated: 2026-06-13

## Slices
- [x] 1. CLI skeleton parses one command — argparse + 2 tests (a1b2c3d)
- [ ] 2. <next slice: what it does + what I'll learn from it>
- [ ] 3. ...

## Not building yet
- <thing> — <why it can safely wait>

## Decisions
- 2026-06-11 — storage: sqlite over postgres — zero ops, single user; revisit if multi-user

## What's fake / unproven
- payment step stubbed (always returns success)
- no test covers empty input

## Open questions / spike candidates
- does <lib> handle streaming under load? → /spike
