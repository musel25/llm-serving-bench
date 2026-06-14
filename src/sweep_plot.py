"""Render a sweep into a PNG: the latency/throughput tradeoff and the cost curve.

Slice 5. An OPTIONAL reporting layer over the sweep (slice 4) and cost (slice 5). It draws
two stacked panels against concurrency on the x-axis:
  - top: p50 & p99 latency (left axis, seconds) and token throughput (right axis) — the
    classic benchmark bend, latency climbing while throughput plateaus at saturation.
  - bottom: dollars per million tokens — falling as batching amortises the fixed GPU cost.

This module imports matplotlib, so anything that only needs the NUMBERS (the sweep, the
tests, the table) must not import it. The live runner imports it lazily, inside __main__.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: render straight to a file, no display server
import matplotlib.pyplot as plt

from src.cost import dollars_per_million_tokens
from src.qps_sweep import SweepPoint


def plot_sweep(
    points: list[SweepPoint],
    gpu_dollars_per_hour: float,
    output_path: str | Path,
) -> Path:
    """Draw the sweep curve and save it as a PNG; return the path written.

    points must be ordered by ascending concurrency (as sweep() returns them).
    """
    if not points:
        raise ValueError("cannot plot an empty sweep")

    concurrency = [p.concurrency for p in points]
    p50 = [p.summary.p50_latency_s for p in points]
    p99 = [p.summary.p99_latency_s for p in points]
    throughput = [p.summary.throughput_tokens_per_second for p in points]
    cost = [
        dollars_per_million_tokens(p.summary.throughput_tokens_per_second, gpu_dollars_per_hour)
        for p in points
    ]

    fig, (ax_latency, ax_cost) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Top panel: latency on the left axis, throughput on a twinned right axis.
    ax_latency.plot(concurrency, p50, "o-", color="tab:blue", label="p50 latency")
    ax_latency.plot(concurrency, p99, "o--", color="tab:red", label="p99 latency")
    ax_latency.set_ylabel("latency (s)")
    ax_latency.set_title("Latency vs throughput as concurrency climbs")

    ax_throughput = ax_latency.twinx()
    ax_throughput.plot(
        concurrency, throughput, "s-", color="tab:green", label="throughput"
    )
    ax_throughput.set_ylabel("throughput (tokens/s)")

    # One combined legend for both y-axes.
    lines = ax_latency.get_lines() + ax_throughput.get_lines()
    ax_latency.legend(lines, [ln.get_label() for ln in lines], loc="upper left")

    # Bottom panel: cost per million tokens.
    ax_cost.plot(concurrency, cost, "o-", color="tab:purple")
    ax_cost.set_ylabel("$ / 1M tokens")
    ax_cost.set_xlabel("concurrency (in-flight requests)")
    ax_cost.set_title(f"Cost per million tokens (GPU @ ${gpu_dollars_per_hour:.2f}/hr)")
    ax_cost.set_xticks(concurrency)

    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)  # release the figure so repeated runs don't leak memory
    return output_path
