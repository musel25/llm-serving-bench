from src.qps_sweep import SweepPoint
from src.run_summary import RunSummary
from src.sweep_plot import plot_sweep


def _point(concurrency, p50, p99, throughput):
    return SweepPoint(
        concurrency=concurrency,
        achieved_qps=throughput / 50,  # arbitrary; not exercised by the plot
        summary=RunSummary(
            request_count=40,
            p50_latency_s=p50,
            p99_latency_s=p99,
            total_completion_tokens=40 * 50,
            throughput_tokens_per_second=throughput,
        ),
    )


def test_plot_sweep_writes_a_nonempty_png(tmp_path):
    # No display needed: sweep_plot forces the headless Agg backend. We only assert that a
    # real, non-empty PNG lands on disk — rendering correctness is judged by eye, not unit test.
    points = [
        _point(1, 0.5, 0.6, 100.0),
        _point(8, 1.2, 2.0, 600.0),
        _point(32, 3.0, 5.0, 900.0),
    ]
    out = tmp_path / "sweep.png"

    result = plot_sweep(points, gpu_dollars_per_hour=0.59, output_path=out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0
