import pytest

from src.measurement import Measurement
from src.run_summary import RunSummary, percentile, summarize


# --- percentile (nearest-rank) -------------------------------------------------


def test_percentile_p50_picks_the_middle_by_rank():
    # rank = ceil(0.50 * 10) = 5  ->  sorted[4]  ->  5
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5


def test_percentile_p99_of_small_sample_is_the_worst_one():
    # rank = ceil(0.99 * 10) = 10  ->  sorted[9]  ->  10 (the slowest)
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 99) == 10


def test_percentile_sorts_unordered_input_first():
    # Same answer regardless of input order: p50 of these is 30.
    assert percentile([50, 10, 30, 20, 40], 50) == 30


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 50)


# --- summarize -----------------------------------------------------------------


def test_summarize_computes_percentiles_and_aggregate_throughput():
    # Latencies 0.1..0.5; tokens 10..50 (total 150). The batch's real wall-clock
    # was 0.6s (shorter than the 1.5s sum, because the requests overlapped).
    measurements = [
        Measurement(latency_s=0.1, completion_tokens=10),
        Measurement(latency_s=0.2, completion_tokens=20),
        Measurement(latency_s=0.3, completion_tokens=30),
        Measurement(latency_s=0.4, completion_tokens=40),
        Measurement(latency_s=0.5, completion_tokens=50),
    ]

    summary = summarize(measurements, wall_clock_s=0.6)

    assert summary == RunSummary(
        request_count=5,
        p50_latency_s=0.3,  # rank ceil(0.50*5)=3 -> sorted[2]
        p99_latency_s=0.5,  # rank ceil(0.99*5)=5 -> sorted[4] (slowest)
        total_completion_tokens=150,
        throughput_tokens_per_second=250.0,  # 150 tokens / 0.6 s
    )


def test_summarize_rejects_an_empty_batch():
    with pytest.raises(ValueError):
        summarize([], wall_clock_s=1.0)
