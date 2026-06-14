import pytest

from src.cost import dollars_per_million_tokens


def test_clean_round_number():
    # 1000 tok/s for an hour = 3.6M tokens. At $3.60/hr that's exactly $1.00 / 1M tokens.
    assert dollars_per_million_tokens(1000.0, 3.60) == pytest.approx(1.00)


def test_higher_throughput_is_cheaper_per_token():
    # Same hourly price, twice the throughput -> half the cost per million tokens.
    slow = dollars_per_million_tokens(500.0, 0.59)
    fast = dollars_per_million_tokens(1000.0, 0.59)
    assert fast == pytest.approx(slow / 2)


def test_t4_ballpark():
    # T4 at ~$0.59/hr sustaining 800 tok/s: 800*3600 = 2.88M tok/hr -> 0.59/2.88 ≈ $0.205/M.
    assert dollars_per_million_tokens(800.0, 0.59) == pytest.approx(0.2049, abs=1e-3)


def test_zero_throughput_is_rejected():
    # Dividing the hourly cost over zero tokens is undefined (infinite cost) -> refuse.
    with pytest.raises(ValueError):
        dollars_per_million_tokens(0.0, 0.59)


def test_negative_price_is_rejected():
    with pytest.raises(ValueError):
        dollars_per_million_tokens(800.0, -1.0)
