from cfr.ratelimit import TokenBucket


def test_allows_up_to_capacity_then_refuses():
    b = TokenBucket(capacity=5, refill_per_sec=0.0)
    assert all(b.allow("ip")[0] for _ in range(5))
    ok, wait = b.allow("ip")
    assert not ok and wait > 0  # inf when the bucket never refills


def test_clients_are_independent():
    b = TokenBucket(capacity=2, refill_per_sec=0.0)
    assert b.allow("a")[0] and b.allow("a")[0]
    assert not b.allow("a")[0]
    assert b.allow("b")[0], "one client must not exhaust another's quota"


def test_refills_over_time(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("cfr.ratelimit.time.monotonic", lambda: now[0])
    b = TokenBucket(capacity=2, refill_per_sec=1.0)
    assert b.allow("ip")[0] and b.allow("ip")[0]
    assert not b.allow("ip")[0]
    now[0] += 1.0
    assert b.allow("ip")[0]


def test_refill_is_capped_at_capacity(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("cfr.ratelimit.time.monotonic", lambda: now[0])
    b = TokenBucket(capacity=3, refill_per_sec=1.0)
    b.allow("ip")
    now[0] += 10_000
    assert all(b.allow("ip")[0] for _ in range(3))
    assert not b.allow("ip")[0], "idle time must not bank unlimited credit"


def test_retry_after_is_reported():
    b = TokenBucket(capacity=1, refill_per_sec=0.5)
    b.allow("ip")
    ok, wait = b.allow("ip")
    assert not ok
    assert 1.0 < wait <= 2.0


def test_sweep_bounds_memory():
    """A scraper rotating source addresses must not grow this without bound."""
    b = TokenBucket(capacity=1, refill_per_sec=1.0)
    for i in range(500):
        b.allow("ip-{}".format(i))
    b.sweep(max_entries=100)
    assert len(b._state) <= 250


def test_api_retry_after_is_valid_json():
    """`Infinity` is not valid JSON and JSON.parse rejects it in the browser."""
    import json

    from cfr.api import _retry_after

    for raw in (float("inf"), float("nan"), 3.14159, 0.0):
        json.loads(json.dumps({"retry_after_s": _retry_after(raw)}))
