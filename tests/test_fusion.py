from cfr.search.fusion import rrf


def test_rrf_matches_hand_computation():
    a = [("a", 9.0), ("b", 8.0), ("c", 7.0)]
    b = [("c", 0.9), ("a", 0.8), ("d", 0.7)]
    out = dict(rrf([a, b], k=60))
    assert out["a"] == 1 / 61 + 1 / 62
    assert out["c"] == 1 / 63 + 1 / 61
    assert out["b"] == 1 / 62
    assert out["d"] == 1 / 63


def test_rrf_ranks_consensus_above_single_list_winner():
    """'a' is never first in either list but appears high in both; 'x' is first
    in one and absent from the other. Consensus should win."""
    a = [("x", 1.0), ("a", 0.9)]
    b = [("y", 1.0), ("a", 0.9)]
    order = [d for d, _ in rrf([a, b], k=60)]
    assert order[0] == "a"


def test_rrf_ignores_incomparable_score_scales():
    """BM25 scores are unbounded, cosine sits in [-1,1]. Only rank should matter."""
    bm25 = [("a", 143.2), ("b", 12.1)]
    dense = [("b", 0.81), ("a", 0.80)]
    scaled = [("a", 1_000_000.0), ("b", 0.0001)]
    assert rrf([bm25, dense], k=60) == rrf([scaled, dense], k=60)


def test_weights_apply():
    a = [("a", 1.0)]
    b = [("b", 1.0)]
    out = dict(rrf([a, b], k=60, weights=(2.0, 1.0)))
    assert out["a"] == 2 / 61
    assert out["b"] == 1 / 61
