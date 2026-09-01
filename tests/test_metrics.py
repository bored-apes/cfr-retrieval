import math

from cfr.eval import metrics as M


def test_ndcg_matches_hand_computation():
    ranked = ["d1", "d2", "d3"]
    qrels = {"d1": 3, "d2": 0, "d3": 1}
    dcg = 7 / math.log2(2) + 0 + 1 / math.log2(4)
    idcg = 7 / math.log2(2) + 1 / math.log2(3)
    assert M.ndcg_at_k(ranked, qrels, 10) == dcg / idcg


def test_ndcg_is_one_for_ideal_ordering():
    qrels = {"a": 3, "b": 2, "c": 1}
    assert M.ndcg_at_k(["a", "b", "c"], qrels, 10) == 1.0


def test_ndcg_penalises_burying_the_best_result():
    qrels = {"a": 3, "b": 1}
    good = M.ndcg_at_k(["a", "b"], qrels, 10)
    bad = M.ndcg_at_k(["b", "a"], qrels, 10)
    assert good > bad


def test_metrics_are_none_for_unanswerable_queries():
    """Out-of-scope queries have no relevant document, so retrieval metrics are
    undefined - they must not be silently scored as zero."""
    qrels = {"a": 0}
    assert M.ndcg_at_k(["a"], qrels, 10) is None
    assert M.recall_at_k(["a"], qrels, 10) is None
    assert M.mrr_at_k(["a"], qrels, 10) is None


def test_recall_counts_only_graded_relevant():
    qrels = {"a": 2, "b": 1, "c": 0}
    assert M.recall_at_k(["a", "c"], qrels, 10) == 0.5
    assert M.recall_at_k(["a", "b"], qrels, 10) == 1.0


def test_recall_respects_cutoff():
    qrels = {"a": 1, "b": 1}
    assert M.recall_at_k(["a", "b"], qrels, 1) == 0.5


def test_mrr_uses_first_relevant_rank():
    qrels = {"z": 3}
    assert M.mrr_at_k(["a", "b", "z"], qrels, 10) == 1 / 3
    assert M.mrr_at_k(["a", "b"], qrels, 2) == 0.0


def test_dedupe_preserves_first_occurrence_order():
    assert M.dedupe_docs(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_bootstrap_ci_brackets_the_mean():
    vals = [0.5, 0.6, 0.7, 0.8, 0.9] * 10
    mean, lo, hi = M.bootstrap_ci(vals, iterations=500)
    assert lo <= mean <= hi
    assert abs(mean - 0.7) < 1e-9


def test_paired_bootstrap_detects_a_real_improvement():
    a = [0.20] * 60
    b = [0.60] * 60
    d = M.paired_bootstrap(a, b, iterations=500)
    assert d["delta"] > 0
    assert M.significant(d["lo"], d["hi"])


def test_paired_bootstrap_calls_noise_a_tie():
    """The guard against claiming wins that are not there."""
    a = [0.5, 0.2, 0.9, 0.4, 0.6, 0.3, 0.8, 0.1]
    b = [0.4, 0.3, 0.8, 0.5, 0.5, 0.4, 0.7, 0.2]
    d = M.paired_bootstrap(a, b, iterations=500)
    assert not M.significant(d["lo"], d["hi"])
