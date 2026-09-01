"""Citation verification and the abstention gate."""

import pytest

from cfr import answer as A
from cfr.search import Hit
from cfr.search.pipeline import SearchResult

SRC = ("A large quantity generator accumulates hazardous waste on site for no more "
       "than 90 days, unless in compliance with the accumulation time limit extension.")


def make_hit(text=SRC, char_start=4102, rerank=0.9):
    return Hit(chunk_id="ecfr:40:262.17#structured:0", doc_id="ecfr:40:262.17",
               score=0.5, text=text, heading_path="p", citation="40 CFR 262.17",
               heading="Conditions", char_start=char_start,
               char_end=char_start + len(text), rerank_score=rerank)


@pytest.mark.parametrize("quote", [
    "accumulates hazardous waste on site for no more than 90 days",
    "accumulates hazardous   waste on site\nfor no more than 90 days",
    "ACCUMULATES HAZARDOUS WASTE ON SITE FOR NO MORE THAN 90 DAYS",
    "“accumulates hazardous waste on site for no more than 90 days”",
])
def test_real_quotes_verify_despite_cosmetic_edits(quote):
    assert A.locate_quote(quote, SRC) is not None


@pytest.mark.parametrize("quote", [
    "must be incinerated within 30 days of generation",
    "the generator shall notify the Administrator by rail",
    "90 days",           # too short to be evidence of anything
    "",
])
def test_fabricated_or_useless_quotes_are_rejected(quote):
    assert A.locate_quote(quote, SRC) is None


def test_offsets_resolve_against_the_document():
    hit = make_hit()
    verified, dropped = A.verify_citations(
        [{"source": 1, "quote": "for no more than 90 days"}], [hit])
    assert dropped == 0
    v = verified[0]
    assert v["doc_char_start"] == 4102 + SRC.index("for no more than 90 days")
    assert v["quote"] == "for no more than 90 days"


def test_hallucinated_citation_is_dropped_and_counted():
    hit = make_hit()
    verified, dropped = A.verify_citations([
        {"source": 1, "quote": "for no more than 90 days"},
        {"source": 1, "quote": "must be shipped by rail within one week"},
        {"source": 7, "quote": "source index out of range"},
        {"source": "not a number", "quote": "x"},
    ], [hit])
    assert len(verified) == 1
    assert dropped == 3


def test_abstains_below_threshold(monkeypatch):
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.30)
    res = SearchResult(query="pizza", hits=[make_hit(rerank=0.05)])
    abstain, reason, score = A.should_abstain(res)
    assert abstain and reason == "low_confidence"


def test_abstains_on_empty_results():
    abstain, reason, _ = A.should_abstain(SearchResult(query="x", hits=[]))
    assert abstain and reason == "no_results"


def _hits(scores, doc_ids=None):
    out = []
    for i, r in enumerate(scores):
        h = make_hit(rerank=r)
        if doc_ids:
            h.doc_id = doc_ids[i]
        out.append(h)
    return out


def test_abstains_when_distinct_sections_tie_near_the_threshold(monkeypatch):
    """Several different sections scoring alike, just above tau: ambiguous."""
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.50)
    hits = _hits([0.56, 0.555, 0.55, 0.545, 0.54],
                 ["d1", "d2", "d3", "d4", "d5"])
    abstain, reason, _ = A.should_abstain(SearchResult(query="x", hits=hits))
    assert abstain and reason == "ambiguous"


def test_confident_cluster_from_one_section_is_not_ambiguous(monkeypatch):
    """Regression: the original ratio test fired on every good result.

    Temperature-scaled confidences compress relevant hits into a narrow band
    near 0.9, so top/median was always about 1.0. Eight chunks of the single
    section that answers the question is the best possible outcome, and the
    system used to refuse to answer it."""
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.50)
    hits = _hits([0.911, 0.905, 0.904, 0.899, 0.890],
                 ["ecfr:40:262.17"] * 5)
    abstain, reason, score = A.should_abstain(SearchResult(query="x", hits=hits))
    assert not abstain, "a confidently peaked result set must be answered"
    assert score == 0.911


def test_tight_cluster_of_distinct_docs_high_above_threshold_is_answered(monkeypatch):
    """Well above tau, tight clustering means several sections are all
    relevant - a multi-hop answer, not an ambiguous question."""
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.50)
    hits = _hits([0.92, 0.915, 0.91, 0.905, 0.90], ["d1", "d2", "d3", "d4", "d5"])
    abstain, _, _ = A.should_abstain(SearchResult(query="x", hits=hits))
    assert not abstain


def test_answers_when_confident_and_peaked(monkeypatch):
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.30)
    hits = [make_hit(rerank=r) for r in (0.95, 0.30, 0.22, 0.18, 0.10)]
    abstain, reason, score = A.should_abstain(SearchResult(query="x", hits=hits))
    assert not abstain
    assert score == 0.95


def test_prompt_numbers_sources_from_one():
    prompt = A.build_prompt("q", [make_hit(), make_hit()])
    assert "[1] 40 CFR 262.17" in prompt
    assert "[2] 40 CFR 262.17" in prompt


# --- reranker score calibration -------------------------------------------

def test_temperature_scaling_keeps_scores_distinguishable():
    """A plain sigmoid maps the observed logit range to 1.000 and 0.000, which
    makes every score identical and the abstention gate inoperable."""
    from cfr.search import rerank as R

    relevant, irrelevant = 10.0, -11.0
    assert R.sigmoid(relevant) == pytest.approx(1.0, abs=1e-4)
    assert R.sigmoid(irrelevant) == pytest.approx(0.0, abs=1e-4)

    hi, lo = R.to_confidence(relevant), R.to_confidence(irrelevant)
    assert hi - lo > 0.5          # a usable spread to threshold on
    assert 0.0 < lo < 0.5 < hi < 1.0


def test_temperature_scaling_is_monotonic():
    """Calibration must not reorder results - only rescale them."""
    from cfr.search import rerank as R

    logits = [-11.0, -4.0, -0.5, 0.0, 2.5, 8.8, 10.0]
    scaled = [R.to_confidence(x) for x in logits]
    assert scaled == sorted(scaled)


def test_zero_logit_is_the_midpoint():
    from cfr.search import rerank as R

    assert R.to_confidence(0.0) == pytest.approx(0.5)


def test_no_abstention_without_a_calibrated_score(monkeypatch):
    """RRF scores are rank-derived (1/(k+rank)) and have no absolute meaning:
    the top hit is ~0.032 whether it is a perfect match or nonsense. Thresholding
    on them made the system refuse every single query."""
    monkeypatch.setattr("cfr.config.ABSTAIN_THRESHOLD", 0.50)
    hit = make_hit(rerank=None)
    hit.score = 0.032          # a realistic RRF score for a top hit
    abstain, reason, score = A.should_abstain(SearchResult(query="x", hits=[hit]))
    assert not abstain, "must not refuse everything when reranking is disabled"
    assert reason == "no_confidence_signal"
    assert score == 0.032


def test_empty_results_still_abstain_without_rerank():
    abstain, reason, _ = A.should_abstain(SearchResult(query="x", hits=[]))
    assert abstain and reason == "no_results"
