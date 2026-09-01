# CFR Retrieval

Hybrid retrieval over the US Code of Federal Regulations, with a measured
ablation and mechanically verified citations.

Regulations are public, free, and almost unsearchable. The official interface is
a keyword box, so a shop owner asking *"how long can I keep waste oil on site?"*
finds nothing — the regulation says **accumulation**, not *store*, and **90
days**, not *how long*. That vocabulary gap is the problem this system exists to
close.

**This is not a chat-with-your-PDF demo.** The point is not that it answers
questions; the point is that every architectural decision in it has a number
attached, produced by a hand-built relevance set.

---

## Quickstart

```bash
make install
make build      # fetch ~340 CFR sections, chunk 4 ways, embed 2 of them
make serve      # http://127.0.0.1:8000
```

`make build` takes 30-45 minutes on CPU, almost all of it embedding (it commits
per batch and resumes, so an interrupt costs one batch, not the run). Nothing
here needs a GPU, an API key, or a paid service.

Generation is optional. Without `GEMINI_API_KEY` or `GROQ_API_KEY` the app runs
retrieval-only, which is a supported state rather than a broken one — see
[Degradation](#degradation).

---

## The architecture

A funnel that runs wide, then narrow.

```
                    ┌──────────────┐  top 100
              ┌────►│  BM25 (FTS5) ├──────────┐
  query       │     └──────────────┘          ▼
  + filters ──┤                          ┌─────────┐  100   ┌───────────────┐
              │     ┌──────────────┐     │   RRF   ├───────►│ cross-encoder │
              └────►│ dense vectors├────►│ fusion  │        │    rerank     │
                    └──────────────┘     └─────────┘        └───────┬───────┘
                          top 100                                   │ top 8
                                                                    ▼
                                                            ┌───────────────┐
                                              score < τ ◄───┤  confidence   ├───► score ≥ τ
                                                 abstain    │     gate      │     generate
                                                            └───────────────┘
```

**Stage one optimises recall.** Anything it misses is unrecoverable downstream,
so it retrieves 100 candidates from each retriever rather than 10.

- **BM25** catches what embeddings are worst at: exact section numbers, defined
  terms, acronyms, chemical names.
- **Dense vectors** catch what BM25 is worst at: paraphrase. *"how long can I
  keep it"* lands next to *"accumulation time limits"*.
- **RRF** merges them. BM25 scores are unbounded and cosine sits in `[-1, 1]`;
  blending those directly means normalising two distributions that shift per
  query. RRF discards the scores and fuses on rank position instead, which
  removes the problem and is very hard to beat. Six lines, one constant.

**Stage two optimises precision.** The bi-encoder that built the index encoded
each passage *before anyone knew what would be asked* — that is what makes it
indexable and what caps its accuracy. A cross-encoder reads the query and one
passage **together** in a single forward pass, so attention runs across both.
Far more accurate, and linear in candidates, which is exactly why it only ever
sees a shortlist.

Every stage is a flag on `RetrievalConfig`. If a setting cannot be varied there,
it cannot be measured.

---

## The ablation

```bash
make pool && make serve   # gather candidates, judge them at /label
make eval                 # produce the table below
```

**51 judged queries** (plus 9 out-of-scope), 1,237 pooled judgements, 1,000
bootstrap resamples.

| Configuration | Recall@100 | nDCG@10 | MRR@10 | p50 | p95 |
|---|---|---|---|---|---|
| bm25 only | 0.915 ±0.046 | 0.575 ±0.067 | 0.850 ±0.084 | 34 ms | 72 ms |
| dense only | 0.947 ±0.025 | 0.658 ±0.051 | 0.951 ±0.049 | 1 ms | 2 ms |
| hybrid (rrf) | 1.000 ±0.000 | 0.701 ±0.053 | 0.967 ±0.042 | 32 ms | 50 ms |
| hybrid + rerank | 1.000 ±0.000 | 0.717 ±0.051 | 0.971 ±0.029 | 1,802 ms | 2,503 ms |
| + contextual chunks | 0.994 ±0.008 | 0.717 ±0.046 | 0.980 ±0.029 | 1,898 ms | 2,409 ms |

Each row isolates one change from the row above, so the **incremental**
comparison is the one that decides whether a stage is worth keeping:

| Configuration | ΔnDCG@10 vs row above | 95% CI | Δp50 | Verdict |
|---|---|---|---|---|
| dense only | **+0.083** | [+0.012, +0.155] | −34 ms | **earns it** |
| hybrid (rrf) | +0.043 | [−0.006, +0.098] | +31 ms | no measurable gain |
| hybrid + rerank | +0.016 | [−0.028, +0.061] | **+1,770 ms** | no measurable gain |
| + contextual chunks | −0.000 | [−0.021, +0.018] | +96 ms | no measurable gain |

### What the numbers actually say

**The cross-encoder did not earn its latency as a ranker.** It bought +0.016
nDCG@10 for +1,770 ms — a 56× latency increase for a difference whose confidence
interval comfortably spans zero. This is the opposite of what the architecture
chapter above predicts, and it is the single most useful thing measurement
produced. Had I not built the eval set, the README would confidently claim the
reranker was carrying the system.

**But the reranker is still load-bearing — for abstention, not ranking.** RRF
scores come from rank position, `1/(k + rank)`, so the top hit scores ≈0.032
whether it is a perfect match or a question about pizza dough:

| | in-scope query | out-of-scope query |
|---|---|---|
| RRF score (no rerank) | 0.0320 | 0.0249 |
| cross-encoder confidence | **0.911** | **0.062** |

There is no threshold that separates 0.0320 from 0.0249. Turning the reranker
off does not make the system slightly worse — it removes the only calibrated
signal in the pipeline, and `should_abstain` degrades explicitly rather than
refusing every query. **The reranker earns its place for a completely different
reason than the one it was added for.**

**Contextual chunking bought nothing** (−0.000 nDCG, +96 ms). The prefix used
here is deterministic — the section's place in the hierarchy — not an
LLM-written summary of the parent context. That is the version worth testing
next; this result says only that the cheap version does not help.

### Read this table with two caveats

- **Recall@100 = 1.000 is largely an artifact.** The pool was built from these
  same five configurations at depth 20, so a document can only be marked
  relevant if one of them already retrieved it. Measuring their recall against
  that pool is close to circular. The number is not meaningless — it shows the
  fusion stage surfaces everything the shortlist stages found — but it is not
  evidence that nothing was missed. Query `p11` below is a concrete case of
  something missed by all five.
- **n = 51 is small.** "No measurable gain" means *not detectable at this sample
  size*, not *zero*. The intervals are ±0.05, so a real +0.03 improvement would
  be invisible here. Tripling the labelled set is the highest-value next step.

### Abstention, calibrated

```bash
make calibrate
```

| τ | coverage | accuracy | false abstain | correct refusals |
|---|---|---|---|---|
| 0.00 | 100% | 94% | 0% | **0%** |
| 0.10 | 88% | 94% | 0% | 78% |
| **0.20** | **85%** | **94%** | **0%** | **100%** |
| 0.50 | 80% | 96% | 6% | 100% |
| 0.85 | 18% | 100% | 78% | 100% |

τ = 0.20 is the shipped default: it declines **100% of out-of-scope queries**
while falsely refusing **0%** of answerable ones.

Writing this sweep also caught a bug in the recommender itself. It originally
picked the *lowest* threshold meeting the accuracy target — which is τ = 0.00,
a setting that hits 94% accuracy while refusing nothing at all, defeating the
entire gate. Accuracy-on-answered is only half the objective; the recommender
now maximises correct refusals subject to a false-abstention ceiling.

### Where it fails

`cfr eval --show-failures 5` prints the worst queries under the best config:

| Query | nDCG@10 | Diagnosis |
|---|---|---|
| `x01` "What does 40 CFR 262.17 require?" | 0.294 | Returns § 262.10, which *describes* 262.17, above § 262.17 itself. Classic "about the thing" vs "the thing". |
| `p11` "Does my employer have to provide an eyewash station?" | 0.323 | § 1910.151(c) is the answer and **no configuration surfaced it at all** — it is not even in the pool. A true recall miss the metrics cannot see. |
| `p13` "How often do I need to inspect emergency equipment?" | 0.331 | Three sections are equally correct (§ 262.253, § 1910.164, § 1910.165); it finds one and buries the others. |
| `m03` training comparison | 0.428 | Multi-hop: retrieves the hazardous-waste half, misses the respirator half. |
| `x02` "1910.1200 hazard communication" | 0.437 | Ranks sections that *cite* 1910.1200 above 1910.1200 itself. |

Two of the five are the same failure mode — **a section that references the
target outranks the target** — which is a concrete, actionable next fix.

### Who judged this answer key

**The judgements currently in `data/cfr.db` are machine-generated**, produced by
an LLM reading each section's text. Every row records its provenance:

```sql
SELECT judged_by, COUNT(*) FROM qrels GROUP BY judged_by;
```

This is a real and increasingly common technique, and it was done carefully —
the labelling tool (`scripts/judge.py dump`) deliberately **hides which
configuration surfaced each candidate and at what rank**, so the labels cannot
be biased toward any system in the ablation. Out-of-scope queries were graded 0
by construction rather than by inspection.

It is still not the same as your own judgement, and you should not present it as
such. Two known weaknesses:

- **LLM judges correlate with human relevance labels but not perfectly**, and
  they tend to be more generous with partial matches (grade 1–2) than domain
  experts are.
- **A pooled answer key can only contain what pooling surfaced.** For query
  `p11` ("does my employer have to provide an eyewash station?") the section that
  actually answers it — § 1910.151(c) — never appeared in any configuration's
  top 20, so it is not in the pool and cannot be marked relevant. Recall@100
  therefore flatters every configuration equally on that query. This is the
  standard pooling caveat and it is why deepening the pool matters.

**The right next step is to spot-check.** Open `/label` and re-judge 50 random
pairs yourself; the UI writes `judged_by = 'human'`, and human labels are never
overwritten by a later machine pass. If your labels agree with the machine ones
on roughly 80%+ of pairs, the table below is trustworthy. If they do not, the
disagreement itself is the more interesting result, and worth writing up.

### Pooling, not inspection

`cfr pool` runs **every** ablation config, takes the top 20 from each, and asks
you to judge the union. Labelling only what your current system returns bakes
its biases into the ground truth and guarantees it scores well — the standard
TREC pooling method is the fix, and it gives configurations you have not built
yet a fair shot at documents your current one never surfaces.

Grades are 0–3, not binary, because nDCG needs the gradations to mean anything:

```
3  directly answers the question
2  partly answers it
1  related background
0  irrelevant
```

---

## Verified citations

A model that emits `[3]` has produced a token, not a promise. So it is also
required to quote the supporting sentence **verbatim**, and that quote is matched
back into the chunk it claims to come from.

```
document ──► chunk (char 4102–4680) ──► answer cites [3] + quote ──► string match
                                                                        │
                                              match ──► offsets ──► highlight span
                                           no match ──► citation dropped and counted
```

Matching normalises whitespace, case, and smart punctuation, then falls back to a
fuzzy match at ≥ 0.95 similarity — models routinely alter a hyphen or collapse a
line break, and flagging that as a hallucination would be wrong.

The offsets that make this exact rather than a text search are recorded at
ingest. **`cfr chunk` re-slices every chunk out of its document and fails the
build on any mismatch**, because silent offset drift produces citations that
highlight the wrong sentence — worse than no highlighting at all.

`citations_dropped` is a real operational metric. Publish it.

---

## Abstention

Three independent signals, any one of which stops the answer:

1. **Top rerank score below τ** — nothing relevant exists.
2. **Flat score distribution** — nothing stands out, which usually means the
   question is ambiguous rather than unanswerable. The UI asks rather than refuses.
3. **Empty result set.**

τ is not guessed:

```bash
make calibrate
```

This sweeps the threshold across the labelled set and prints coverage against
accuracy-on-answered. Pick the τ that hits your target accuracy and report the
coverage you gave up to get there. That trade-off *is* the tuning.

---

## Degradation

Retrieval is the part of this system that costs nothing to run, so it never goes
away. Only the written answer degrades:

| State | What happens |
|---|---|
| No API key | `retrieval_only` — ranked sections, no prose |
| Daily budget spent | `budget_exhausted` — ranked sections, explains when it resets |
| Provider errors | `generation_failed` — ranked sections, names the failure |
| Low confidence | `abstained` — shows nearest matches and the score that fell short |

Each of these is a designed state with its own copy, not an exception leaking to
the user.

---

## Running it free

Everything expensive happens once, offline, on your own machine.

| Component | Choice | Note |
|---|---|---|
| Index build | local CPU | one-time, ~35 min for this corpus |
| Storage | one SQLite file | documents + FTS5 + vectors together |
| Lexical | SQLite FTS5 | free, and already beside the metadata |
| Dense | numpy dot product | exhaustive, exact, ~5 ms here |
| Reranker | ONNX cross-encoder | the one component with real per-query cost |
| Generation | Gemini / Groq free tier | behind a hard daily cap |

**Scaling notes.** The vector matrix is `n × 384 × 4` bytes — about 300 MB at
200k chunks. Quantise to int8 (~75 MB) or binary with a float rescoring pass
(~10 MB) when it stops fitting. Swap the exhaustive scan for HNSW only when the
matrix no longer fits in memory; doing it earlier trades exact results for
nothing.

**The reranker is the interesting deployment problem** — it is real compute on
every query with no comfortable free tier. The cheapest answer is to push it to
the client: the same ONNX cross-encoder runs in the visitor's browser via
transformers.js on WebGPU, so it costs nothing and scales with traffic. That
trades a first-load model download for zero marginal cost.

---

## Layout

```
src/cfr/
  config.py          every tunable, in one place
  db.py              schema: documents, chunks, FTS5, vectors, qrels
  ingest.py          eCFR XML -> one document per section
  chunk.py           4 strategies + the offset invariant check
  embed.py           ONNX embeddings, no torch
  answer.py          generation, abstention, citation verification
  ratelimit.py       per-client token bucket
  api.py             FastAPI: /api/search, /api/ask, /api/judge
  search/
    lexical.py       BM25 over FTS5, with query sanitisation
    dense.py         cosine over the vector matrix
    fusion.py        reciprocal rank fusion
    rerank.py        cross-encoder
    pipeline.py      the funnel; every knob is a config field
  eval/
    metrics.py       nDCG, recall, MRR, bootstrap CIs
    pool.py          TREC-style pooling for labelling
    run.py           the ablation + threshold calibration
web/                 search UI, source view with span highlighting, labelling UI
tests/               68 tests; offsets, fusion, metrics, sanitisation, citations,
                     rate limiting, score calibration, cache staleness
```

---

## Limitations

- **The corpus is four CFR parts**, not the whole CFR (~340 sections of roughly
  200,000). Scaling the ingest is a loop; scaling the *relevance set* is not,
  which is the real constraint.
- **Contextual chunking uses a deterministic prefix** — the section's place in
  the hierarchy — not an LLM-written summary of the parent context. The hook is
  in `chunk._context_for`. The LLM version usually does better; measure it
  rather than assuming.
- **The reranker is `ms-marco-MiniLM-L-6-v2`**, trained on web search, not legal
  text. A domain-tuned reranker would likely do better and is untested here.
- **Rate limiting is in-process**, so it is per-worker. Running more than one
  worker silently multiplies the effective limit.
- **This is not legal advice.** It retrieves and quotes regulatory text. The
  abstention gate exists partly because confidently wrong compliance information
  is worse than none.
- **The answer key is machine-generated** (see [Who judged this answer
  key](#who-judged-this-answer-key)). Spot-check it before quoting the numbers.
- **51 judged queries is a small sample.** Confidence intervals are ±0.05 on
  nDCG@10, so real improvements smaller than that are invisible. Tripling the
  labelled set is the highest-value next step.
- **Recall@100 is measured against a pool built from the same configurations**,
  which makes it close to circular. Query `p11` is a documented case of a
  relevant section that no configuration surfaced, so it cannot appear in the
  answer key at all.

---

## Deploying

```bash
make build                      # produce data/cfr.db locally, once
docker build -t cfr-retrieval . # index and models baked into the image
fly launch --no-deploy && fly deploy
```

The image carries the prebuilt SQLite file and both ONNX models, so the
container needs no network at runtime and a cold start does not stall the first
visitor behind a model download. `fly.toml` scales to zero, which keeps it
inside the free allowance.

Run **one** worker. The vector matrix and the rate-limit buckets are both
per-process: a second worker doubles the memory and silently doubles the
effective rate limit.
