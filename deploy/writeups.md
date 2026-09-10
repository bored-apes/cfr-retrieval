# Write-ups

Paste-ready copy for LinkedIn, the résumé, and interviews. The through-line in
all of them is the same, and it is deliberately not "I built a RAG app": it is
*I measured my own work and published the part where I was wrong.*

---

## LinkedIn post

Current version — leads with the agent finding. The reranker post (kept below)
was the previous one; don't run both, the second reads as a rerun of the first.

> I gave an AI agent exactly one decision to make on its own. It used it to talk
> my safety check out of doing its job.
>
> I built a search engine over US federal regulations. The hard part isn't
> finding the text — it's that you can't search for what you can't name. Ask
> *"how long can I store waste oil"* and you get nothing, because the regulation
> says **accumulate**, not store, and **90 days**, not how long.
>
> So: let an agent rewrite your question into the regulator's vocabulary before
> searching. Everyone builds this. It's the textbook agentic-RAG move, and on
> paper it's the exact fix for the exact problem.
>
> I built it. Then I measured it — 36 rewrites against a graded relevance set.
>
> **It rescued zero queries.**
>
> Not "few". Zero. Because every question the corpus could actually answer was
> already being found. There was nothing to rescue. What there was, was
> something to break:
>
> "Write a Python function that reverses a linked list"
> → the agent rewrote it to **"40 CFR"**
> → confidence went **0.07 → 0.80**
>
> 0.20 is the threshold below which the system refuses to answer. That refusal
> is the most important thing it does — a regulatory search tool that guesses is
> worse than no tool. And one clever rewrite walked a question about linked
> lists straight past it, into hazardous waste law, confidently.
>
> **11% of the rewrites did this.**
>
> So it ships disabled. The code is there, the measurement is in the repo, and a
> flag turns it on for anyone whose corpus has a real vocabulary gap. On mine it
> was a plausible idea with zero upside and a 1-in-9 chance of defeating the
> only guarantee I actually make.
>
> Third feature in this project I've built, measured, and switched off.
>
> The part I'd defend in an interview is what I *didn't* hand to a model. Three
> agents — researcher, writer, auditor — under a supervisor that routes on
> measured state: a calibrated threshold, attempt budgets, and the auditor's
> verdict on **whose fault** a failure was. Fabricated quotes are the writer's
> problem and earn a retry. Sources that can't support an answer are the
> researcher's problem — re-prompting the writer there just buys you a more
> confident fabrication.
>
> Exactly one LLM call in the whole control flow. That was the one that broke it.
>
> Live: https://cfr.bhargavsuhagiya.com
> Code + full measurement: https://github.com/bored-apes/cfr-retrieval
>
> #AIEngineering #AgenticAI #LLM #RAG #MachineLearning

**Notes on posting.** The first two lines are the whole game — everything after
"see more" is read by people who already decided to care. Don't soften "zero" and
don't move the linked-list example lower; it is the only part a non-technical
reader can repeat to someone else. Attach `social/cfr-agent-finding.png`.

If you want it shorter, cut from "The part I'd defend" to the end — the post
survives intact, it just stops being about architecture.

### X / Twitter version

> I gave an AI agent one decision to make on its own.
>
> It used it to defeat my safety check.
>
> "Write a Python function that reverses a linked list"
> → agent rewrote it to "40 CFR"
> → confidence 0.07 → 0.80
>
> (0.20 is where the system is supposed to refuse.)
>
> 11% of rewrites did this. It rescued 0 queries.
>
> Ships disabled. 🧵

---

## LinkedIn post — previous version (reranker)

> I spent two weeks building a search engine for US federal regulations. The
> most useful thing I did was prove one of my own design decisions was wrong.
>
> **The problem.** Say you run a small auto shop with waste oil, and you need to
> know how long you can legally keep it on site. That answer is public, free and
> online. But search "how long can I store waste" and you get nothing — because
> the regulation says *accumulate*, not *store*, and *90 days*, not *how long*.
> You can't find it unless you already know the words.
>
> **What I built.** You ask in plain English. It searches two ways at once — one
> that matches exact words, one that matches meaning, because each fails where
> the other works — then shows you the answer with the exact sentence
> highlighted in the real regulation.
>
> **The part that mattered.** Almost every system like this adds a component
> called a reranker. It's the standard best practice, so I added it. Then I
> built a 1,237-judgement relevance set to check whether it actually helped.
>
> It didn't. It made the system 56× slower for an improvement statistically
> indistinguishable from zero.
>
> But the same test showed it was doing something I hadn't designed it for: it
> was the only component producing a calibrated confidence score — the only
> reason the system can tell when it doesn't know. So I kept it, for a
> completely different reason than I added it.
>
> That's the whole point. Without measurement I'd have shipped it and written a
> confident README about how essential it was. Instead the README documents
> where I was wrong, which queries it still fails, and why.
>
> It now runs entirely in your browser — no server, nothing to pay for.
>
> Live: https://cfr.bhargavsuhagiya.com
> Code and full method: https://github.com/bored-apes/cfr-retrieval
>
> #MachineLearning #InformationRetrieval #AIEngineering

**Notes on posting.** Lead with the problem, not the link — people who see a URL
first form an opinion in ten seconds. The reranker paragraph is the hook: it is
a story about judgement, which travels much further than a stack list. If you
want a shorter version, cut everything between "What I built" and "The part that
mattered" — the story survives on its own.

---

## Résumé

Slots into **FOUNDER EXPERIENCE — AI & ML**, under Pluto. Matches the existing
density and the habit of stating limitations plainly.

**CFR Retrieval — hybrid search over US federal regulations** · *live, runs
client-side* — 2026

- Built a two-stage retrieval system over **338 CFR sections / 5,244 chunks**:
  BM25 and dense vectors fused by reciprocal rank fusion, then a cross-encoder
  rerank, with character offsets preserved end to end so every citation
  highlights the exact sentence in the source regulation.
- Built a **1,237-judgement graded relevance set** using TREC-style pooling
  across five configurations (33% of candidates surfaced by only one config),
  judged blind to retrieval provenance with per-row provenance recorded;
  published **nDCG@10, recall@100 and per-stage latency with 95% bootstrap
  confidence intervals** for each.
- **Measurement overturned the architecture**: the cross-encoder cost
  **+1,770 ms p50 for +0.016 nDCG@10** — a paired interval spanning zero.
  Retained it on the evidence that fusion scores are rank-derived (0.032
  in-scope vs 0.025 out-of-scope) and it is the only calibrated signal the
  abstention gate can threshold on.
- Calibrated abstention to **decline 100% of out-of-scope queries at 0% false
  refusals**; citations verified mechanically by matching each verbatim quote
  back into its cited section, dropping and counting unmatched ones.
- Re-architected to run **entirely client-side** when Docker hosting moved
  behind a paywall — 5.4 MB payload, int8 vectors at **98% top-10 agreement**
  with float32, models via transformers.js; hosting cost is zero and scales
  with visitors rather than servers.

**One-line version**, if space is tight:

> Hybrid BM25 + dense retrieval over federal regulations with verified span
> citations and calibrated abstention; a 1,237-judgement graded relevance set
> showed the cross-encoder cost 56× latency for statistically zero ranking gain —
> kept only because it is the sole calibrated signal the abstention gate can use.

**Skills line additions** (AI & Machine Learning): *information retrieval — BM25,
dense retrieval, reciprocal rank fusion, cross-encoder reranking; IR evaluation
— TREC pooling, graded nDCG, bootstrap confidence intervals; ONNX Runtime,
transformers.js.*

---

## Interview answers

**"Walk me through a technical decision you got wrong."**
The reranker. Standard practice, added it on faith, measured it, found +0.016
nDCG for 56× the latency with a CI spanning zero. The interesting part is what I
did next: rather than ripping it out, I checked *why* removing it broke things,
and found it was the only calibrated score in the pipeline. Fusion scores come
from rank position, so a perfect match scores 0.032 and a nonsense query scores
0.025 — no threshold separates those. The component was load-bearing for a
reason I hadn't designed.

**"How do you know your retrieval is good?"**
1,237 graded judgements over 60 queries covering six query types, pooled across
five configurations so the answer key isn't a mirror of any one system. Every
number carries a bootstrap CI, and differences inside the interval get reported
as ties.

**"What would you do differently?"**
Three things, in order. Triple the labelled set — at n=51 the intervals are
±0.05, so a real 3-point gain is invisible. Replace the machine-generated
judgements with human ones and measure the disagreement. And fix the failure I
documented but didn't solve: for two of the five worst queries, a section that
*references* the target outranks the target itself.

**"Why should I trust the numbers?"**
You shouldn't, entirely — and the README says so. The judgements are
machine-generated and labelled as such in the data. Recall@100 reads 1.000
largely because the pool was drawn from the same systems being measured, which
is close to circular, and I say that next to the number. There's a documented
query where the correct section was never retrieved by any configuration, so it
can't appear in the answer key at all.
