# Deploying

Target: **Hugging Face Spaces**, free CPU tier. It is the right host for this
app — 2 vCPU and 16 GB RAM at no cost and no card required, which matters
because the vector matrix and two ONNX models want roughly 1 GB resident. Render's
free tier (512 MB) will OOM; Fly now wants a card on file even inside the free
allowance.

The image is self-contained: prebuilt index and both models baked in, so the
container needs no network at runtime and the first visitor is not stuck behind
a model download.

---

## Before you start

**Rotate the Gemini key that is currently in `.env`.** It was pasted into a chat
transcript, so treat it as public. Generate a fresh one at
<https://aistudio.google.com/apikey> and use that below. Never commit either.

---

## 1. Create the Space

At <https://huggingface.co/new-space>:

| Field | Value |
|---|---|
| SDK | **Docker** → Blank |
| Hardware | **CPU basic** (free) |
| Visibility | **Public** |

You get `https://huggingface.co/spaces/<you>/<space-name>`.

## 2. Push the code and the index

`data/` is gitignored, but the Space needs the 112 MB database. This pushes a
`deploy` branch that includes it, without polluting your GitHub history.

```bash
brew install git-lfs                 # once
git remote add space https://huggingface.co/spaces/<you>/<space-name>

./deploy/make-hf-branch.sh           # assembles the deploy branch
git push space deploy:main --force
git checkout main
```

`main` stays clean for GitHub — no YAML frontmatter at the top of the README and
no 112 MB binary, so the repo is ~450 KB. The script adds both onto a `deploy`
branch that only Hugging Face ever sees.

Git will ask for your Hugging Face username and an access token as the password
— create one at <https://huggingface.co/settings/tokens> with **write** scope.

The Space starts building as soon as the push lands. First build takes 5–10
minutes, most of it installing dependencies and baking the models in.

## 3. Set the secrets

In the Space: **Settings → Variables and secrets**.

| Name | Kind | Value |
|---|---|---|
| `GEMINI_API_KEY` | **Secret** | your fresh key |
| `CFR_GEMINI_MODEL` | Variable | `gemini-flash-lite-latest` |
| `CFR_DAILY_BUDGET` | Variable | `200` |
| `CFR_ABSTAIN_THRESHOLD` | Variable | `0.20` |

`GEMINI_API_KEY` must be a **Secret**, not a Variable — Variables are visible to
anyone who can view the Space.

The Space restarts itself after you save. Done.

---

## What visitors get, and what it costs you

Retrieval — ranked sections, confidence scores, `bm25 #n / dense #n` provenance,
span-highlighted citations — is **free and unmetered**. It is pure CPU on
hardware Hugging Face is already giving you.

Only written answers touch the Gemini quota, and they are bounded three ways:

- **200 generations/day**, enforced in SQLite so it survives restarts.
- **Per-IP rate limit**, roughly 3 answer requests/minute sustained.
- **Answer cache**, exact and semantic (cosine > 0.97) — on a public demo most
  traffic is the same handful of questions, so the cache absorbs most of it.

When the cap is reached the app enters its `budget_exhausted` state: retrieval
keeps working, and the answer panel explains that generation resumes at 00:00
UTC. That is a designed state, not a failure.

## Verifying the deploy

```bash
curl https://<you>-<space-name>.hf.space/api/health
curl https://<you>-<space-name>.hf.space/api/stats
```

`/api/stats` reports document count, generation status and remaining budget.
`/api/health` is what the container healthcheck uses.

## Notes

- **Spaces sleep after ~48 h idle** on the free tier and wake on the next
  request. The index is in the image, so waking is seconds, not a rebuild.
- **One worker, deliberately.** The vector matrix and rate-limit buckets are
  per-process; a second worker doubles memory and silently doubles the effective
  rate limit.
- **Updating:** commit to `main`, then
  `git checkout deploy && git merge main && git push space deploy:main`.
- **The labelling UI ships at `/label`.** It writes to the container's ephemeral
  database, so judgements made on the deployed instance are lost on restart.
  Judge locally and commit `evaldata/qrels.jsonl` instead.
