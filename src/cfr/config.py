"""Central configuration. Everything tunable lives here or in the environment."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from .env into the environment.

    Hand-rolled rather than pulling in python-dotenv: it is fifteen lines and
    keeps the dependency list to five packages. Real environment variables
    always win, so `GEMINI_API_KEY=... python -m cfr.cli` still overrides the
    file.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.environ.get("CFR_DATA_DIR", ROOT / "data"))
EVAL_DIR = Path(os.environ.get("CFR_EVAL_DIR", ROOT / "evaldata"))
WEB_DIR = ROOT / "web"
DB_PATH = Path(os.environ.get("CFR_DB", DATA_DIR / "cfr.db"))

# --- corpus -----------------------------------------------------------------
# Start narrow. Each part is a few hundred sections; a whole title is tens of
# thousands and turns the index build into an overnight job.
DEFAULT_PARTS = [
    ("40", "262"),  # Hazardous waste generators
    ("40", "263"),  # Transporters of hazardous waste
    ("29", "1910"), # OSHA general industry standards
    ("21", "312"),  # Investigational new drug application
]
ECFR_API = "https://www.ecfr.gov/api/versioner/v1"
ECFR_DATE = os.environ.get("CFR_ECFR_DATE", "2025-08-01")

# --- chunking ---------------------------------------------------------------
CHUNK_TARGET_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200
CHUNK_MIN_CHARS = 120

# --- models -----------------------------------------------------------------
EMBED_MODEL = os.environ.get("CFR_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("CFR_EMBED_DIM", "384"))
RERANK_MODEL = os.environ.get("CFR_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
# ms-marco cross-encoders emit logits in roughly [-11, +11], and a plain sigmoid
# collapses that to 1.000/0.000 with no usable range to threshold on. Dividing by
# a temperature before the sigmoid restores the spread; 4.0 maps the observed
# relevant/irrelevant clusters to about 0.90 and 0.06.
RERANK_TEMPERATURE = float(os.environ.get("CFR_RERANK_TEMPERATURE", "4.0"))
# Measured, not guessed: ONNX Runtime pads every batch to its longest sequence,
# so small batches waste less compute on padding. On this corpus batch=8 reranks
# 50 passages in ~1.7s against ~2.6s at batch=32. Re-measure with `cfr bench` if
# you change the model or the chunk size.
RERANK_BATCH_SIZE = int(os.environ.get("CFR_RERANK_BATCH_SIZE", "8"))
# Truncating passages before reranking cuts cost close to linearly (400 chars is
# ~4x faster than 1400) but throws away text the model might need. Left off by
# default because it trades quality for latency and that trade needs measuring.
# 0 disables truncation.
RERANK_MAX_CHARS = int(os.environ.get("CFR_RERANK_MAX_CHARS", "0"))

# --- retrieval --------------------------------------------------------------
CANDIDATES_PER_RETRIEVER = 100   # recall stage width
RRF_K = 60                       # standard constant; rarely needs tuning
# How many fused candidates the cross-encoder reads. This is the main latency
# dial: measured p50 is 0.45s at 10, 1.3s at 25, 2.8s at 50 and 6.4s at 100.
# It also caps quality - a relevant section sitting at fused rank 60 can never
# be promoted when only the top 50 are rescored. Run `cfr bench` for the cost
# and `cfr eval` for the benefit before moving it.
RERANK_TOP_N = int(os.environ.get("CFR_RERANK_TOP_N", "50"))
FINAL_TOP_K = 8                  # what reaches the answer stage

# --- abstention -------------------------------------------------------------
# Calibrate with `cfr calibrate`, then set this. Sigmoid of the top rerank
# logit; below this we decline to answer.
# Calibrated, not guessed. `cfr calibrate` swept this against the labelled set:
# 0.20 refuses 100% of out-of-scope queries, falsely abstains on 0% of
# answerable ones, and keeps 94% accuracy over the 85% of queries it answers.
# Re-run `cfr calibrate` after any change to the reranker or the query mix.
ABSTAIN_THRESHOLD = float(os.environ.get("CFR_ABSTAIN_THRESHOLD", "0.20"))
# Ambiguity is only meaningful in the marginal band just above the threshold.
# Above that the result set is confidently peaked, and tight clustering there
# means several chunks of one relevant section scored alike - the system
# working, not failing. Ambiguity additionally requires the cluster to span
# several *different* sections; one section answering well is not ambiguous.
# The ambiguity rule is off by default because it was measured and did not earn
# its place - see `cfr calibrate --ambiguity`. Its premise (several sections
# tied => the question is unclear) is indistinguishable at runtime from several
# sections being *all relevant*, which is what a multi-source answer looks like.
AMBIGUITY_ENABLED = os.environ.get("CFR_AMBIGUITY", "0") not in ("0", "false", "")
AMBIGUITY_BAND = 0.15      # only consider tau .. tau + this
AMBIGUITY_SPREAD = 0.05    # top minus median of the shortlist
AMBIGUITY_MIN_DOCS = 3     # distinct sections in the tie

# --- generation -------------------------------------------------------------
LLM_PROVIDER = os.environ.get("CFR_LLM_PROVIDER", "gemini")  # gemini | groq | none
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Pin an explicit version. Google retires models from the API faster than the
# listing suggests - both gemini-2.0-flash and gemini-2.5-flash return 404
# "no longer available to new users" while still appearing in ListModels, so
# check `generateContent` actually works rather than trusting the list.
# `gemini-flash-latest` also works but moves under you, which is the wrong
# trade when eval numbers need to be reproducible.
# Measured on 6 real queries, cache disabled: flash-lite p50 5.4s with 19/19
# citations surviving verbatim-quote verification; gemini-3.6-flash p50 27.7s
# with 36/36. Both are perfectly groundable, so the 5x latency buys more
# citations per answer, not more trustworthy ones. Lite is the right default
# for a public demo; switch via CFR_GEMINI_MODEL for thoroughness.
#
# The "-latest" alias is deliberate here: Google retires pinned versions
# aggressively (gemini-2.0-flash and gemini-2.5-flash both return 404 "no
# longer available to new users" while still appearing in ListModels), so a
# pinned default rots. Pin explicitly when you need reproducible eval numbers.
GEMINI_MODEL = os.environ.get("CFR_GEMINI_MODEL", "gemini-flash-lite-latest")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("CFR_GROQ_MODEL", "llama-3.3-70b-versatile")

# Hard ceiling on generation calls per UTC day. Retrieval keeps working after
# this is hit; only the answer panel degrades.
DAILY_GENERATION_BUDGET = int(os.environ.get("CFR_DAILY_BUDGET", "500"))
# Cached answers are reused when the query embedding is this close.
SEMANTIC_CACHE_THRESHOLD = 0.97
# Verbatim quotes must match the source at least this well to survive.
QUOTE_MATCH_THRESHOLD = 0.95

DATA_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
