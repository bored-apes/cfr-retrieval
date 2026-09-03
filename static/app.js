/**
 * The whole retrieval pipeline, client-side.
 *
 * BM25 is rebuilt in JS from the shipped chunk text; dense vectors ship
 * int8-quantised and are scored with an integer dot product; both models come
 * from the Hugging Face CDN via transformers.js and are cached by the browser.
 * There is no server, which is why this can be hosted free forever.
 *
 * The port has to agree with the Python index on two things or dense retrieval
 * silently degrades: CLS pooling, and NO instruction prefix on queries. A canary
 * vector shipped in meta.json checks both at boot.
 */
import {
  pipeline, AutoTokenizer, AutoModelForSequenceClassification, env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

env.allowLocalModels = false;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const S = {
  meta: null, chunks: null, docs: null, vecs: null,
  bm25: null, embedder: null, rerankTok: null, rerankModel: null,
  device: "wasm", rerankDepth: 20, lastHits: [],
};

// ---------------------------------------------------------------- boot ----

function step(id, state, label) {
  const el = $(id);
  if (!el) return;
  el.className = "boot-step " + state;
  if (label) el.lastElementChild.textContent = label;
}
function progress(pct) { $("bootbar").style.width = pct + "%"; }

async function fetchGz(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  // Hosts disagree about .gz: some send it raw, others set Content-Encoding
  // and the browser has already decompressed it. Sniff the gzip magic bytes
  // instead of trusting either behaviour - guessing wrong breaks boot.
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (head[0] === 0x1f && head[1] === 0x8b) {
    const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
    return JSON.parse(await new Response(stream).text());
  }
  return JSON.parse(new TextDecoder().decode(buf));
}

async function boot() {
  try {
    step("s-data", "active");
    S.meta = await (await fetch("./data/meta.json")).json();
    const [chunks, docs, vbuf] = await Promise.all([
      fetchGz("./data/chunks.json.gz"),
      fetchGz("./data/docs.json.gz"),
      fetch("./data/vectors.i8.bin").then((r) => r.arrayBuffer()),
    ]);
    S.chunks = chunks;
    S.docs = docs;
    S.vecs = new Int8Array(vbuf);
    if (S.vecs.length !== S.chunks.length * S.meta.dim) {
      throw new Error("vector/chunk count mismatch");
    }
    step("s-data", "done", `Corpus loaded — ${S.meta.documents} sections, ${S.meta.chunks} chunks`);
    progress(25);

    step("s-bm25", "active");
    await new Promise((r) => setTimeout(r, 0));   // let the paint land
    S.bm25 = buildBM25(S.chunks);
    step("s-bm25", "done", `BM25 index — ${S.bm25.vocab.size.toLocaleString()} terms`);
    progress(45);

    step("s-embed", "active");
    S.embedder = await pipeline("feature-extraction", S.meta.embed_model, {
      dtype: "q8",
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          progress(45 + Math.round((p.loaded / p.total) * 25));
        }
      },
    });
    const ok = await verifyCanary();
    step("s-embed", ok ? "done" : "err",
      ok ? "Embedding model ready — matches the index"
         : "Embedding model MISMATCHES the index — dense results unreliable");
    progress(70);

    step("s-rerank", "active");
    S.rerankTok = await AutoTokenizer.from_pretrained(S.meta.rerank_model);
    // The cross-encoder dominates latency: 50 passages take ~21s on WASM but a
    // fraction of that on WebGPU, at identical quality. Try the GPU, fall back
    // silently, and tell the user which one they got - it changes the depth we
    // can afford.
    // WebGPU is opt-in via ?device=webgpu. Detection is not enough: a
    // software-emulated adapter reports as available and then reranks 50
    // passages in >90s, against ~21s on WASM. Real discrete GPUs are much
    // faster, so the escape hatch stays.
    const wantGpu = new URLSearchParams(location.search).get("device") === "webgpu"
      && "gpu" in navigator;
    const opts = (device) => ({
      dtype: "q8", device,
      progress_callback: (p) => {
        if (p.status === "progress" && p.total) {
          progress(70 + Math.round((p.loaded / p.total) * 30));
        }
      },
    });
    try {
      if (!wantGpu) throw new Error("no webgpu");
      S.rerankModel = await AutoModelForSequenceClassification.from_pretrained(
        S.meta.rerank_model, opts("webgpu"));
      S.device = "webgpu";
    } catch (e) {
      S.rerankModel = await AutoModelForSequenceClassification.from_pretrained(
        S.meta.rerank_model, opts("wasm"));
      S.device = "wasm";
    }
    // Depth 12, not the server's 50. Each passage costs ~400ms on WASM, and the
    // ablation found the reranker's *ranking* contribution was inside the noise
    // (+0.016 nDCG@10, CI spanning zero). What it actually provides is the
    // calibrated score the abstention gate needs - and that only requires
    // scoring enough candidates to trust the top one. Paying 20s to reorder 50
    // buys nothing measurable; 12 keeps the gate working at ~4s.
    S.rerankDepth = S.device === "webgpu" ? S.meta.rerank_top_n : 12;
    step("s-rerank", "done",
      `Cross-encoder ready — ${S.device.toUpperCase()}, reranking top ${S.rerankDepth}`);
    progress(100);

    $("corpusline").textContent =
      `${S.meta.documents} sections · ${S.meta.chunks} chunks · runs in your browser`;
    setTimeout(() => { $("boot").hidden = true; }, 600);
    for (const id of ["form", "controls", "keyrow", "examples"]) $(id).hidden = false;
    $("q").focus();
  } catch (err) {
    $("boot").innerHTML =
      `<h3>Could not start</h3><p class="err">${esc(err.message)}</p>
       <p class="note">Model files come from the Hugging Face CDN; a blocked
       network or an unsupported browser will stop this. Needs WebAssembly and
       DecompressionStream (Chrome/Edge 103+, Firefox 113+, Safari 16.4+).</p>`;
  }
}

/** Catch a pooling or prefix mismatch against the Python index at boot. */
async function verifyCanary() {
  if (!S.meta.canary) return true;
  const v = await embedQuery(S.meta.canary.text);
  const ref = S.meta.canary.vector;
  let dot = 0;
  for (let i = 0; i < ref.length; i++) dot += v[i] * ref[i];
  console.info(`[canary] cosine vs Python embedding: ${dot.toFixed(6)}`);
  return dot > 0.99;
}

// ------------------------------------------------------------- lexical ----

// Mirrors src/cfr/search/lexical.py: section numbers such as 262.17 must stay
// one token, and a handful of stopwords are dropped so an OR query is not
// dominated by words that appear in every regulation.
const TOKEN_RE = /[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)+|\d+/g;
const STOP = new Set(`a an the and or of to in for on at by is are was were be been being as
that this these those it its from with which what when where who whom how do does did can
could shall should may might must will would i you we they my your our their there here if
then than so such not no nor but`.split(/\s+/));

const tokenize = (text) => (text.toLowerCase().match(TOKEN_RE) || []);

function buildBM25(chunks) {
  const df = new Map();
  const postings = new Map();
  const lens = new Float32Array(chunks.length);
  let total = 0;

  for (let i = 0; i < chunks.length; i++) {
    const toks = tokenize(chunks[i].t);
    lens[i] = toks.length;
    total += toks.length;
    const tf = new Map();
    for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
    for (const [t, f] of tf) {
      df.set(t, (df.get(t) || 0) + 1);
      if (!postings.has(t)) postings.set(t, []);
      postings.get(t).push(i, f);   // flat pairs keep this compact
    }
  }
  return { vocab: df, postings, lens, avgdl: total / chunks.length, N: chunks.length };
}

function bm25Search(query, limit) {
  const { vocab, postings, lens, avgdl, N } = S.bm25;
  const k1 = 1.2, b = 0.75;
  let terms = tokenize(query).filter((t) => t.length > 1 && !STOP.has(t));
  if (!terms.length) terms = tokenize(query).filter((t) => t.length > 1);
  if (!terms.length) return [];

  const scores = new Map();
  for (const t of new Set(terms)) {
    const post = postings.get(t);
    if (!post) continue;
    const idf = Math.log(1 + (N - vocab.get(t) + 0.5) / (vocab.get(t) + 0.5));
    for (let p = 0; p < post.length; p += 2) {
      const i = post[p], f = post[p + 1];
      const norm = f + k1 * (1 - b + (b * lens[i]) / avgdl);
      scores.set(i, (scores.get(i) || 0) + (idf * f * (k1 + 1)) / norm);
    }
  }
  return [...scores.entries()].sort((x, y) => y[1] - x[1]).slice(0, limit);
}

// --------------------------------------------------------------- dense ----

async function embedQuery(text) {
  // No instruction prefix - the index was built without one. See meta.query_prefix.
  const out = await S.embedder(S.meta.query_prefix + text, {
    pooling: "cls", normalize: true,
  });
  return out.data;
}

function denseSearch(qvec, limit) {
  const dim = S.meta.dim;
  const n = S.chunks.length;
  const v = S.vecs;
  // The int8 rows share one global scale, so a raw integer dot product ranks
  // identically to the dequantised cosine - no need to divide anything.
  const scored = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const off = i * dim;
    for (let j = 0; j < dim; j++) s += qvec[j] * v[off + j];
    scored[i] = [i, s];
  }
  scored.sort((a, b) => b[1] - a[1]);
  return scored.slice(0, limit);
}

// ---------------------------------------------------------------- fuse ----

function rrf(rankings, k) {
  const scores = new Map();
  for (const ranking of rankings) {
    for (let rank = 0; rank < ranking.length; rank++) {
      const id = ranking[rank][0];
      scores.set(id, (scores.get(id) || 0) + 1 / (k + rank + 1));
    }
  }
  return [...scores.entries()].sort((a, b) => b[1] - a[1]);
}

// -------------------------------------------------------------- rerank ----

const sigmoid = (x) => (x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)));
const toConfidence = (logit) => sigmoid(logit / S.meta.rerank_temperature);

const RERANK_MAX_TOKENS = 192;

async function rerankPairs(query, passages) {
  const scores = [];
  // Every batch is padded to the SAME length on purpose. With `padding: true`
  // each batch is padded to its own longest sequence, so every batch is a new
  // tensor shape - and WebGPU recompiles its shaders per shape, which made the
  // GPU path (48s) slower than WASM (21s). A fixed shape compiles once.
  const BATCH = S.device === "webgpu" ? 16 : 8;
  for (let i = 0; i < passages.length; i += BATCH) {
    const batch = passages.slice(i, i + BATCH);
    // Pad the last batch out to full width too, so it reuses the same shader.
    const padded = batch.length === BATCH
      ? batch : batch.concat(Array(BATCH - batch.length).fill(""));
    const inputs = await S.rerankTok(Array(BATCH).fill(query), {
      text_pair: padded,
      padding: "max_length", max_length: RERANK_MAX_TOKENS, truncation: true,
    });
    const { logits } = await S.rerankModel(inputs);
    const data = logits.data;
    for (let j = 0; j < batch.length; j++) scores.push(Number(data[j]));
    if (i + BATCH < passages.length) await new Promise((r) => setTimeout(r, 0));
  }
  return scores;
}

// ------------------------------------------------------------- pipeline ----

async function search(query, opts) {
  const t = {};
  const rankings = [];
  const lexRank = new Map(), denRank = new Map();
  let t0;

  if (opts.lexical) {
    t0 = performance.now();
    const lex = bm25Search(query, S.meta.candidates);
    t.lexical = performance.now() - t0;
    lex.forEach(([i], r) => lexRank.set(i, r + 1));
    rankings.push(lex);
  }
  if (opts.dense) {
    t0 = performance.now();
    const qvec = await embedQuery(query);
    const den = denseSearch(qvec, S.meta.candidates);
    t.dense = performance.now() - t0;
    den.forEach(([i], r) => denRank.set(i, r + 1));
    rankings.push(den);
  }
  if (!rankings.length) return null;

  let fused;
  if (rankings.length === 1) {
    fused = rankings[0];
  } else {
    t0 = performance.now();
    fused = rrf(rankings, S.meta.rrf_k);
    t.fusion = performance.now() - t0;
  }

  const depth = opts.rerank
    ? Math.min(S.rerankDepth || S.meta.rerank_top_n, fused.length) : 10;
  const hits = fused.slice(0, depth).map(([i, score]) => ({
    idx: i, score,
    lexical_rank: lexRank.get(i) || null,
    dense_rank: denRank.get(i) || null,
    rerank_score: null,
  }));

  t.total = Object.values(t).reduce((a, b) => a + b, 0);
  // Reranking is returned as a continuation rather than awaited. On WASM it
  // costs seconds, and the recall stage already has usable results in ~200ms -
  // so the UI paints those first and refines when the cross-encoder lands.
  const refine = !opts.rerank || !hits.length ? null : async () => {
    const t1 = performance.now();
    const logits = await rerankPairs(query, hits.map((h) => S.chunks[h.idx].t));
    hits.forEach((h, i) => {
      h.rerank_logit = logits[i];
      h.rerank_score = toConfidence(logits[i]);
    });
    hits.sort((a, b) => b.rerank_score - a.rerank_score);
    t.rerank = performance.now() - t1;
    t.total = Object.values(t).reduce((a, b) => a + (b === t.total ? 0 : b), 0);
    return { query, hits: hits.slice(0, 10), timings: t, candidates: fused.length };
  };
  return { query, hits: hits.slice(0, 10), timings: t, candidates: fused.length, refine };
}

/** Mirrors answer.should_abstain: only a cross-encoder score has absolute meaning. */
function shouldAbstain(res) {
  if (!res.hits.length) return { abstain: true, reason: "no_results", score: 0 };
  const top = res.hits[0];
  if (top.rerank_score === null) {
    return { abstain: false, reason: "no_confidence_signal", score: top.score };
  }
  if (top.rerank_score < S.meta.abstain_threshold) {
    return { abstain: true, reason: "low_confidence", score: top.rerank_score };
  }
  return { abstain: false, reason: "", score: top.rerank_score };
}

export { S, boot, search, shouldAbstain, tokenize, embedQuery };

// ------------------------------------------------- citation verification ----

const QUOTE_MAP = { "‘": "'", "’": "'", "“": '"', "”": '"',
                    "–": "-", "—": "-", " ": " ", "§": "S" };

/** Fold case, whitespace and smart punctuation, keeping a map to real offsets. */
function normalize(text) {
  const out = [], map = [];
  let prevSpace = true;
  const src = text.normalize("NFKC");
  for (let i = 0; i < src.length; i++) {
    let c = src[i];
    c = QUOTE_MAP[c] || c;
    if (/\s/.test(c)) {
      if (prevSpace) continue;
      out.push(" "); map.push(i); prevSpace = true;
    } else {
      out.push(c.toLowerCase()); map.push(i); prevSpace = false;
    }
  }
  while (out.length && out[out.length - 1] === " ") { out.pop(); map.pop(); }
  return { text: out.join(""), map };
}

/**
 * Locate a quoted span inside a chunk.
 *
 * Stricter than the Python original, which falls back to a difflib ratio: this
 * accepts only an exact match after normalisation. That errs toward dropping a
 * real citation, never toward accepting a fabricated one - the safe direction.
 */
function locateQuote(quote, haystack) {
  if (!quote || quote.trim().length < 12) return null;
  const q = normalize(quote), h = normalize(haystack);
  if (!q.text || !h.text) return null;
  const pos = h.text.indexOf(q.text);
  if (pos === -1) return null;
  return [h.map[pos], h.map[Math.min(pos + q.text.length - 1, h.map.length - 1)] + 1];
}

function verifyCitations(cites, hits) {
  const verified = [];
  let dropped = 0;
  for (const c of cites) {
    const idx = parseInt(c.source, 10) - 1;
    if (!(idx >= 0 && idx < hits.length)) { dropped++; continue; }
    const hit = hits[idx];
    const chunk = S.chunks[hit.idx];
    const found = locateQuote(String(c.quote || ""), chunk.t);
    if (!found) { dropped++; continue; }
    verified.push({
      source: idx + 1, doc_id: chunk.d, citation: S.docs[chunk.d].c,
      quote: chunk.t.slice(found[0], found[1]),
      doc_char_start: chunk.s + found[0], doc_char_end: chunk.s + found[1],
    });
  }
  return { verified, dropped };
}

// ---------------------------------------------------------- generation ----

const SYSTEM_PROMPT = `You answer questions about US federal regulations using ONLY the numbered sources provided.

Rules:
- Use only the sources. If they do not contain the answer, set "sufficient" to false.
- Cite with [n] inline, matching the source numbers.
- For every [n] you use, add a citation entry whose "quote" is copied VERBATIM from that source. Do not paraphrase inside "quote".
- Quote the shortest span that supports the claim, at least 12 characters.
- Regulations are precise. Preserve numbers, deadlines and conditions exactly.
- Do not give legal advice or add requirements that are not in the sources.

Return ONLY JSON of this shape:
{"answer": "...", "sufficient": true, "citations": [{"source": 1, "quote": "..."}]}`;

async function generate(query, hits, key) {
  const blocks = hits.map((h, i) => {
    const c = S.chunks[h.idx], d = S.docs[c.d];
    return `[${i + 1}] ${d.c} - ${d.h}\n${c.t}`;
  });
  const body = {
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [{ role: "user", parts: [{
      text: `Sources:\n\n${blocks.join("\n\n---\n\n")}\n\nQuestion: ${query}`,
    }] }],
    generationConfig: { temperature: 0, responseMimeType: "application/json" },
  };
  const res = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-lite-latest:generateContent",
    { method: "POST", headers: { "content-type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify(body) });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`Gemini ${res.status}: ${msg.slice(0, 180)}`);
  }
  const data = await res.json();
  return JSON.parse(data.candidates[0].content.parts[0].text);
}

// ---------------------------------------------------------------- view ----

function renderStrip(res, gate, status) {
  const t = res.timings;
  const pct = Math.round(Math.min(1, Math.max(0, gate.score)) * 100);
  const cls = { answered: "ok", ok: "ok", abstained: "warn",
                no_key: "mute", failed: "warn" }[status] || "info";
  const stages = ["lexical", "dense", "fusion", "rerank"]
    .filter((k) => t[k] !== undefined)
    .map((k) => `${k} ${Math.round(t[k])}ms`).join("  ·  ");
  $("strip").innerHTML = `
    <span class="pill ${cls}">${esc(status.replace(/_/g, " "))}</span>
    <span class="meter" title="Top rerank score vs the abstention threshold">
      confidence <span class="track"><span class="fill" style="width:${pct}%"></span></span>
      <span>${gate.score.toFixed(2)}</span><span class="tick"></span>
      <span>τ=${S.meta.abstain_threshold}</span>
    </span>
    <span>${res.candidates} candidates → ${res.hits.length} shown</span>
    <span>${stages}</span>`;
  $("strip").hidden = false;
}

function renderAnswer(text, cites, kind) {
  const html = esc(text).replace(/\[(\d+)\]/g, (m, n) => {
    const c = cites.find((x) => String(x.source) === n);
    return c ? `<button class="cite" data-doc="${esc(c.doc_id)}" data-start="${c.doc_char_start}"
      data-end="${c.doc_char_end}" title="${esc(c.citation)}">${n}</button>` : m;
  }).replace(/\n{2,}/g, "</p><p>");
  $("answer").innerHTML = `<div class="answer ${kind}"><p>${html}</p></div>`;
  $("answer").querySelectorAll(".cite").forEach((b) => b.addEventListener("click",
    () => openSource(b.dataset.doc, +b.dataset.start, +b.dataset.end)));
}

function renderHits(hits) {
  if (!hits.length) { $("results").innerHTML = `<p class="empty">No sections matched.</p>`; return; }
  $("results").innerHTML = `<h2 class="sec">Ranked sections</h2>` + hits.map((h, i) => {
    const c = S.chunks[h.idx], d = S.docs[c.d];
    const score = h.rerank_score ?? h.score;
    const prov = [h.lexical_rank ? `bm25 #${h.lexical_rank}` : null,
                  h.dense_rank ? `dense #${h.dense_rank}` : null].filter(Boolean).join("  ");
    return `<article class="hit">
      <div class="hit-head"><span class="hit-rank">${i + 1}.</span>
        <span class="hit-cite">${esc(d.c)}</span>
        <span class="hit-score">${score.toFixed(3)}</span></div>
      <div class="hit-heading">${esc(d.h)}</div>
      <div class="hit-text">${esc(c.t.replace(/\s+/g, " ").slice(0, 340))}${c.t.length > 340 ? "…" : ""}</div>
      <div class="hit-meta"><span class="prov">${esc(prov || "fused")}</span>
        <span>chars ${c.s}–${c.e}</span>
        <button class="chip" data-doc="${esc(c.d)}" data-start="${c.s}" data-end="${c.e}">Open section</button>
        ${d.u ? `<a class="src" href="${esc(d.u)}" target="_blank" rel="noopener">eCFR</a>` : ""}
      </div></article>`;
  }).join("");
  $("results").querySelectorAll(".hit-meta .chip").forEach((b) => b.addEventListener("click",
    () => openSource(b.dataset.doc, +b.dataset.start, +b.dataset.end)));
}

function openSource(docId, start, end) {
  const d = S.docs[docId];
  if (!d) return;
  const body = start >= 0 && end > start && end <= d.t.length
    ? esc(d.t.slice(0, start)) + "<mark>" + esc(d.t.slice(start, end)) + "</mark>" + esc(d.t.slice(end))
    : esc(d.t);
  $("drawer-cite").textContent = d.c;
  $("drawer-heading").textContent = d.h;
  $("drawer-body").innerHTML = body;
  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  $("scrim").classList.add("open");
  const el = $("drawer-body");
  const mark = el.querySelector("mark");
  if (mark) requestAnimationFrame(() => {
    el.scrollTop += mark.getBoundingClientRect().top - el.getBoundingClientRect().top - el.clientHeight / 3;
  });
}
function closeSource() {
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("open");
}

/** Resolve after a paint, or after 40ms in a background tab where rAF is idle. */
function nextPaint() {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    requestAnimationFrame(() => requestAnimationFrame(finish));
    setTimeout(finish, 40);
  });
}

// ------------------------------------------------------------- wiring ----

async function run() {
  const query = $("q").value.trim();
  if (!query) return;
  const opts = { lexical: $("lexical").checked, dense: $("dense").checked, rerank: $("rerank").checked };
  if (!opts.lexical && !opts.dense) {
    $("results").innerHTML = `<p class="err">Enable at least one retriever — BM25 or Dense.</p>`;
    return;
  }
  $("go").disabled = true; $("go").textContent = "…";
  $("answer").innerHTML = ""; $("strip").hidden = true;
  $("results").innerHTML = `<p class="empty">Retrieving…</p>`;

  try {
    // Phase 1: recall stage only. Fast enough to paint immediately.
    let res = await search(query, opts);
    S.lastHits = res.hits;
    renderStrip(res, { score: 0, abstain: false }, res.refine ? "scoring…" : "ok");
    renderHits(res.hits);

    // Phase 2: cross-encoder. Seconds on WASM, so it refines in place.
    // Yield before starting it: JS is single-threaded, and beginning the rerank
    // synchronously after renderHits means the browser never paints the phase-1
    // results, silently defeating the split. rAF is raced against a timer
    // because requestAnimationFrame does not fire in a background tab - relying
    // on it alone hangs the search for anyone who switches away mid-query.
    if (res.refine) {
      await nextPaint();
      res = await res.refine();
      S.lastHits = res.hits;
      renderHits(res.hits);
    }

    const gate = shouldAbstain(res);
    let status = gate.abstain ? "abstained" : "ok";

    if (gate.abstain) {
      renderAnswer(
        `The closest sections scored ${gate.score.toFixed(2)}, below the ${S.meta.abstain_threshold} `
        + `confidence threshold, so this is left unanswered rather than guessed. `
        + `The nearest matches are shown below.`, [], "abstained");
    } else if ($("generate").checked) {
      const key = localStorage.getItem("cfr_gemini_key");
      if (!key) {
        status = "no_key";
        renderAnswer("Add a Gemini API key above to get a written answer. "
          + "Retrieval below works without one.", [], "degraded");
      } else {
        renderAnswer("Writing an answer…", [], "");
        try {
          const out = await generate(query, res.hits.slice(0, 8), key);
          const { verified, dropped } = verifyCitations(out.citations || [], res.hits);
          status = "answered";
          renderAnswer(out.answer || "", verified, "");
          if (dropped) {
            $("answer").querySelector(".answer").insertAdjacentHTML("beforeend",
              `<p class="warnline">${dropped} citation(s) quoted text not present in the cited
               source and were removed. Verification matches every quoted span against the
               section it claims to come from.</p>`);
          }
        } catch (err) {
          status = "failed";
          renderAnswer(`Generation failed: ${err.message}`, [], "degraded");
        }
      }
    }
    renderStrip(res, gate, status);
  } catch (err) {
    $("results").innerHTML = `<p class="err">${esc(err.message)}</p>`;
  } finally {
    $("go").disabled = false; $("go").textContent = "Search";
  }
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("examples").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) { $("q").value = e.target.textContent; run(); }
});
$("savekey").addEventListener("click", () => {
  const v = $("apikey").value.trim();
  if (v) { localStorage.setItem("cfr_gemini_key", v); $("generate").checked = true; }
  else localStorage.removeItem("cfr_gemini_key");
  $("apikey").value = "";
  $("savekey").textContent = v ? "Saved" : "Cleared";
  setTimeout(() => { $("savekey").textContent = "Save"; }, 1500);
});
$("drawer-close").addEventListener("click", closeSource);
$("scrim").addEventListener("click", closeSource);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSource(); });

if (localStorage.getItem("cfr_gemini_key")) {
  $("apikey").placeholder = "Gemini key saved in this browser — type to replace";
  $("generate").checked = true;
}

boot();
