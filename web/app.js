const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let lastHits = [];

// --- corpus line -----------------------------------------------------------
fetch("/api/stats").then((r) => r.json()).then((s) => {
  const chunks = (s.strategies.find((x) => x.strategy === "structured") || {}).chunks || 0;
  const bits = [
    `${s.documents.toLocaleString()} sections`,
    `${chunks.toLocaleString()} chunks`,
    s.generation_enabled
      ? `${s.budget.remaining}/${s.budget.limit} answers left today`
      : "retrieval only",
  ];
  $("corpusline").textContent = bits.join(" · ");
}).catch(() => { $("corpusline").textContent = "corpus unavailable"; });

// --- search ----------------------------------------------------------------
$("form").addEventListener("submit", (e) => { e.preventDefault(); run(); });
$("examples").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) { $("q").value = e.target.textContent; run(); }
});

async function run() {
  const query = $("q").value.trim();
  if (!query) return;

  const generate = $("generate").checked;
  const body = {
    query,
    strategy: $("strategy").value,
    lexical: $("lexical").checked,
    dense: $("dense").checked,
    rerank: $("rerank").checked,
  };
  if (!body.lexical && !body.dense) {
    $("results").innerHTML = `<p class="err">Enable at least one retriever — BM25 or Dense.</p>`;
    return;
  }

  $("go").disabled = true;
  $("go").textContent = "…";
  $("answer").innerHTML = "";
  $("results").innerHTML = `<p class="empty">Retrieving…</p>`;
  $("strip").hidden = true;

  try {
    const res = await fetch(generate ? "/api/ask" : "/api/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (res.status === 429) {
      $("results").innerHTML =
        `<p class="err">Rate limited. Try again in ${data.retry_after_s}s.</p>`;
      return;
    }
    if (!res.ok) {
      $("results").innerHTML = `<p class="err">${esc(data.detail || "Request failed")}</p>`;
      return;
    }
    render(data, generate);
  } catch (err) {
    $("results").innerHTML = `<p class="err">Network error: ${esc(err.message)}</p>`;
  } finally {
    $("go").disabled = false;
    $("go").textContent = "Search";
  }
}

// --- rendering -------------------------------------------------------------
function render(data, generated) {
  lastHits = data.hits || [];
  renderStrip(data, generated);
  if (generated) renderAnswer(data);
  renderHits(lastHits);
}

function renderStrip(data, generated) {
  const t = data.timings_ms || {};
  const conf = data.confidence ?? 0;
  const thr = data.threshold ?? 0;
  const pct = Math.round(Math.min(1, Math.max(0, conf)) * 100);

  const status = generated ? data.status : (data.abstain ? "abstained" : "ok");
  const cls = { answered: "ok", ok: "ok", abstained: "warn", insufficient: "warn",
                budget_exhausted: "mute", retrieval_only: "mute",
                generation_failed: "warn" }[status] || "info";

  const stages = ["lexical", "dense", "fusion", "rerank"]
    .filter((k) => t[k] !== undefined)
    .map((k) => `${k} ${Math.round(t[k])}ms`)
    .join("  ·  ");

  $("strip").innerHTML = `
    <span class="pill ${cls}">${esc(String(status).replace(/_/g, " "))}</span>
    <span class="meter" title="Top rerank score vs the abstention threshold (${thr})">
      confidence
      <span class="track"><span class="fill" style="width:${pct}%"></span></span>
      <span>${conf.toFixed(2)}</span>
      <span class="tick" style="margin-left:2px"></span>
      <span>τ=${thr}</span>
    </span>
    <span>${data.candidate_count ?? 0} candidates → ${lastHits.length} shown</span>
    <span>${stages}</span>
    ${data.cached ? `<span class="pill info">${esc(data.cached)} cache hit</span>` : ""}`;
  $("strip").hidden = false;
}

function renderAnswer(data) {
  const abstained = data.status === "abstained";
  const degraded = ["budget_exhausted", "retrieval_only", "generation_failed"].includes(data.status);
  const cites = data.citations || [];

  // Turn [1] markers into buttons that open the cited span in the source.
  let html = esc(data.answer || "").replace(/\[(\d+)\]/g, (m, n) => {
    const c = cites.find((x) => String(x.source) === n);
    return c
      ? `<button class="cite" data-doc="${esc(c.doc_id)}" data-start="${c.doc_char_start}" data-end="${c.doc_char_end}" title="${esc(c.citation)}">${n}</button>`
      : m;
  }).replace(/\n{2,}/g, "</p><p>");

  const warn = data.citation_warning
    ? `<p class="warnline">${esc(data.citation_warning)} Verification compares each quoted span against the cited section, so invented citations are removed rather than displayed.</p>`
    : "";

  $("answer").innerHTML =
    `<div class="answer${abstained ? " abstained" : ""}${degraded ? " degraded" : ""}">
       <p>${html}</p>${warn}
     </div>`;

  $("answer").querySelectorAll(".cite").forEach((b) =>
    b.addEventListener("click", () =>
      openSource(b.dataset.doc, +b.dataset.start, +b.dataset.end)));
}

function renderHits(hits) {
  if (!hits.length) {
    $("results").innerHTML = `<p class="empty">No sections matched.</p>`;
    return;
  }
  const rows = hits.map((h, i) => {
    const score = h.rerank_score ?? h.score;
    const prov = [
      h.lexical_rank ? `bm25 #${h.lexical_rank}` : null,
      h.dense_rank ? `dense #${h.dense_rank}` : null,
    ].filter(Boolean).join("  ");
    const body = h.text.replace(/\s+/g, " ").slice(0, 340);
    return `<article class="hit">
      <div class="hit-head">
        <span class="hit-rank">${i + 1}.</span>
        <span class="hit-cite">${esc(h.citation)}</span>
        <span class="hit-score">${score.toFixed(3)}</span>
      </div>
      <div class="hit-heading">${esc(h.heading)}</div>
      <div class="hit-text">${esc(body)}${h.text.length > 340 ? "…" : ""}</div>
      <div class="hit-meta">
        <span class="prov">${esc(prov || "fused")}</span>
        <span>chars ${h.char_start}–${h.char_end}</span>
        <button class="chip" data-doc="${esc(h.doc_id)}" data-start="${h.char_start}" data-end="${h.char_end}">Open section</button>
        ${h.source_url ? `<a class="src" href="${esc(h.source_url)}" target="_blank" rel="noopener">eCFR</a>` : ""}
      </div>
    </article>`;
  }).join("");

  $("results").innerHTML = `<h2 class="sec">Ranked sections</h2>${rows}`;
  $("results").querySelectorAll(".hit-meta .chip").forEach((b) =>
    b.addEventListener("click", () =>
      openSource(b.dataset.doc, +b.dataset.start, +b.dataset.end)));
}

// --- source drawer with span highlighting ----------------------------------
async function openSource(docId, start, end) {
  const res = await fetch(`/api/document/${encodeURIComponent(docId)}`);
  if (!res.ok) return;
  const doc = await res.json();

  // The offsets recorded at ingest are what make this exact rather than a
  // text search that might land on the wrong occurrence.
  const body = start >= 0 && end > start && end <= doc.text.length
    ? esc(doc.text.slice(0, start)) + "<mark>" + esc(doc.text.slice(start, end)) +
      "</mark>" + esc(doc.text.slice(end))
    : esc(doc.text);

  $("drawer-cite").textContent = doc.citation;
  $("drawer-heading").textContent = doc.heading;
  $("drawer-body").innerHTML = body;
  $("drawer").classList.add("open");
  $("drawer").setAttribute("aria-hidden", "false");
  $("scrim").classList.add("open");

  scrollToMark();
}

// The drawer animates in with a transform, and scrollIntoView is unreliable
// against a transformed ancestor mid-transition - it silently no-ops and the
// highlighted span stays thousands of pixels below the fold. Positioning the
// scroll container directly, after a frame, is deterministic.
function scrollToMark() {
  const body = $("drawer-body");
  const mark = body.querySelector("mark");
  if (!mark) return;
  requestAnimationFrame(() => {
    const markTop = mark.getBoundingClientRect().top;
    const bodyTop = body.getBoundingClientRect().top;
    body.scrollTop += markTop - bodyTop - body.clientHeight / 3;
  });
}

function closeSource() {
  $("drawer").classList.remove("open");
  $("drawer").setAttribute("aria-hidden", "true");
  $("scrim").classList.remove("open");
}
$("drawer-close").addEventListener("click", closeSource);
$("scrim").addEventListener("click", closeSource);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSource(); });
