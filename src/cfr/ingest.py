"""Fetch CFR parts from the eCFR API and store one document per section.

The eCFR XML nests DIV1 (title) > DIV3 (chapter) > DIV5 (part) > DIV6 (subpart)
> DIV8 (section). We flatten to DIV8 because a section is the unit people cite,
link to, and reason about.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import httpx

from . import config

# Tags whose text forms a paragraph in the rendered document.
BLOCK_TAGS = {"P", "FP", "PSPACE", "HED", "EXTRACT", "NOTE"}
# Source-note citations like "[45 FR 33142, May 19, 1980]" - metadata, not content.
SKIP_TAGS = {"CITA", "EDNOTE", "AUTH", "SOURCE"}

_WS = re.compile(r"[ \t ]+")
_BLANKS = re.compile(r"\n{3,}")


def _inline_text(el: ET.Element) -> str:
    """All text inside an element, with inline markup (<I>, <E>) flattened."""
    return _WS.sub(" ", "".join(el.itertext())).strip()


def _render(el: ET.Element, out: List[str]) -> None:
    """Walk an element tree, appending readable paragraphs to `out`."""
    tag = el.tag.upper()
    if tag in SKIP_TAGS:
        return
    if tag == "TABLE":
        for row in el.iter("TR"):
            cells = [_inline_text(c) for c in row if c.tag.upper() in ("TD", "TH")]
            line = " | ".join(c for c in cells if c)
            if line:
                out.append(line)
        return
    if tag in BLOCK_TAGS:
        text = _inline_text(el)
        if text:
            out.append(text)
        return
    for child in el:
        _render(child, out)


def _citation(el: ET.Element, title: str, section: str) -> str:
    raw = el.get("hierarchy_metadata")
    if raw:
        try:
            meta = json.loads(raw.replace("&quot;", '"').replace("&amp;quot;", '"'))
            cite = meta.get("citation")
            if cite:
                return str(cite)
        except (ValueError, AttributeError):
            pass
    return "{} CFR {}".format(title, section)


def parse_part(xml_bytes: bytes, title: str, part: str, source_url: str) -> List[Dict]:
    """Turn one part's XML into document rows, one per section or appendix.

    Sections are not at a fixed depth: most sit directly under a DIV6 subpart,
    but a good number hide under a DIV7 subject group. Walking a fixed path
    silently drops those, so we find every DIV8/DIV9 anywhere in the tree and
    reconstruct its heading path from its ancestors.
    """
    root = ET.fromstring(xml_bytes)
    fetched = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    parent = {child: p for p in root.iter() for child in p}

    def head_of(el: ET.Element) -> str:
        h = el.find("HEAD")
        return _inline_text(h) if h is not None else ""

    def ancestor_headings(el: ET.Element) -> List[str]:
        chain: List[str] = []
        cur = el
        while cur in parent:
            cur = parent[cur]
            if cur.tag.upper() in ("DIV5", "DIV6", "DIV7"):
                h = head_of(cur)
                if h:
                    chain.append(h)
        return list(reversed(chain))

    docs: List[Dict] = []
    seen: Dict[str, int] = {}

    for el in root.iter():
        tag = el.tag.upper()
        if tag not in ("DIV8", "DIV9"):
            continue
        number = (el.get("N") or "").strip()
        if not number:
            continue

        head_el = el.find("HEAD")
        heading = _inline_text(head_el) if head_el is not None else number

        body: List[str] = []
        for child in el:
            if child is head_el:
                continue
            _render(child, body)
        if not body:
            continue  # [Reserved] and stub sections carry no content

        if tag == "DIV9":
            # Appendices are numbered per parent section ("A", "B"), so they
            # need the section number to form a unique, citable id.
            owner = ""
            cur = el
            while cur in parent:
                cur = parent[cur]
                if cur.tag.upper() == "DIV8":
                    owner = (cur.get("N") or "").strip()
                    break
            ident = "{}-app{}".format(owner or part, number.replace(" ", ""))
        else:
            ident = number

        doc_id = "ecfr:{}:{}".format(title, ident)
        if doc_id in seen:
            seen[doc_id] += 1
            doc_id = "{}~{}".format(doc_id, seen[doc_id])
        else:
            seen[doc_id] = 0

        text = _BLANKS.sub("\n\n", heading + "\n\n" + "\n\n".join(body)).strip()
        docs.append(
            {
                "doc_id": doc_id,
                "title": title,
                "part": part,
                "section": ident,
                "heading": heading,
                "citation": _citation(el, title, ident),
                "text": text,
                "source_url": source_url,
                "fetched_at": fetched,
                "heading_path": " > ".join(
                    [p for p in ["Title " + title] + ancestor_headings(el) + [heading] if p]
                ),
            }
        )
    return docs


def fetch_part(title: str, part: str, date: Optional[str] = None) -> Tuple[bytes, str]:
    date = date or config.ECFR_DATE
    url = "{}/full/{}/title-{}.xml".format(config.ECFR_API, date, title)
    # The API rejects clients that do not accept compression.
    headers = {"Accept-Encoding": "gzip, deflate", "User-Agent": "cfr-retrieval/0.1"}
    with httpx.Client(timeout=180.0, follow_redirects=True, headers=headers) as client:
        resp = client.get(url, params={"part": part})
        resp.raise_for_status()
        return resp.content, str(resp.url)


def ingest(
    conn: sqlite3.Connection,
    parts: Optional[List[Tuple[str, str]]] = None,
    date: Optional[str] = None,
) -> Dict[str, int]:
    parts = parts or config.DEFAULT_PARTS
    written = 0
    headings: Dict[str, str] = {}

    for title, part in parts:
        print("  fetching {} CFR part {} ...".format(title, part), flush=True)
        try:
            xml_bytes, url = fetch_part(title, part, date)
        except httpx.HTTPError as exc:
            print("    ! skipped: {}".format(exc))
            continue

        docs = parse_part(xml_bytes, title, part, url)
        for d in docs:
            headings[d["doc_id"]] = d.pop("heading_path")
            conn.execute(
                """INSERT INTO documents
                   (doc_id, title, part, section, heading, citation, text, source_url, fetched_at)
                   VALUES (:doc_id, :title, :part, :section, :heading, :citation,
                           :text, :source_url, :fetched_at)
                   ON CONFLICT(doc_id) DO UPDATE SET
                     heading=excluded.heading, citation=excluded.citation,
                     text=excluded.text, source_url=excluded.source_url,
                     fetched_at=excluded.fetched_at""",
                d,
            )
        written += len(docs)
        print("    {} sections".format(len(docs)), flush=True)
    conn.commit()

    # Heading paths are chunk-level metadata; stash them for the chunker.
    (config.DATA_DIR / "heading_paths.json").write_text(json.dumps(headings), encoding="utf-8")
    return {"documents": written, "parts": len(parts)}
