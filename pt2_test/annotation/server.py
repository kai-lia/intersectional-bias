"""
Span annotation server for identity mentions in model outputs.

Why a local server rather than a static page: the annotations have to land in a
CSV on disk, incrementally, so a session can be interrupted and resumed.  A
browser page alone can only offer a download at the end, which loses work.

Stdlib only -- no install step.

    python pt2_test/annotation/server.py
    open http://localhost:8765

Data in:   pt2_test/data/eval/{model}_mentions_{full,extra}.csv
Data out:  pt2_test/annotation/spans.csv     one row per labelled span
           pt2_test/annotation/progress.csv  one row per item marked done

The span CSV keeps character offsets into the exact `output` string shown, so
spans can be recovered later for training without re-deriving anything:

    item_id, model, template, condition, identity_a, identity_b,
    label, start, end, text, note, annotator, ts

`label` is identity_a / identity_b / other.  Multiple spans per identity are the
point -- a model may name the same identity three times in one justification, and
each mention is its own row.
"""
import csv
import json
import os
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "data" / "eval"
SPANS = HERE / "spans.csv"
PROGRESS = HERE / "progress.csv"
PORT = int(os.environ.get("PORT", 8765))

SPAN_COLS = ["item_id", "model", "template", "condition", "identity_a", "identity_b",
             "label", "start", "end", "text", "note", "annotator", "ts"]

SENTS = HERE / "sentences.csv"
SENT_COLS = ["item_id", "model", "template", "condition", "identity_a", "identity_b",
             "sent_idx", "start", "end", "sentence",
             "evokes", "blames_a", "blames_b", "annotator", "ts"]

import re as _re
# Split on sentence-final punctuation followed by space, and on line breaks --
# these outputs contain numbered lists and bullets, which are units of argument
# even without terminal punctuation.  Offsets are preserved so a sentence can be
# located back in the original string.
_SPLIT = _re.compile(r"(?<=[.!?])\s+|\n+")


# A bare enumeration marker ("1.", "-", "*") is a unit of formatting, not of
# argument, and asking a human to label it wastes a keystroke.  Deliberately
# narrow: "No." and "Here's why:" are short but ARE content and stay in.
_MARKER = _re.compile(r"^[\s\W]*(\d+[.)]?|[-*•]+)[\s\W]*$")


def is_marker(t):
    return bool(_MARKER.match(t))


def split_sentences(text):
    out, pos = [], 0
    for m in _SPLIT.finditer(text):
        seg = text[pos:m.start()]
        if seg.strip():
            out.append((pos, m.start(), seg))
        pos = m.end()
    if text[pos:].strip():
        out.append((pos, len(text), text[pos:]))
    return out


def load_items():
    """One row per model output, solo rows carrying identity_b = None."""
    frames = []
    for m in ["granite", "llama", "mistral"]:
        for tag in ["_full", "_extra"]:
            p = EVAL / f"{m}_mentions{tag}.csv"
            if p.exists():
                d = pd.read_csv(p)
                d["model"] = m
                frames.append(d)
    if not frames:
        sys.exit(f"no mention CSVs found in {EVAL}")
    d = pd.concat(frames, ignore_index=True)
    d["identity_a"] = d.s1
    d["identity_b"] = d.s2.where(d.condition != "single")
    d["output"] = d.response.astype(str)
    # stable id: survives reordering, so annotations keep pointing at the same text
    # fillna BEFORE concatenating: pandas' string dtype propagates NA through
    # astype(str), so a null identity_b would null the whole id on every solo row
    d["item_id"] = (d.model + "|" + d.pattern_id.astype(str) + "|" + d.condition
                    + "|" + d.identity_a.fillna("none").astype(str)
                    + "|" + d.identity_b.fillna("none").astype(str))
    keep = ["item_id", "model", "pattern_id", "condition", "identity_a", "identity_b", "output"]
    d = d[keep].rename(columns={"pattern_id": "template"})
    return d.drop_duplicates("item_id").reset_index(drop=True)


def stratify(d):
    """Round-robin over (model, template) so the first N items annotated are a
    balanced sample of the design rather than a contiguous block of the file.

    File order is model-major and template-major: annotating in it yields, at
    n=50, one template of one model -- which cannot support any claim that has
    to survive a template bootstrap.  Order is deterministic (fixed seed) so a
    session is reproducible, and annotations key on item_id, not index, so
    re-ordering never orphans existing work.
    """
    import numpy as np
    # A DISTINCT seed per cell.  Sharing one seed gave every (model, template)
    # cell the same permutation, so the round-robin served the SAME identity
    # pairs in every template -- 39 llama outputs turned out to be 4 pairs, and
    # a bootstrap over templates counted one fact about a pair as twelve.
    cells = [g.sample(frac=1, random_state=abs(hash(k)) % (2 ** 31)).index.tolist()
             for k, g in d.groupby(["model", "template"], sort=True)]
    order = [c[i] for i in range(max(map(len, cells))) for c in cells if i < len(c)]
    return d.loc[order].reset_index(drop=True)


ITEMS = load_items()
if os.environ.get("ANNOT_ORDER", "stratified") != "file":
    ITEMS = stratify(ITEMS)
print(f"loaded {len(ITEMS):,} items  "
      f"({(ITEMS.condition == 'single').sum():,} solo, "
      f"{(ITEMS.condition != 'single').sum():,} pairs)  "
      f"order={os.environ.get('ANNOT_ORDER', 'stratified')}")


def ensure(path, cols):
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(cols)


ensure(SPANS, SPAN_COLS)
ensure(SENTS, SENT_COLS)
ensure(PROGRESS, ["item_id", "annotator", "ts"])


def read_spans(item_id):
    if not SPANS.exists():
        return []
    out = []
    with open(SPANS, newline="") as f:
        for r in csv.DictReader(f):
            if r["item_id"] == item_id:
                r["start"] = int(r["start"]); r["end"] = int(r["end"])
                out.append(r)
    return out


def done_ids():
    if not PROGRESS.exists():
        return set()
    with open(PROGRESS, newline="") as f:
        return {r["item_id"] for r in csv.DictReader(f)}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass                                    # quiet

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _fail(self, e):
        import traceback; traceback.print_exc()
        self._send(500, json.dumps({"error": str(e)}))

    def do_GET(self):
        try:
            return self._get()
        except Exception as e:
            return self._fail(e)

    def _get(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path in ("/sentences", "/sentences.html"):
            return self._send(200, (HERE / "sentences.html").read_bytes(),
                              "text/html; charset=utf-8")
        if u.path == "/api/sitem":
            i = int(q.get("i", ["0"])[0]) % len(ITEMS)
            row = ITEMS.iloc[i]
            def native(v):
                if v is None or (not isinstance(v, str) and pd.isna(v)):
                    return None
                return v.item() if hasattr(v, "item") else v
            d = {k: native(row[k]) for k in ITEMS.columns}
            d["index"] = i; d["total"] = len(ITEMS)
            # keep marker units in the list so sent_idx stays stable against
            # annotations already on disk; flag them so the client skips past
            d["sentences"] = [{"i": k, "start": a, "end": b, "text": t,
                               "skip": is_marker(t)}
                              for k, (a, b, t) in enumerate(split_sentences(d["output"] or ""))]
            done = {}
            if SENTS.exists():
                with open(SENTS, newline="") as f:
                    for r in csv.DictReader(f):
                        if r["item_id"] == row.item_id:
                            done[int(r["sent_idx"])] = r
            d["done"] = done
            return self._send(200, json.dumps(d))
        if u.path == "/api/sstats":
            n = 0; items = set()
            with open(SENTS, newline="") as f:
                for r in csv.DictReader(f):
                    n += 1; items.add(r["item_id"])
            return self._send(200, json.dumps({"sentences": n, "items": len(items),
                                               "total": len(ITEMS)}))
        if u.path == "/api/item":
            i = int(q.get("i", ["0"])[0]) % len(ITEMS)
            row = ITEMS.iloc[i]
            # numpy scalars are not JSON-serialisable; coerce to python natives
            def native(v):
                if v is None or (not isinstance(v, str) and pd.isna(v)):
                    return None
                return v.item() if hasattr(v, "item") else v
            d = {k: native(row[k]) for k in ITEMS.columns}
            d["index"] = i
            d["total"] = len(ITEMS)
            d["spans"] = read_spans(row.item_id)
            d["done"] = row.item_id in done_ids()
            return self._send(200, json.dumps(d))
        if u.path == "/api/stats":
            dn = done_ids()
            n = 0
            with open(SPANS, newline="") as f:
                n = sum(1 for _ in csv.DictReader(f))
            return self._send(200, json.dumps({"done": len(dn), "spans": n,
                                               "total": len(ITEMS)}))
        if u.path == "/api/find":
            # first item not yet marked done, from index i onward
            dn = done_ids()
            start = int(q.get("i", ["0"])[0])
            for k in range(start, start + len(ITEMS)):
                j = k % len(ITEMS)
                if ITEMS.iloc[j].item_id not in dn:
                    return self._send(200, json.dumps({"index": j}))
            return self._send(200, json.dumps({"index": start}))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        try:
            return self._post()
        except Exception as e:
            return self._fail(e)

    def _post(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or "{}")
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/span":
            row = [body.get(c, "") for c in SPAN_COLS[:-1]] + [time.strftime("%F %T")]
            with open(SPANS, "a", newline="") as f:
                csv.writer(f).writerow(row)
            return self._send(200, json.dumps({"ok": True}))
        if u.path == "/api/unspan":
            keep = []
            with open(SPANS, newline="") as f:
                rd = csv.DictReader(f)
                for r in rd:
                    if not (r["item_id"] == body["item_id"]
                            and int(r["start"]) == body["start"]
                            and int(r["end"]) == body["end"]
                            and r["label"] == body["label"]):
                        keep.append(r)
            with open(SPANS, "w", newline="") as f:
                w = csv.DictWriter(f, SPAN_COLS); w.writeheader(); w.writerows(keep)
            return self._send(200, json.dumps({"ok": True}))
        if u.path == "/api/sent":
            keep = []
            if SENTS.exists():
                with open(SENTS, newline="") as f:
                    for r in csv.DictReader(f):
                        if not (r["item_id"] == body["item_id"]
                                and int(r["sent_idx"]) == body["sent_idx"]):
                            keep.append(r)          # overwrite, so re-labelling works
            keep.append({c: body.get(c, "") for c in SENT_COLS[:-1]}
                        | {"ts": time.strftime("%F %T")})
            with open(SENTS, "w", newline="") as f:
                w = csv.DictWriter(f, SENT_COLS); w.writeheader(); w.writerows(keep)
            return self._send(200, json.dumps({"ok": True}))
        if u.path == "/api/done":
            with open(PROGRESS, "a", newline="") as f:
                csv.writer(f).writerow([body["item_id"], body.get("annotator", ""),
                                        time.strftime("%F %T")])
            return self._send(200, json.dumps({"ok": True}))
        return self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    print(f"annotating -> {SPANS}")
    print(f"open http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
