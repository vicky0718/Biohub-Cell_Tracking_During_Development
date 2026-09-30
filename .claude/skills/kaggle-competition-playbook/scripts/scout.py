#!/usr/bin/env python3
"""Scout a Kaggle competition's public notebooks by their authors' leaderboard scores, and
search Kaggle for a missing dataset or model before writing a notebook off.

    python scout.py board <competition>                     # notebooks ranked by author's LB score
    python scout.py board <competition> --beat 0.947        # only authors scoring better than 0.947
    python scout.py board <competition> --lower-is-better   # for RMSE-style metrics
    python scout.py board <competition> --since 2026-09-01 --sources --json scout.json
    python scout.py find <name-or-mount-path> [...]         # datasets, models, notebooks matching it

Why by author: a notebook's title and votes say what it is called and how popular it is; the
author's leaderboard entry says what their best submission scored. Joining the two finds the
people above you who publish working code. Sorting by title or votes finds the notebook that
everyone has already forked. The author's score is their best submission, not necessarily
this notebook's, so submit a fork before believing it.

Why find: a notebook that dies on a missing dataset, weights file or mount usually needs
something that is public under a slightly different name, often published by the same author.

Credentials: KAGGLE_USERNAME and KAGGLE_KEY in the environment, or ~/.kaggle/kaggle.json
(KAGGLE_CONFIG_DIR is honoured). Standard library only. Read-only: it never submits anything.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = "https://www.kaggle.com/api/v1"


def _creds() -> tuple[str, str]:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        return d["username"], d["key"]
    raise SystemExit("no Kaggle credentials: set KAGGLE_USERNAME/KAGGLE_KEY "
                     "or create ~/.kaggle/kaggle.json")


def _get(path: str, **params) -> bytes:
    user, key = _creds()
    url = f"{API}{path}" + (f"?{urllib.parse.urlencode(params)}" if params else "")
    auth = base64.b64encode(f"{user}:{key}".encode()).decode()
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}",
                                               "User-Agent": "kaggle-playbook-scout/1"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise SystemExit(f"{e.code} on {path}: check the credentials, and that this "
                                 f"account has joined the competition (accepted its rules)")
            if e.code == 404:
                raise SystemExit(f"404 on {path}: check the competition slug")
            if e.code != 429 and e.code < 500:
                raise
            err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            err = f"unreachable ({e.reason})"
        if attempt < 3:
            time.sleep(5 * 2 ** attempt)
    # Say so plainly: an unreachable API is a fact about the plumbing, not about the field.
    raise SystemExit(f"could not reach the Kaggle API for {path}: {err}")


def leaderboard(comp: str) -> list[dict]:
    """The full public leaderboard: Rank, TeamName, Score, TeamMemberUserNames, ..."""
    blob = _get(f"/competitions/{comp}/leaderboard/download")
    if blob[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(blob))
        blob = z.read(z.namelist()[0])
    return list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))


def _index(rows: list[dict]) -> dict[str, dict]:
    """username or team name (lower-case) -> leaderboard row."""
    idx: dict[str, dict] = {}
    for r in rows:
        keys = [r.get("TeamName", "")] + (r.get("TeamMemberUserNames") or "").split(",")
        for k in keys:
            k = k.strip().lower()
            if k and k not in idx:        # rows are in rank order: keep the best entry
                idx[k] = r
    return idx


def notebooks(comp: str, pages: int) -> list[dict]:
    out: list[dict] = []
    for page in range(1, pages + 1):
        ks = json.loads(_get("/kernels/list", competition=comp, pageSize=100, page=page,
                             sortBy="dateRun"))
        out += ks
        if len(ks) < 100:
            break
    return out


def _score(r: dict | None) -> float | None:
    try:
        return float(r["Score"]) if r else None
    except (KeyError, ValueError):
        return None


def board(a) -> int:
    rows = leaderboard(a.competition)
    idx = _index(rows)
    lower = a.lower_is_better
    me = _creds()[0].lower()
    if me in idx:
        print(f"you ({me}): rank {idx[me]['Rank']}, score {idx[me]['Score']}, "
              f"team {idx[me]['TeamName']!r}")
    print(f"leaderboard: {len(rows)} teams, best {rows[0]['Score'] if rows else 'n/a'}")

    table = []
    for k in notebooks(a.competition, a.pages):
        author = (k.get("author") or (k.get("ref") or "/").split("/")[0]).strip()
        r = idx.get(author.lower())
        s = _score(r)
        run = (k.get("lastRunTime") or "")[:10]
        if a.since and run < a.since:
            continue
        if a.beat is not None and (s is None or (s >= a.beat if lower else s <= a.beat)):
            continue
        if s is None and not a.all:
            continue
        table.append({"author_score": s, "author_rank": int(r["Rank"]) if r else None,
                      "team": r["TeamName"] if r else None, "author": author,
                      "ref": k.get("ref", ""), "title": k.get("title", ""),
                      "votes": k.get("totalVotes", 0), "last_run": run, "mounts": []})

    def key(t):
        s = t["author_score"]
        if s is None:
            return (1, 0.0, t["last_run"])
        return (0, s if lower else -s, "")
    table.sort(key=key)
    if a.top:
        table = table[:a.top]
    if a.sources:     # the list endpoint leaves data sources empty; each notebook's metadata has them
        for t in table:
            user, _, slug = t["ref"].partition("/")
            try:
                md = json.loads(_get("/kernels/pull", userName=user, kernelSlug=slug))["metadata"]
                t["mounts"] = [m for key in ("datasetDataSources", "modelDataSources",
                                             "kernelDataSources") for m in md.get(key) or []]
            except (SystemExit, KeyError, ValueError) as e:
                t["mounts"] = [f"(could not read: {e})"]
            time.sleep(0.3)

    print(f"{len(table)} notebooks shown, ranked by their author's leaderboard score "
          f"({'lower' if lower else 'higher'} is better)\n")
    print(f"{'score':>8} {'rank':>6} {'votes':>5}  {'last run':<10}  ref  -  title")
    for t in table:
        s = f"{t['author_score']:.4f}" if t["author_score"] is not None else "-"
        rk = t["author_rank"] if t["author_rank"] is not None else "-"
        print(f"{s:>8} {rk:>6} {t['votes']:>5}  {t['last_run']:<10}  {t['ref']}  -  "
              f"{t['title'][:60]}")
        if a.sources and t["mounts"]:
            print(f"{'':>34}mounts: {', '.join(t['mounts'])}")
    if a.json:
        Path(a.json).write_text(json.dumps(table, indent=1))
        print(f"\nwrote {a.json}")
    return 0


def _slug(name: str) -> list[str]:
    """A mount path like /kaggle/input/foo-bar/weights.pt -> 'foo-bar' (and the raw name)."""
    m = re.search(r"/kaggle/input/(?:datasets/[^/]+/|models/[^/]+/)?([^/]+)", name)
    return [m.group(1)] if m else [name]


def find(a) -> int:
    for raw in a.names:
        for q in _slug(raw):
            print(f"=== {q}")
            ds = json.loads(_get("/datasets/list", search=q, page=1))
            print(f"datasets ({len(ds)}):")
            for d in ds[:a.limit]:
                print(f"  {d.get('ref', '?'):<60} updated {str(d.get('lastUpdated', ''))[:10]}")
            ms = json.loads(_get("/models/list", search=q, pageSize=a.limit)).get("models", [])
            print(f"models ({len(ms)}):")
            for m in ms[:a.limit]:
                ref = m.get("ref") or f"{m.get('owner', '?')}/{m.get('slug', '?')}"
                print(f"  {ref}")
            ks = json.loads(_get("/kernels/list", search=q, pageSize=a.limit, page=1))
            print(f"notebooks mentioning it ({len(ks)}):")
            for k in ks[:a.limit]:
                print(f"  {k.get('ref', '?'):<60} {k.get('title', '')[:50]}")
            if not (ds or ms or ks):
                print("  nothing: try a shorter distinctive fragment of the name, and look "
                      "through the failing notebook's author's datasets")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("board", help="rank a competition's notebooks by author LB score")
    b.add_argument("competition", help="competition slug, as in the URL")
    b.add_argument("--beat", type=float, help="only authors scoring better than this")
    b.add_argument("--lower-is-better", action="store_true", help="metric is an error/loss")
    b.add_argument("--since", default="", help="only notebooks last run on/after YYYY-MM-DD")
    b.add_argument("--top", type=int, default=0, help="show at most this many rows")
    b.add_argument("--pages", type=int, default=10, help="notebook pages of 100 to scan")
    b.add_argument("--all", action="store_true", help="include authors not on the leaderboard")
    b.add_argument("--sources", action="store_true",
                   help="also fetch and print each shown notebook's mounts (one call each)")
    b.add_argument("--json", help="also write the table to this file")
    b.set_defaults(fn=board)
    f = sub.add_parser("find", help="search Kaggle for a missing dataset/model/mount")
    f.add_argument("names", nargs="+", help="name, slug fragment, or /kaggle/input/... path")
    f.add_argument("--limit", type=int, default=8)
    f.set_defaults(fn=find)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
