#!/usr/bin/env python3
"""Scrape every discussion thread + comment from the competition's Kaggle forum.

Kaggle's discussion pages are a client-rendered SPA, so the served HTML carries no
content.  The pages are fed by an internal JSON-RPC API under /api/i/ which answers
anonymously as long as the request carries the XSRF cookie/header pair handed out
with any page load.  That is what this does: bootstrap a session, walk the topic
list, then pull each topic with its full (nested) comment tree.

    python discussions/scrape_discussions.py

Writes:
    discussions/raw/topics_index.json   the paginated topic list, concatenated
    discussions/raw/topic_<id>.json     one untouched API response per thread
    discussions/threads/<id>-<slug>.md  the same thread, readable
    discussions/00-index.md             table of contents, sorted by votes
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
BASE = "https://www.kaggle.com"
RPC = f"{BASE}/api/i"
PAGE_SIZE = 20  # server-side cap; larger pageSize values are ignored

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
THREADS = HERE / "threads"


class Kaggle:
    """Anonymous session against Kaggle's internal RPC endpoints."""

    def __init__(self) -> None:
        self.jar = CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )
        # Deliberately no browser user-agent override: spoofing Chrome without the
        # rest of a browser's header set trips Kaggle's bot check, which answers
        # every RPC with the HTML app shell instead of JSON.  urllib's own default
        # is served normally.
        self.xsrf = ""

    def bootstrap(self) -> None:
        """Load a page once, purely to be issued the XSRF-TOKEN cookie."""
        self.opener.open(f"{BASE}/competitions/{COMPETITION}/discussion", timeout=60).read()
        for cookie in self.jar:
            if cookie.name == "XSRF-TOKEN":
                self.xsrf = cookie.value
        if not self.xsrf:
            raise RuntimeError("no XSRF-TOKEN cookie returned; Kaggle may have changed")

    def rpc(self, method: str, payload: dict, retries: int = 4) -> dict:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(f"{RPC}/{method}", data=body, method="POST")
        req.add_header("content-type", "application/json")
        req.add_header("accept", "application/json")
        req.add_header("x-xsrf-token", self.xsrf)
        for attempt in range(retries):
            try:
                with self.opener.open(req, timeout=90) as resp:
                    text = resp.read().decode()
                if text.lstrip().startswith("<"):
                    raise RuntimeError(
                        f"{method} returned the HTML app shell, not JSON — the "
                        "session was rejected (bot check or expired XSRF token)"
                    )
                return json.loads(text)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                    RuntimeError) as exc:
                if attempt == retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                print(f"    ! {type(exc).__name__} on {method}, retry in {wait}s", file=sys.stderr)
                time.sleep(wait)
        raise RuntimeError("unreachable")


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "untitled").lower()).strip("-")
    return slug[:limit].rstrip("-") or "untitled"


def flatten(comments: list, depth: int = 0) -> list:
    """Depth-first walk of the nested reply tree, keeping the nesting level."""
    out = []
    for c in comments or []:
        out.append((depth, c))
        out.extend(flatten(c.get("replies") or [], depth + 1))
    return out


def author_of(node: dict) -> tuple[str, str]:
    """(display name, tier) for a comment or topic node."""
    a = node.get("author") or {}
    name = a.get("displayName") or node.get("authorUserDisplayName") or "unknown"
    tier = a.get("tier") or node.get("authorPerformanceTier") or ""
    return name, str(tier)


def votes_of(node: dict) -> int:
    v = node.get("votes")
    if isinstance(v, dict):
        return v.get("totalVotes") or 0
    return v or 0


def render(topic: dict, meta: dict) -> str:
    """One thread as markdown: opening post, then every comment in tree order."""
    name, tier = author_of(topic)
    lines = [
        f"# {topic.get('name', 'untitled')}",
        "",
        f"- **URL**: {BASE}{topic.get('url', '')}",
        f"- **Topic id**: {topic.get('id')}",
        f"- **Author**: {name}" + (f" ({tier})" if tier else ""),
        f"- **Posted**: {topic.get('postDate', '')}",
        f"- **Votes**: {meta.get('votes', topic.get('totalVotes', 0))}",
        f"- **Comments**: {meta.get('commentCount', 0)}",
    ]
    if topic.get("isStickied"):
        lines.append("- **Pinned**: yes")
    lines += ["", "---", "", "## Opening post", ""]
    first = topic.get("firstMessage")
    if isinstance(first, dict):
        body = first.get("rawMarkdown") or first.get("content") or ""
    else:
        body = first or ""
    lines.append(body.rstrip() or "*(no body)*")

    walked = flatten(topic.get("comments") or [])
    lines += ["", "---", "", f"## Comments ({len(walked)})", ""]
    if not walked:
        lines.append("*(none)*")
    for depth, c in walked:
        cname, ctier = author_of(c)
        marker = "###" if depth == 0 else "####"
        indent = "> " * depth
        head = f"{marker} {'↳ ' * min(depth, 3)}{cname}"
        if ctier:
            head += f" ({ctier})"
        head += f" — {c.get('postDate', '')}"
        nvotes = votes_of(c)
        if nvotes:
            head += f" — {nvotes} votes"
        lines += ["", head, ""]
        body = (c.get("rawMarkdown") or c.get("content") or "").rstrip()
        for line in (body or "*(empty)*").split("\n"):
            lines.append(f"{indent}{line}" if indent else line)
    return "\n".join(lines) + "\n"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    THREADS.mkdir(parents=True, exist_ok=True)

    kg = Kaggle()
    kg.bootstrap()
    print("session bootstrapped")

    comp = kg.rpc("competitions.CompetitionService/GetCompetition",
                  {"competitionName": COMPETITION})
    forum_id = comp["forumId"]
    (RAW / "competition.json").write_text(json.dumps(comp, indent=2))
    print(f"competition {comp['id']} -> forumId {forum_id}")

    # --- topic list, paginated -------------------------------------------------
    topics, page = [], 1
    while True:
        resp = kg.rpc("discussions.DiscussionsService/GetTopicListByForumId",
                      {"forumId": forum_id, "page": page})
        batch = resp.get("topics") or []
        total = resp.get("count", 0)
        topics.extend(batch)
        print(f"  page {page}: {len(batch)} topics ({len(topics)}/{total})")
        if len(batch) < PAGE_SIZE or len(topics) >= total:
            break
        page += 1
        time.sleep(0.4)

    # the same topic can repeat across pages if the ordering shifts mid-walk
    seen, deduped = set(), []
    for t in topics:
        if t["id"] not in seen:
            seen.add(t["id"])
            deduped.append(t)
    topics = deduped
    (RAW / "topics_index.json").write_text(json.dumps(topics, indent=2))
    print(f"{len(topics)} unique topics")

    # --- each thread in full ---------------------------------------------------
    index_rows, total_comments = [], 0
    for i, meta in enumerate(topics, 1):
        tid = meta["id"]
        resp = kg.rpc("discussions.DiscussionsService/GetForumTopicById",
                      {"forumTopicId": tid, "includeComments": True})
        topic = resp.get("forumTopic") or {}
        (RAW / f"topic_{tid}.json").write_text(json.dumps(resp, indent=2))

        n = len(flatten(topic.get("comments") or []))
        total_comments += n
        title = topic.get("name") or meta.get("title") or "untitled"
        path = THREADS / f"{tid}-{slugify(title)}.md"
        path.write_text(render(topic, meta))

        name, _ = author_of(topic)
        index_rows.append({
            "id": tid,
            "title": title,
            "author": name,
            "votes": meta.get("votes", 0),
            "comments": n,
            "posted": (topic.get("postDate") or "")[:10],
            "pinned": bool(topic.get("isStickied")),
            "file": path.name,
            "url": f"{BASE}{topic.get('url', '')}",
        })
        print(f"  [{i}/{len(topics)}] {tid} {n:>3} comments  {title[:60]}")
        time.sleep(0.3)

    (RAW / "index_rows.json").write_text(json.dumps(index_rows, indent=2))

    # --- public leaderboard, for calibrating our own scores against the field ---
    try:
        lb = kg.rpc("competitions.LeaderboardService/GetLeaderboard",
                    {"competitionId": comp["id"],
                     "leaderboardType": "LEADERBOARD_TYPE_PUBLIC"})
        (RAW / "leaderboard.json").write_text(json.dumps(lb, indent=2))
        scores = [float(r["displayScore"]) for r in lb.get("publicLeaderboard", [])
                  if r.get("displayScore")]
        print(f"leaderboard: {len(scores)} teams, top {scores[0] if scores else 'n/a'}")
    except Exception as exc:  # the archive is still valid without it
        print(f"  ! leaderboard fetch failed ({exc}); continuing", file=sys.stderr)

    # --- table of contents -----------------------------------------------------
    rows = sorted(index_rows, key=lambda r: (not r["pinned"], -r["votes"]))
    md = [
        "# Discussion archive — Biohub Cell Tracking During Development",
        "",
        f"Scraped {time.strftime('%Y-%m-%d')} from <{BASE}/competitions/{COMPETITION}/discussion>.",
        f"**{len(rows)} threads, {total_comments} comments.** Raw API responses in `raw/`,",
        "rendered threads in `threads/`. Regenerate with `python discussions/scrape_discussions.py`.",
        "",
        "The analysis of all of it is in [`01-scouting-report.md`](01-scouting-report.md) —",
        "read that first; this page is the table of contents.",
        "",
        "| Votes | Comments | Date | Title | Author |",
        "|------:|---------:|------|-------|--------|",
    ]
    for r in rows:
        pin = "📌 " if r["pinned"] else ""
        title = r["title"].replace("|", "\\|")
        md.append(f"| {r['votes']} | {r['comments']} | {r['posted']} | "
                  f"{pin}[{title}](threads/{r['file']}) | {r['author']} |")
    (HERE / "00-index.md").write_text("\n".join(md) + "\n")

    print(f"\ndone: {len(rows)} threads, {total_comments} comments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
