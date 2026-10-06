#!/usr/bin/env python3
"""Archive the solution writeups of the private leaderboard's top N teams.

    python discussions/scrape_writeups.py              # top 25, into discussions/writeups/
    python discussions/scrape_writeups.py --out ../rogii/winning_writeups/biohub_top25
    python discussions/scrape_writeups.py --top 50 --no-images

Kaggle links each team to its writeup through `solutionWriteUpUrl` on the private
leaderboard. The writeup body comes from `discussions.WriteUpsService/GetWriteUpBySlug`, and
its comment thread (often where authors answer the useful questions) from the writeup's forum
topic. Teams in the top N with no writeup are listed with whatever they did publish: forum
topics by a team member and, when Kaggle API credentials are available, their public
notebooks for the competition.

Uses the same anonymous XSRF session as `scrape_discussions.py`.

Writes, under --out (default discussions/writeups/):
    index.md                      index: rank, team, private/public score, link, subtitle
                                  (README.md is left alone: it holds the hand-written digest)
    NN-<slug>.md                  one writeup plus its comments, images pointing at local copies
    images/NN/                    that writeup's images up to 2 MB each (bigger ones stay remote)
    raw/leaderboard_private.csv   every team: private and public rank/score, members, writeup
    raw/writeup_NN.json, raw/topic_<id>.json, raw/forum_topics.json   untouched API responses
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_discussions import (BASE, COMPETITION, PAGE_SIZE, Kaggle,  # noqa: E402
                                author_of, flatten, slugify, votes_of)

HERE = Path(__file__).resolve().parent
IMG_MAX_BYTES = 2 * 1024 * 1024   # larger images (mostly animated GIFs) stay as remote links
# a markdown image URL may itself contain one level of parentheses, e.g. ".../figure(1).png"
IMG_RE = re.compile(r'!\[[^\]]*\]\(((?:[^()\s]|\([^()\s]*\))+)(?:\s+"[^"]*")?\)|<img[^>]+src="([^"]+)"')


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def members(team: dict) -> list[dict]:
    return team.get("teamMembers") or []


def member_line(team: dict) -> str:
    return ", ".join(f"{m.get('displayName', '?')} (@{m.get('userName', '?')})"
                     for m in members(team)) or "?"


def forum_topics(kg: Kaggle, forum_id: int) -> list[dict]:
    topics, page = [], 1
    while True:
        resp = kg.rpc("discussions.DiscussionsService/GetTopicListByForumId",
                      {"forumId": forum_id, "page": page})
        batch = resp.get("topics") or []
        topics += batch
        if len(batch) < PAGE_SIZE or len(topics) >= resp.get("count", 0):
            break
        page += 1
        time.sleep(0.4)
    seen, out = set(), []
    for t in topics:
        if t["id"] not in seen:
            seen.add(t["id"])
            out.append(t)
    return out


def notebooks_by(users: list[str]) -> list[dict]:
    """Public notebooks for this competition by these users (official API; needs credentials)."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not (user and key) and cfg.exists():
        d = json.loads(cfg.read_text())
        user, key = d.get("username"), d.get("key")
    if not (user and key):
        return []
    auth = "Basic " + base64.b64encode(f"{user}:{key}".encode()).decode()
    out = []
    for u in users:
        q = urllib.parse.urlencode({"competition": COMPETITION, "user": u, "pageSize": 50})
        req = urllib.request.Request(f"{BASE}/api/v1/kernels/list?{q}",
                                     headers={"Authorization": auth})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out += json.loads(r.read())
        except Exception as exc:  # a listing failure must not sink the archive
            print(f"    ! notebooks for {u}: {exc}", file=sys.stderr)
        time.sleep(0.3)
    return out


def fetch_images(kg: Kaggle, body: str, dest: Path, rel: str) -> tuple[str, int]:
    """Download every image the markdown references and point the markdown at the copies."""
    urls = []
    for m in IMG_RE.finditer(body):
        u = m.group(1) or m.group(2)
        if u and u.startswith("http") and u not in urls:
            urls.append(u)
    got = 0
    for i, url in enumerate(urls, 1):
        name = urllib.parse.unquote(urllib.parse.urlparse(url).path).rsplit("/", 1)[-1]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[-80:] or "image"
        if "." not in name:
            name += ".png"
        local = dest / f"{i:02d}-{name}"
        try:
            # an <img src> may carry HTML entities ("&amp;"), which a browser decodes first
            with kg.opener.open(html.unescape(url), timeout=120) as r:
                data = r.read(IMG_MAX_BYTES + 1)
            if len(data) > IMG_MAX_BYTES:
                print(f"    - image {i} over {IMG_MAX_BYTES >> 20} MB, left remote")
                continue
            dest.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
            body = body.replace(url, f"{rel}/{local.name}")
            got += 1
        except Exception as exc:
            print(f"    ! image {i}: {exc}", file=sys.stderr)
        time.sleep(0.2)
    return body, got


def render_comments(topic: dict) -> list[str]:
    walked = flatten(topic.get("comments") or [])
    lines = ["", "---", "", f"## Comments ({len(walked)})", ""]
    if not walked:
        lines.append("*(none)*")
    for depth, c in walked:
        name, tier = author_of(c)
        head = f"{'###' if depth == 0 else '####'} {'↳ ' * min(depth, 3)}{name}"
        head += f" ({tier})" if tier else ""
        head += f" — {(c.get('postDate') or '')[:10]}"
        if votes_of(c):
            head += f" — {votes_of(c)} votes"
        lines += ["", head, ""]
        body = (c.get("rawMarkdown") or c.get("content") or "").rstrip() or "*(empty)*"
        prefix = "> " * depth
        lines += [f"{prefix}{line}" if prefix else line for line in body.split("\n")]
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", type=Path, default=HERE / "writeups")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()
    out, raw = a.out, a.out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    kg = Kaggle()
    kg.bootstrap()
    comp = kg.rpc("competitions.CompetitionService/GetCompetition",
                  {"competitionName": COMPETITION})
    lb = kg.rpc("competitions.LeaderboardService/GetLeaderboard",
                {"competitionId": comp["id"], "leaderboardType": "LEADERBOARD_TYPE_PRIVATE"})
    teams = {t["teamId"]: t for t in lb.get("teams") or []}
    pub = {r["teamId"]: r for r in lb.get("publicLeaderboard") or []}
    priv = sorted((r for r in lb.get("privateLeaderboard") or [] if r.get("rank")),
                  key=lambda r: r["rank"])
    print(f"{comp['title']}: {len(priv)} ranked teams on the private leaderboard")

    with open(raw / "leaderboard_private.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["private_rank", "private_score", "medal", "public_rank", "public_score",
                    "team_id", "team_name", "members", "writeup_url"])
        for r in priv:
            t, p = teams.get(r["teamId"], {}), pub.get(r["teamId"], {})
            w.writerow([r["rank"], r.get("displayScore", ""), r.get("medal", ""),
                        p.get("rank", ""), p.get("displayScore", ""), r["teamId"],
                        t.get("teamName", ""), ";".join(m.get("userName", "") for m in members(t)),
                        t.get("solutionWriteUpUrl", "")])

    topics = forum_topics(kg, comp["forumId"])
    (raw / "forum_topics.json").write_text(json.dumps(topics, indent=1))
    print(f"{len(topics)} forum topics")

    index, missing = [], []
    for r in priv[:a.top]:
        rank, t, p = r["rank"], teams[r["teamId"]], pub.get(r["teamId"], {})
        row = {"rank": rank, "team": t.get("teamName", ""), "members": member_line(t),
               "private": r.get("displayScore", ""), "medal": r.get("medal", ""),
               "public": p.get("displayScore", ""), "public_rank": p.get("rank", "")}
        url = t.get("solutionWriteUpUrl") or ""
        if not url:
            users = {m.get("userName", "").lower() for m in members(t)}
            pat = re.compile(rf"\b{rank}(st|nd|rd|th)\b.*\b(place|solution)\b", re.I)
            row["topics"] = [x for x in topics
                             if (x.get("authorUser") or {}).get("url", "").strip("/").lower() in users
                             or pat.search(x.get("title") or "")]
            row["notebooks"] = notebooks_by(sorted(users))
            missing.append(row)
            print(f"  {rank:>3} {row['team']}: no writeup "
                  f"({len(row['topics'])} forum topics, {len(row['notebooks'])} notebooks)")
            continue

        slug = url.rstrip("/").rsplit("/", 1)[-1]
        wu = kg.rpc("discussions.WriteUpsService/GetWriteUpBySlug",
                    {"slug": slug, "competitionName": COMPETITION})
        (raw / f"writeup_{rank:02d}.json").write_text(json.dumps(wu, indent=1))
        topic = {}
        if wu.get("topicId"):
            resp = kg.rpc("discussions.DiscussionsService/GetForumTopicById",
                          {"forumTopicId": wu["topicId"], "includeComments": True})
            topic = resp.get("forumTopic") or {}
            (raw / f"topic_{wu['topicId']}.json").write_text(json.dumps(resp, indent=1))

        body = ((wu.get("message") or {}).get("rawMarkdown") or "").rstrip()
        fname = f"{rank:02d}-{slugify(slug, 70)}.md"
        n_comments = len(flatten(topic.get("comments") or []))
        head = [
            f"# {ordinal(rank)} place — {t.get('teamName', '')}: {wu.get('title', '')}",
            "",
        ]
        if wu.get("subtitle"):
            head += [f"> {wu['subtitle']}", ""]
        head += [
            "| | |",
            "|---|---|",
            f"| Private | rank {rank}, {r.get('displayScore', '')} ({r.get('medal', '').title()}) |",
            f"| Public | rank {p.get('rank', '?')}, {p.get('displayScore', '?')} |",
            f"| Team | {member_line(t)} |",
            f"| Writeup by | {wu.get('authors', '')} |",
            f"| Published | {(wu.get('publishTime') or '')[:10]}"
            f" (updated {(wu.get('updateTime') or '')[:10]}) |",
            f"| Source | {BASE}{wu.get('url', url)} |",
            f"| Comments | {n_comments} |",
            "",
            "---",
            "",
        ]
        text = "\n".join(head) + body + "\n" + "\n".join(render_comments(topic)) + "\n"
        n_img = 0
        if not a.no_images:   # body and comments: authors often answer with a figure
            text, n_img = fetch_images(kg, text, out / "images" / f"{rank:02d}",
                                       f"images/{rank:02d}")
        (out / fname).write_text(text)
        row.update(file=fname, title=wu.get("title", ""), subtitle=wu.get("subtitle", ""),
                   words=len(body.split()), comments=n_comments, images=n_img)
        index.append(row)
        print(f"  {rank:>3} {row['team']}: {row['words']} words, {n_comments} comments, "
              f"{n_img} images")
        time.sleep(0.4)

    md = [
        f"# {comp['title']}: the top {a.top} on the private leaderboard",
        "",
        f"Scraped {time.strftime('%Y-%m-%d')} from "
        f"<{BASE}/competitions/{COMPETITION}/leaderboard> and each team's solution writeup.",
        f"{len(index)} of the top {a.top} published a writeup. Regenerate from the Biohub repo "
        "with `python discussions/scrape_writeups.py --out <this folder>`.",
        "",
        "| Private | Score | Public (rank) | Team | Writeup |",
        "|---:|---|---|---|---|",
    ]
    for row in sorted(index + missing, key=lambda x: x["rank"]):
        pubcol = f"{row['public']} ({row['public_rank']})"
        if row.get("file"):
            label = row["subtitle"] or row["title"]
            link = f"[{label.replace('|', '/')}]({row['file']})"
        else:
            link = "no writeup"
        md.append(f"| {row['rank']} | {row['private']} | {pubcol} | "
                  f"{row['team'].replace('|', '/')} | {link} |")
    if missing:
        md += ["", "## Teams without a writeup", ""]
        for row in missing:
            md.append(f"- **{ordinal(row['rank'])}, {row['team']}** ({row['members']}).")
            for x in row["topics"]:
                md.append(f"  - forum post by a team member: "
                          f"[{x.get('title', '')}]({BASE}{x.get('topicUrl', '')})")
            for k in row["notebooks"]:
                md.append(f"  - notebook: [{k.get('title', '')}]({BASE}/code/{k.get('ref', '')})")
            if not (row["topics"] or row["notebooks"]):
                md.append("  - nothing public found")
    (out / "index.md").write_text("\n".join(md) + "\n")
    print(f"\ndone: {len(index)} writeups, {len(missing)} teams without one -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
