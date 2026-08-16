"""Scrape the full Kaggle competition discussion forum: every topic, every comment.

Kaggle's public pages are JavaScript shells and api/v1 needs credentials, but the SPA's
own internal endpoints (`/api/i/<service>/<Method>`) serve public forum content to an
anonymous session. The requirement is a real session: fetch any page to get the
XSRF-TOKEN / ka_sessionid cookies, then echo the token in `x-xsrf-token`. Without that
pair every call is a bare 400 with no body, which reads like a bad payload and is not.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAR = HERE / "cj.txt"
FORUM_ID = 10656304          # from competitions.CompetitionService/GetCompetition
OUT = HERE / "discussions.json"
BASE = "https://www.kaggle.com/api/i"


def xsrf() -> str:
    for line in JAR.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 7 and parts[5] == "XSRF-TOKEN":
            return parts[6]
    raise SystemExit("no XSRF-TOKEN in the cookie jar — refresh it")


TOKEN = xsrf()


def rpc(method: str, body: dict, tries: int = 4) -> dict:
    for attempt in range(tries):
        p = subprocess.run(
            ["curl", "-sS", "-m", "40", "-b", str(JAR), "-X", "POST",
             f"{BASE}/{method}",
             "-H", "content-type: application/json",
             "-H", f"x-xsrf-token: {TOKEN}",
             "--data", json.dumps(body)],
            capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip():
            try:
                return json.loads(p.stdout)
            except json.JSONDecodeError:
                pass
        time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------- topics
topics, page = [], 1
while True:
    d = rpc("discussions.DiscussionsService/GetTopicListByForumId",
            {"forumId": FORUM_ID, "page": page})
    got = d.get("topics", [])
    if not got:
        break
    known = {t["id"] for t in topics}
    fresh = [t for t in got if t["id"] not in known]
    topics += fresh
    total = int(d.get("count", 0))
    print(f"page {page}: +{len(fresh)} (total {len(topics)}/{total})", flush=True)
    if not fresh or len(topics) >= total:
        break
    page += 1
    time.sleep(0.4)

print(f"\n{len(topics)} topics\n")

# ---------------------------------------------------------------- comments
for i, t in enumerate(topics, 1):
    d = rpc("discussions.DiscussionsService/GetForumMessagesInTopic",
            {"topicId": t["id"]})
    t["comments"] = d.get("comments", [])
    if not t["comments"]:
        # fall back to the by-id route, which returns the topic plus its thread
        d2 = rpc("discussions.DiscussionsService/GetForumTopicById",
                 {"topicId": t["id"]})
        t["comments"] = (d2.get("comments")
                         or d2.get("topic", {}).get("comments", []))
        t["_topic_by_id"] = {k: v for k, v in d2.items() if k != "comments"}
    print(f"[{i:>3}/{len(topics)}] {t.get('commentCount', 0):>3} comments declared, "
          f"{len(t['comments']):>3} fetched  votes={t.get('votes', 0):>3}  "
          f"{t.get('title', '?')[:58]}",
          flush=True)
    time.sleep(0.3)

OUT.write_text(json.dumps({"forum_id": FORUM_ID, "n_topics": len(topics),
                           "topics": topics}, indent=1))
tot_c = sum(len(t["comments"]) for t in topics)
declared = sum(t.get("commentCount", 0) for t in topics)
print(f"\nwrote {OUT}  ({OUT.stat().st_size:,} bytes)")
print(f"{len(topics)} topics, {tot_c} comments fetched, {declared} declared")
if tot_c < declared:
    print(f"!! {declared - tot_c} comments NOT retrieved — thread fetch is incomplete")
