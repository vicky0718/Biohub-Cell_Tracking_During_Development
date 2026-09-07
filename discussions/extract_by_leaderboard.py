"""Pull every forum post written by a currently-top-ranked competitor.

    python discussions/extract_by_leaderboard.py 15        # top 15 teams
    python discussions/extract_by_leaderboard.py 15 --md   # also write the markdown digest

`notes/58` did this for one author picked by reputation (hengck23) and it paid — 43 posts,
two implementable suggestions, and a measurement that reordered a queue. This does it by
*rank* instead, which is a different and better filter: hengck23 is a well-known Kaggler,
but the people at the top of THIS leaderboard are the ones who have solved THIS problem.

The join is `TeamMemberUserNames` from the leaderboard CSV against `authorUserName` on a
topic and `author.userName` on each comment and reply. Team names are display strings and
often differ from usernames ("Vibes & Edges Trade-Off" is five accounts), so the CSV's
username column is the only reliable key — matching on team name would silently miss most
of them.

Posts are printed newest-first with rank, score and thread, because a top-5 competitor's
throwaway remark about what did not work is worth more than a long post from rank 15.
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def leaderboard_users(top_n: int) -> dict[str, tuple[int, str, str]]:
    """username -> (rank, score, team), for every member of the top N teams."""
    rows = list(csv.DictReader((RAW / "leaderboard_full.csv").open()))
    rank_key = list(rows[0].keys())[0]          # BOM-prefixed 'Rank'
    out = {}
    for r in rows[:top_n]:
        for u in (r.get("TeamMemberUserNames") or "").split(","):
            if u.strip():
                out[u.strip().lower()] = (int(r[rank_key]), r["Score"], r["TeamName"])
    return out


def posts_in(topic: dict):
    """Yield (username, display, date, body, kind) for the topic and every comment/reply."""
    t = topic.get("forumTopic") or topic
    title = t.get("name", "")
    body = t.get("rawMarkdown") or t.get("content") or ""
    if body:
        yield (t.get("authorUserName", ""), t.get("authorUserDisplayName", ""),
               t.get("postDate") or t.get("createDate") or "", body, "TOPIC", title)

    def walk(items):
        for c in items or []:
            a = c.get("author") or {}
            # Comment authors carry NO `userName` field -- the username is the profile
            # path in `url` ("/madarshbb"). v1 read a.get("userName") and matched nothing
            # against 339 posts, which looked exactly like "the top 15 never post".
            user = (a.get("userName") or (a.get("url") or "").lstrip("/")).strip()
            b = c.get("rawMarkdown") or c.get("content") or ""
            if b:
                yield (user, a.get("displayName", ""),
                       c.get("postDate") or c.get("createDate") or "", b, "comment", title)
            yield from walk(c.get("replies"))

    yield from walk(t.get("comments"))


def main(argv) -> int:
    top_n = int(argv[0]) if argv and argv[0].isdigit() else 15
    want = leaderboard_users(top_n)
    print(f"top {top_n} teams -> {len(want)} usernames")

    found = []
    for p in sorted(RAW.glob("topic_*.json")):
        try:
            topic = json.loads(p.read_text())
        except Exception:
            continue
        for user, disp, date, body, kind, title in posts_in(topic):
            key = (user or "").lower()
            if key in want:
                rank, score, team = want[key]
                found.append((rank, date, user, disp, score, team, kind, title, body))

    found.sort(key=lambda x: (x[0], x[1]))
    print(f"{len(found)} posts by top-{top_n} competitors\n")
    for rank, date, user, disp, score, team, kind, title, body in found:
        print("=" * 88)
        print(f"RANK {rank}  ({score})  {team}  |  {user}  |  {date[:10]}  |  {kind}")
        print(f"THREAD: {title}")
        print("-" * 88)
        print(body.strip()[:4000])
        print()

    if "--md" in argv:
        out = HERE / "by_leaderboard" / f"top{top_n}.md"
        out.parent.mkdir(exist_ok=True)
        lines = [f"# Forum posts by the current top {top_n} teams\n",
                 f"{len(found)} posts, {len(want)} usernames. Newest snapshot of "
                 f"`leaderboard_full.csv`.\n"]
        for rank, date, user, disp, score, team, kind, title, body in found:
            lines.append(f"\n## rank {rank} — {team} ({score}) — @{user} — {date[:10]}\n")
            lines.append(f"*thread: {title}*\n")
            lines.append(f"> {body.strip()[:4000]}\n".replace("\n", "\n> "))
        out.write_text("\n".join(lines))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
