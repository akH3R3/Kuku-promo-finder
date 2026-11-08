"""
Stage 2: turn per-episode candidates into a ranked posting list.

  1. Code  : drop near-duplicate clips inside an episode (time overlap).
  2. LLM   : cluster candidates that tell the same story beat across episodes
             (microdramas replay the last cliffhanger at the start of the next episode),
             keep the best of each, and rank the winners by comparing them to each other.
             The candidates' own 1-5 scores are NOT used: they proved inflated.
  3. LLM   : propose cross-episode montages from the top clips' lines (event IDs only).
  4. Code  : validate montages with the same checker as single clips; one repair round.
"""
from __future__ import annotations

import json
import re

from .candidates import Promo, Segment, validate, _from_json, merge_repairs
from .parser import Episode, fmt_ts

OVERLAP_DUPLICATE = 0.5   # two clips sharing >50% of the shorter one are duplicates


# ---------------- helpers ----------------

def promo_from_row(r: dict) -> Promo:
    segs = [Segment(s["episode"], s["start_event"], s["end_event"], s["start"], s["end"]) for s in r["segments"]]
    keep = {k: r[k] for k in ("episode", "title", "hook_event", "hook", "cliffhanger_event", "cliffhanger",
                              "why", "context_needed", "scores", "source", "status", "repaired",
                              "problems", "warnings", "opening_lines")}
    return Promo(segments=segs, promo_type=r.get("promo_type", "single"), **keep)


def overlap(p: Promo, q: Promo) -> float:
    return sum(max(0.0, min(a.end, b.end) - max(a.start, b.start))
               for a in p.segments for b in q.segments if a.episode == b.episode)


def seg_text(p: Promo) -> str:
    return " + ".join(f"Ep{s.episode} {fmt_ts(s.start)}-{fmt_ts(s.end)}" for s in p.segments)


# ---------------- 1. code dedup ----------------

def dedup_overlaps(cands: list[tuple[str, Promo]]) -> tuple[list[tuple[str, Promo]], dict[str, str]]:
    """Within an episode, if two clips overlap by more than half of the shorter one, keep the one
    the model listed first (it was asked to list best first). Returns (kept, {dropped_id: reason})."""
    kept, dropped = [], {}
    for cid, p in cands:
        dup = next((kid for kid, k in kept
                    if overlap(p, k) > OVERLAP_DUPLICATE * min(p.duration, k.duration)), None)
        if dup:
            dropped[cid] = f"overlaps {dup} by {overlap(p, dict(kept)[dup]):.0f}s"
        else:
            kept.append((cid, p))
    return kept, dropped


# ---------------- 2. LLM cluster + rank ----------------

RANK_SYSTEM = """You are head of user acquisition for a short vertical drama app (ReelShort/DramaBox style).
You receive every promo clip an assistant found in one show, plus each episode's logline and continuity notes.
Each clip is 20-60s and will run as a paid/organic ad on TikTok, Reels and Shorts to bring in NEW viewers.

Do three things:
1. CLUSTER: put clips that show the same story beat in one cluster. Watch for recaps: microdramas often replay
   the previous episode's cliffhanger at the start of the next episode, so two clips from adjacent episodes can be
   the same moment. A unique clip is a cluster of one. Choose the best clip of each cluster.
2. RANK the cluster winners in the order you would post them. Compare clips with each other. IGNORE any scores;
   judge from the hook, cliffhanger and context. Criteria, most important first:
   a. Scroll-stopping power of the first 3 seconds for someone who has never heard of the show.
   b. Cliffhanger strength: how badly the viewer needs the next moment.
   c. Clarity with zero context.
   d. Funnel fit: a viewer who taps the ad starts at episode 1, so early-episode moments convert more easily.
      Avoid spoiling twists that go beyond the show's title premise.
3. TIER each ranked clip: "post_first" (at most 5), "strong", or "filler".

Every clip ID must appear in exactly one cluster. "ranking" lists only cluster winners, best first.
Return ONLY JSON:
{"clusters": [{"members": ["C001", "C004"], "best": "C001", "beat": "short name of the story beat"}],
 "ranking": [{"id": "C001", "tier": "post_first", "reason": "one comparative line: why it sits here"}]}"""


def build_rank_prompt(show_title: str, episodes: list[Episode], cands: list[tuple[str, Promo]]) -> str:
    lines = [f"SHOW: {show_title}", "", "EPISODES:"]
    for ep in episodes:
        lines.append(f"Ep{ep.number}: {ep.logline}")
        if ep.continuity:
            lines.append(f"   continuity: {ep.continuity}")
    lines += ["", f"CLIPS ({len(cands)}):"]
    for cid, p in cands:
        lines.append(f"{cid} | {seg_text(p)} | {p.duration:.0f}s | {p.title}\n"
                     f"   hook: {p.hook}\n   cliffhanger: {p.cliffhanger}\n"
                     f"   context needed: {p.context_needed or 'none'}")
    return "\n".join(lines)


def check_ranking(data: dict, valid_ids: list[str]) -> tuple[list[dict], dict[str, str], list[str]]:
    """Code-side checks on the LLM's ranking. Returns (ranking, {loser_id: winner_id}, issues)."""
    issues, valid = [], set(valid_ids)
    loser_to_winner: dict[str, str] = {}
    seen: set[str] = set()
    for c in data.get("clusters", []):
        members = [m for m in c.get("members", []) if m in valid and m not in seen]
        best = c.get("best") if c.get("best") in members else (members[0] if members else None)
        if c.get("best") not in members and members:
            issues.append(f"cluster '{c.get('beat')}': best '{c.get('best')}' not a member; used {best}")
        seen.update(members)
        for m in members:
            if m != best:
                loser_to_winner[m] = best
    unknown = {c["id"] for c in data.get("ranking", []) if c.get("id") not in valid}
    if unknown:
        issues.append(f"ranking had unknown ids, ignored: {sorted(unknown)}")
    ranking, ranked = [], set()
    for r in data.get("ranking", []):
        cid = r.get("id")
        if cid in valid and cid not in ranked and cid not in loser_to_winner:
            ranking.append({"id": cid, "tier": r.get("tier", "strong"), "reason": r.get("reason", "")})
            ranked.add(cid)
    missing = [c for c in valid_ids if c not in ranked and c not in loser_to_winner]
    if missing:
        issues.append(f"{len(missing)} clips left unranked by the model, appended at the end: {missing}")
        ranking += [{"id": c, "tier": "filler", "reason": "not ranked by model"} for c in missing]
    n_first = sum(r["tier"] == "post_first" for r in ranking)
    if n_first > 5:
        issues.append(f"model marked {n_first} clips post_first; demoted all but the top 5")
        count = 0
        for r in ranking:
            if r["tier"] == "post_first":
                count += 1
                if count > 5:
                    r["tier"] = "strong"
    return ranking, loser_to_winner, issues


# ---------------- 3+4. montages ----------------

MONTAGE_SYSTEM = """You are a promo editor for short vertical dramas. Build MONTAGE promos: one ad that joins
2-4 short moments from DIFFERENT episodes into a mini-story (e.g. humiliation -> rescue -> shocking reveal).

Rules:
- Use ONLY the event IDs listed. A segment runs from start_event to end_event inclusive; both must be from the
  same episode. Segments must be in story order (earlier episodes first).
- Total length 20-60 seconds (use the timestamps shown). Short segments of 4-15s work best.
- The first segment must open on the hook (the hook event within its first 3 seconds).
- The last segment must END on an unresolved cliffhanger beat (cut within ~3s after it).
- A new viewer must follow it with zero context.
- Propose up to 3 montages, best first. Don't propose one if it isn't clearly better than a single clip.

Return ONLY JSON:
{"candidates": [{"title": "...", "segments": [{"start_event": "E..-...", "end_event": "E..-..."}],
  "hook_event": "E..-...", "hook": "...", "cliffhanger_event": "E..-...", "cliffhanger": "...",
  "why": "one line", "context_needed": "none or what's missing",
  "scores": {"hook": 1-5, "cliffhanger": 1-5, "standalone": 1-5}}]}"""


def build_montage_prompt(show_title: str, episodes: dict[int, Episode], top: list[tuple[str, Promo]]) -> str:
    lines = [f"SHOW: {show_title}", "", "STRONGEST MOMENTS (lines inside the top-ranked clips):"]
    for cid, p in top:
        lines.append(f"\n--- {cid}: {p.title} ---")
        for s in p.segments:
            for e in episodes[s.episode].events:
                if s.start <= e.start < s.end and e.kind != "camera":
                    who = f"{e.speaker}: \"{e.text}\"" if e.kind == "dialogue" else f"{e.kind.upper()}: {e.text}"
                    lines.append(f"[{e.id} {fmt_ts(e.start)}-{fmt_ts(e.end)}] {who}")
    return "\n".join(lines)


def find_montages(llm, show_title: str, episodes: dict[int, Episode], top: list[tuple[str, Promo]]):
    user = build_montage_prompt(show_title, episodes, top) + "\n\nBuild the montages. JSON only."
    data, raw = llm.complete_json(MONTAGE_SYSTEM, user)
    log = {"raw_response": raw, "repair_raw_response": None}

    def parse(d):
        out = []
        for c in d.get("candidates", []):
            first_ep = _episode_of(c)
            if first_ep is None:
                continue
            p = _from_json(c, episodes[first_ep], llm.model)
            p.promo_type = "montage"
            validate(p, episodes)
            if p.status == "ok" and len({s.episode for s in p.segments}) < 2:
                p.problems.append("montage uses only one episode")
                p.status = "rejected"
            out.append(p)
        return out

    montages = parse(data)
    broken = [m for m in montages if m.status == "rejected"]
    if broken:
        report = "\n".join(f"- '{m.title}': " + "; ".join(m.problems) for m in broken)
        data2, raw2 = llm.complete_json(MONTAGE_SYSTEM, f"{user}\n\nYour previous answer:\n{raw}\n\n"
                                        f"A checker found these violations:\n{report}\n\n"
                                        "Return JSON with ONLY corrected versions of these broken montages, "
                                        "or an empty list if they cannot be fixed. Do not repeat ones that passed.")
        log["repair_raw_response"] = raw2
        montages = merge_repairs(montages, parse(data2))
    return montages, log


def _episode_of(c: dict) -> int | None:
    segs = c.get("segments") or [{}]
    m = re.match(r"E(\d+)-", str(segs[0].get("start_event", "")))
    return int(m.group(1)) if m else None
