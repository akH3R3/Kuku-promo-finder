"""
Per-episode promo candidate finder.

Division of labour:
  LLM   -> judgement: which moment hooks, which moment leaves you hanging, does it stand alone.
           It answers ONLY with event IDs.
  Code  -> everything checkable: IDs exist, timestamps, 20-60s length, hook in first 3s,
           no cut mid-sentence, segment order. Violations go back to the LLM once for repair;
           anything still broken is kept in the output as REJECTED with the reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .parser import Episode, Event, render_for_llm, fmt_ts

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0          # hook must land within the first 3s of the clip
TAIL_AFTER_CLIFF = 4.0     # clip may run at most 4s past the cliffhanger beat

SYSTEM_PROMPT = """You are a senior promo editor for short vertical drama apps like ReelShort and DramaBox.
You pick 20-60 second clips from an episode that will be posted on TikTok/Instagram/YouTube Shorts to pull in NEW viewers who have never seen the show.

A great promo:
1. HOOK - the first 3 seconds stop a scrolling thumb: a slap, a public humiliation, a shocking or cruel line, a reveal, a threat, a kiss, a betrayal. The clip must START on (or at most 3 seconds before) the hook. Never open on establishing shots, walking, or quiet setup.
2. CLIFFHANGER - the clip ENDS on an unresolved beat: a threat not yet carried out, a question not yet answered, a reveal before the reaction, someone about to act. Cut BEFORE the resolution. Never end on a calm or resolved moment.
3. STANDS ALONE - a stranger understands the situation within seconds (who is wronged, who is cruel, what is at stake) without knowing earlier episodes.

A promo is 1-3 segments from this episode, in story order. Use more than one segment only to cut out a slow middle; the total must be 20-60 seconds.

Rules:
- Refer to moments ONLY by the event IDs in square brackets (e.g. "E07-012"). Never write timestamps.
- A segment runs from start_event to end_event inclusive. Use the timestamps shown to keep the total between 20 and 60 seconds.
- Return 1-3 DISTINCT candidates, best first. Quality over quantity: if the episode has only one strong moment, return one. If it has none (a quiet, transitional episode), return an empty list.
- Score honestly from 1-5. A 5 is rare: reserve it for moments that would stop almost anyone.

Return ONLY this JSON:
{"candidates": [{
  "title": "short internal name",
  "segments": [{"start_event": "E..-...", "end_event": "E..-..."}],
  "hook_event": "E..-...",
  "hook": "what happens or is said in the first 3 seconds",
  "cliffhanger_event": "E..-...",
  "cliffhanger": "the unresolved moment the clip ends on",
  "why": "one line on why this works for a new viewer",
  "context_needed": "what a stranger would NOT understand, or 'none'",
  "scores": {"hook": 1-5, "cliffhanger": 1-5, "standalone": 1-5}
}]}"""


@dataclass
class Segment:
    episode: int
    start_event: str
    end_event: str
    start: float = 0.0
    end: float = 0.0

    @property
    def length(self) -> float:
        return round(self.end - self.start, 2)


@dataclass
class Promo:
    episode: int
    title: str
    segments: list[Segment]
    hook_event: str
    hook: str
    cliffhanger_event: str
    cliffhanger: str
    why: str
    context_needed: str
    scores: dict
    source: str                         # model name or "baseline"
    promo_type: str = "single"          # single | montage
    status: str = "ok"                  # ok | rejected
    repaired: bool = False
    problems: list[str] = field(default_factory=list)   # hard rule violations
    warnings: list[str] = field(default_factory=list)   # soft issues, shown to the editor
    opening_lines: list[str] = field(default_factory=list)  # what's actually on screen first

    @property
    def duration(self) -> float:
        return round(sum(s.length for s in self.segments), 2)

    def to_row(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        d["segments"] = [{**asdict(s), "start_ts": fmt_ts(s.start), "end_ts": fmt_ts(s.end),
                          "length": s.length} for s in self.segments]
        return d


# ---------------- Validation (pure code) ----------------

def _snap(seg: Segment, ep: Episode) -> None:
    """Convert event IDs to times, widening so no line of dialogue is cut mid-sentence."""
    idx = {e.id: e for e in ep.events}
    s, e = idx[seg.start_event], idx[seg.end_event]
    start, end = s.start, max(e.end, s.end)
    for ev in ep.events:
        if ev.kind != "dialogue":
            continue
        if ev.start < start < ev.end:      # clip would open mid-line
            start = ev.start
        if ev.start < end < ev.end:        # clip would cut a line off
            end = ev.end
    seg.start, seg.end = round(start, 2), round(end, 2)


def validate(p: Promo, eps) -> Promo:
    """Check a promo against the script. `eps` is one Episode or {number: Episode} (for montages)."""
    if isinstance(eps, Episode):
        eps = {eps.number: eps}
    p.problems, p.warnings = [], []
    ids = {e.id: e for ep in eps.values() for e in ep.events}

    if not p.segments:
        p.problems.append("no segments")
    for seg in p.segments:
        for ref in (seg.start_event, seg.end_event):
            if ref not in ids:
                p.problems.append(f"unknown event id {ref}")
    for ref, name in ((p.hook_event, "hook_event"), (p.cliffhanger_event, "cliffhanger_event")):
        if ref not in ids:
            p.problems.append(f"{name} {ref} does not exist")
    if p.problems:
        p.status = "rejected"
        return p

    for seg in p.segments:
        a, b = ids[seg.start_event], ids[seg.end_event]
        if a.episode != b.episode:
            p.problems.append(f"segment {seg.start_event}->{seg.end_event} spans two episodes")
            continue
        seg.episode = a.episode
        _snap(seg, eps[a.episode])
        if seg.end <= seg.start:
            p.problems.append(f"segment {seg.start_event}->{seg.end_event} has end before start")
    if p.problems:
        p.status = "rejected"
        return p
    for a, b in zip(p.segments, p.segments[1:]):
        if b.episode < a.episode or (b.episode == a.episode and b.start < a.end):
            p.problems.append("segments overlap or are out of story order")
    p.episode = p.segments[0].episode

    if not MIN_LEN <= p.duration <= MAX_LEN:
        p.problems.append(f"total length {p.duration:.1f}s is outside {MIN_LEN:.0f}-{MAX_LEN:.0f}s")

    first, last = p.segments[0], p.segments[-1]
    hook, cliff = ids[p.hook_event], ids[p.cliffhanger_event]
    if hook.episode != first.episode or not first.start <= hook.start <= first.end:
        p.problems.append(f"hook {p.hook_event} is not inside the first segment")
    elif hook.start - first.start > HOOK_WINDOW:
        p.problems.append(f"hook lands {hook.start - first.start:.1f}s in; must be within {HOOK_WINDOW:.0f}s")
    if cliff.episode != last.episode or not last.start <= cliff.start <= last.end:
        p.problems.append(f"cliffhanger {p.cliffhanger_event} is not inside the last segment")
    elif last.end - cliff.end > TAIL_AFTER_CLIFF:
        p.problems.append(f"clip runs {last.end - cliff.end:.1f}s past the cliffhanger; "
                          f"max {TAIL_AFTER_CLIFF:.0f}s or it may show the resolution")

    # Soft checks: shown to the human editor, don't reject
    ep0 = eps[first.episode]
    opening = [e for e in ep0.events if first.start <= e.start < first.start + HOOK_WINDOW]
    p.opening_lines = [f"{e.kind.upper()}: {e.text[:90]}" for e in opening]
    if opening and all(e.kind == "camera" for e in opening):
        p.warnings.append("first 3s is camera-only (no line, action or sound): hook is purely visual")
    if p.context_needed and p.context_needed.strip().lower() not in ("none", "n/a", ""):
        p.warnings.append(f"needs context: {p.context_needed}")
    if p.scores.get("standalone", 5) <= 2:
        p.warnings.append("low standalone score")

    p.status = "rejected" if p.problems else "ok"
    return p


# ---------------- LLM path ----------------

def _from_json(c: dict, ep: Episode, source: str) -> Promo:
    segs = [Segment(ep.number, str(s.get("start_event", "")).strip(), str(s.get("end_event", "")).strip())
            for s in c.get("segments", [])]
    return Promo(
        episode=ep.number, title=c.get("title", ""), segments=segs,
        hook_event=str(c.get("hook_event", "")).strip(), hook=c.get("hook", ""),
        cliffhanger_event=str(c.get("cliffhanger_event", "")).strip(), cliffhanger=c.get("cliffhanger", ""),
        why=c.get("why", ""), context_needed=c.get("context_needed", ""),
        scores={k: int(v) for k, v in (c.get("scores") or {}).items() if str(v).isdigit()},
        source=source,
    )


def find_candidates(ep: Episode, llm, repair: bool = True) -> tuple[list[Promo], dict]:
    """Returns (promos incl. rejected ones, raw log for the outputs folder)."""
    transcript = render_for_llm(ep)
    user = f"{transcript}\n\nFind the promo candidates in this episode. Return JSON only."
    data, raw = llm.complete_json(SYSTEM_PROMPT, user)
    log = {"episode": ep.number, "model": llm.model, "raw_response": raw, "repair_raw_response": None}

    promos = [validate(_from_json(c, ep, llm.model), ep) for c in data.get("candidates", [])]

    broken = [p for p in promos if p.status == "rejected"]
    if repair and broken:
        report = "\n".join(f"- '{p.title}': " + "; ".join(p.problems) for p in broken)
        repair_user = (f"{user}\n\nYour previous answer was:\n{raw}\n\n"
                       f"An automatic checker found these rule violations:\n{report}\n\n"
                       "Return JSON with ONLY corrected versions of these broken candidates "
                       "(choose different start/end events), or an empty list if one cannot be fixed. "
                       "Do not repeat candidates that passed.")
        data2, raw2 = llm.complete_json(SYSTEM_PROMPT, repair_user)
        log["repair_raw_response"] = raw2
        fixed = [validate(_from_json(c, ep, llm.model), ep) for c in data2.get("candidates", [])]
        promos = merge_repairs(promos, fixed)
    return promos, log


def _signature(p: Promo) -> tuple:
    return tuple((s.start_event, s.end_event) for s in p.segments)


def merge_repairs(originals: list[Promo], fixed: list[Promo]) -> list[Promo]:
    """Valid originals are never lost. Fixed items are added unless identical to a valid original.
    Broken originals stay in the output (status rejected) as evidence."""
    ok = [p for p in originals if p.status == "ok"]
    seen = {_signature(p) for p in ok}
    added = []
    for f in fixed:
        if _signature(f) not in seen:
            f.repaired = True
            added.append(f)
            seen.add(_signature(f))
    broken = [p for p in originals if p.status != "ok"]
    for b in broken:
        b.warnings.append("sent back for repair")
    return ok + added + broken

    return promos, log


# ---------------- Rule-based baseline (no LLM) ----------------

def baseline_candidates(ep: Episode) -> list[Promo]:
    """
    What a rushed editor might do: take the last ~45s of every episode (scripts end on a cliffhanger),
    starting at the nearest line of dialogue. Used only as a comparison point for the LLM.
    """
    dialogue = [e for e in ep.events if e.kind == "dialogue"]
    if not dialogue:
        return []
    target_start = max(0.0, ep.duration - 45.0)
    hook = min(dialogue, key=lambda e: abs(e.start - target_start))
    # Episodes often end on a visual beat after the last line, so the cliffhanger is the last
    # non-camera event (dialogue, action or sound), not necessarily the last line spoken.
    last_beat = [e for e in ep.events if e.kind != "camera"][-1]
    end_ev = ep.events[-1]
    p = Promo(
        episode=ep.number, title="last ~45s", segments=[Segment(ep.number, hook.id, end_ev.id)],
        hook_event=hook.id, hook=hook.text, cliffhanger_event=last_beat.id,
        cliffhanger=last_beat.text, why="baseline: episode ending", context_needed="unknown",
        scores={}, source="baseline",
    )
    return [validate(p, ep)]


# ---------------- Batched path: several episodes per request ----------------
# Free tiers cap requests per DAY (e.g. 20), not tokens, so packing ~5 episodes (~14k tokens) into one
# request cuts a 50-episode show from ~60 requests to ~12. Each episode is still judged on its own.

BATCH_ADDENDUM = """

You will receive SEVERAL episodes. Judge each episode on its own: 0-3 candidates per episode,
each built only from that episode's event IDs (a single promo never mixes episodes).
Add "episode": <number> to every candidate. Return ONLY: {"candidates": [ ... ]}"""


def _episode_from_ids(c: dict) -> int | None:
    import re
    for s in c.get("segments") or []:
        m = re.match(r"E(\d+)-", str(s.get("start_event", "")))
        if m:
            return int(m.group(1))
    return None


def _parse_batch(data: dict, by_num: dict, source: str) -> list[Promo]:
    out = []
    for c in data.get("candidates", []):
        n = _episode_from_ids(c) or c.get("episode")
        try:
            n = int(n)
        except (TypeError, ValueError):
            n = None
        if n not in by_num:
            continue
        p = validate(_from_json(c, by_num[n], source), by_num)
        if p.status == "ok" and len({s.episode for s in p.segments}) > 1:
            p.problems.append("a single promo must come from one episode")
            p.status = "rejected"
        out.append(p)
    return out


def find_candidates_batch(eps: list[Episode], llm, repair: bool = True) -> tuple[list[Promo], dict]:
    by_num = {ep.number: ep for ep in eps}
    transcript = "\n\n".join(render_for_llm(ep) for ep in eps)
    user = (f"{transcript}\n\nFind the promo candidates in EACH of these {len(eps)} episodes "
            f"({', '.join(str(n) for n in by_num)}). Return JSON only.")
    system = SYSTEM_PROMPT + BATCH_ADDENDUM
    data, raw = llm.complete_json(system, user)
    log = {"episodes": list(by_num), "model": llm.model, "raw_response": raw, "repair_raw_response": None}
    promos = _parse_batch(data, by_num, llm.model)

    broken = [p for p in promos if p.status == "rejected"]
    if repair and broken:
        report = "\n".join(f"- Episode {p.episode} '{p.title}': " + "; ".join(p.problems) for p in broken)
        data2, raw2 = llm.complete_json(system, (
            f"{user}\n\nYour previous answer was:\n{raw}\n\nAn automatic checker found these rule violations:\n"
            f"{report}\n\nReturn JSON with ONLY corrected versions of these broken candidates (include "
            "\"episode\"), or an empty list if one cannot be fixed. Do not repeat candidates that passed."))
        log["repair_raw_response"] = raw2
        promos = merge_repairs(promos, _parse_batch(data2, by_num, llm.model))
    return promos, log
