"""
Screenplay parser: turns Kuku microdrama script .txt files into structured events.

Design rule: this layer is PURE CODE, no LLM. Every timestamp the tool ever outputs
comes from here. The LLM later only refers to event IDs, so it can't invent times.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------- Data model ----------

@dataclass
class Event:
    id: str            # globally unique, e.g. "E01-014" = episode 1, event 14
    episode: int
    scene: int
    kind: str          # "dialogue" | "action" | "camera" | "sfx"
    start: float       # seconds from episode start
    end: float
    text: str
    speaker: str | None = None       # normalised name, dialogue only
    speaker_raw: str | None = None   # name exactly as written in the script
    emotion: str | None = None       # e.g. "Cold, Commanding"
    delivery: str | None = None      # "off-screen" / "voice-over" if marked


@dataclass
class Scene:
    number: int
    heading: str
    start: float
    end: float


@dataclass
class Episode:
    number: int
    source_file: str
    logline: str
    continuity: str
    duration: float
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------- Helpers ----------

TS = r"(\d{1,2}):(\d{2}(?:\.\d+)?)"
# Header like **NAME (Emotion):** or **NAME (O.S.) (Emotion):** -- any number of parentheticals
BLOCK_HEADER = re.compile(r"^\*\*(?P<label>[^*(]+?)\s*(?P<parens>(?:\([^)]*\)\s*)*):\*\*\s*$")
PAREN = re.compile(r"\(([^)]*)\)")
DELIVERY = {"O.S.": "off-screen", "OFF-SCREEN": "off-screen", "V.O.": "voice-over",
            "VOICE-OVER": "voice-over", "CONT'D": None}
TIME_LINE = re.compile(rf"^Start:\s*\({TS}\)\s*\|\s*End:\s*\({TS}\)\s*$", re.I)
FIELD = re.compile(r"^(?P<key>[A-Za-z ]+?):\s*(?P<val>.*)$")
SCENE_LINE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)

NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}
TITLE_PREFIXES = ("LADY ", "LORD ", "SIR ", "MISS ", "MR. ", "MRS. ")


def to_seconds(mm: str, ss: str) -> float:
    return round(int(mm) * 60 + float(ss), 2)


def fmt_ts(seconds: float) -> str:
    """Format seconds back to the script's own MM:SS.ss style."""
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


# ---------- Parsing one episode ----------

def parse_episode(path: Path) -> Episode:
    lines = path.read_text(encoding="utf-8").splitlines()

    meta = {"EPISODE NUMBER": "", "EPISODE LOGLINE": "", "CONTINUITY NOTES": ""}
    scenes: list[Scene] = []
    events: list[Event] = []
    warnings: list[str] = []

    cur_scene: dict | None = None
    pending: dict | None = None   # block header seen, waiting for time + text
    i = 0

    def flush_pending():
        nonlocal pending
        if pending and pending.get("start") is not None:
            text = " ".join(pending["text"]).strip().strip('"').strip()
            if text:
                events.append(pending_to_event(pending, text))
            else:
                warnings.append(f"Empty block '{pending['label']}' at {fmt_ts(pending['start'])}")
        pending = None

    def pending_to_event(p: dict, text: str) -> Event:
        label = p["label"].strip()
        kind = NON_DIALOGUE.get(label.upper(), "dialogue")
        return Event(
            id="",  # filled in after parsing
            episode=0,
            scene=cur_scene["number"] if cur_scene else 0,
            kind=kind,
            start=p["start"],
            end=p["end"],
            text=text,
            speaker=label.upper() if kind == "dialogue" else None,
            speaker_raw=label if kind == "dialogue" else None,
            emotion=(p["emotion"] or None) if kind == "dialogue" else None,
            delivery=p.get("delivery") if kind == "dialogue" else None,
        )

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("--- SCRIPT"):
            flush_pending()
        elif m := SCENE_LINE.match(stripped):
            flush_pending()
            cur_scene = {"number": int(m.group(1)), "heading": "", "start": None, "end": None}
            scenes.append(cur_scene)  # type: ignore[arg-type]
        elif m := BLOCK_HEADER.match(stripped):
            flush_pending()
            delivery, emotions = None, []
            for p in PAREN.findall(m.group("parens") or ""):
                key = p.strip().upper()
                if key in DELIVERY:
                    delivery = DELIVERY[key] or delivery
                else:
                    emotions.append(p.strip())
            pending = {"label": m.group("label"), "emotion": ", ".join(emotions) or None,
                       "delivery": delivery, "start": None, "end": None, "text": []}
        elif stripped.startswith("**"):
            # Fail loudly: an unrecognised header would otherwise merge into the previous event
            flush_pending()
            warnings.append(f"Unrecognised block header, skipped: '{stripped}'")
        elif m := TIME_LINE.match(stripped):
            if pending is None or pending["start"] is not None:
                warnings.append(f"Timestamp with no block header: '{stripped}'")
            else:
                pending["start"] = to_seconds(m.group(1), m.group(2))
                pending["end"] = to_seconds(m.group(3), m.group(4))
        elif (m := FIELD.match(stripped)) and pending is None:
            # Case-insensitive: Episode 6 writes "SCENE HEADING:" instead of "Scene heading:"
            key, val = m.group("key").strip().upper(), m.group("val").strip()
            if key in meta:
                meta[key] = val
            elif cur_scene is not None and key == "SCENE HEADING":
                cur_scene["heading"] = val
            elif cur_scene is not None and key in ("SCENE START", "SCENE END"):
                t = re.match(TS, val)
                if t:
                    cur_scene["start" if key == "SCENE START" else "end"] = to_seconds(*t.groups())
            # "Actions and dialogue:" and anything else is just ignored
        elif stripped and pending is not None and pending["start"] is not None:
            pending["text"].append(stripped)
        i += 1
    flush_pending()

    ep_num = int(re.search(r"\d+", meta["EPISODE NUMBER"]).group()) if meta["EPISODE NUMBER"] else \
        int(re.search(r"episode-(\d+)", path.name).group(1))

    # Assign stable IDs and sort by time (ties keep script order)
    events.sort(key=lambda e: e.start)
    for n, e in enumerate(events, 1):
        e.episode = ep_num
        e.id = f"E{ep_num:02d}-{n:03d}"

    scene_objs = [Scene(s["number"], s["heading"], s["start"] or 0.0, s["end"] or 0.0) for s in scenes]
    duration = max([s.end for s in scene_objs] + [e.end for e in events] + [0.0])

    # Sanity checks: record, don't crash
    raw_blocks = sum(1 for l in lines if TIME_LINE.match(l.strip()))
    if raw_blocks != len(events):
        warnings.append(f"Block count mismatch: {raw_blocks} timed blocks in file, {len(events)} parsed")
    for e in events:
        if e.end < e.start:
            warnings.append(f"{e.id}: end before start")
        if e.end > duration + 0.5:
            warnings.append(f"{e.id}: ends after episode end")

    return Episode(ep_num, path.name, meta["EPISODE LOGLINE"], meta["CONTINUITY NOTES"],
                   duration, scene_objs, events, warnings)


# ---------- Show-level: speaker name normalisation ----------

def normalise_speakers(episodes: list[Episode], aliases: dict[str, str] | None = None) -> dict[str, str]:
    """
    Scripts call the same person different things ("LADY JULIANA" / "JULIANA").
    Rule 1: strip a title prefix only if the bare name is also used elsewhere in the show.
    Rule 2: apply a per-show alias file for cases rules can't catch ("THE FORMER GROOM" -> "GERALD").
    Returns the mapping that was applied, so it's auditable.
    """
    aliases = {k.upper(): v.upper() for k, v in (aliases or {}).items()}
    names = {e.speaker for ep in episodes for e in ep.events if e.speaker}
    mapping: dict[str, str] = {}
    for n in names:
        for p in TITLE_PREFIXES:
            if n.startswith(p) and n[len(p):] in names:
                mapping[n] = n[len(p):]
    mapping.update(aliases)
    for ep in episodes:
        for e in ep.events:
            if e.speaker in mapping:
                e.speaker = mapping[e.speaker]
    return mapping


def parse_show(folder: Path, alias_file: Path | None = None) -> tuple[list[Episode], dict[str, str]]:
    files = sorted(folder.glob("episode-*.txt"))
    episodes = [parse_episode(f) for f in files]
    episodes.sort(key=lambda ep: ep.number)
    aliases = json.loads(alias_file.read_text()) if alias_file and alias_file.exists() else {}
    mapping = normalise_speakers(episodes, aliases)
    return episodes, mapping


# ---------- Rendering for the LLM ----------

def render_for_llm(ep: Episode, include_camera: bool = True) -> str:
    """
    Compact, ID-tagged transcript. The LLM sees event IDs and must answer in IDs.
    Example line:  [E01-007 00:11.30-00:13.10] MATRIARCH (Cold, Commanding): "You should turn around, Juliana."
    """
    out = [f"EPISODE {ep.number} ({fmt_ts(ep.duration)} long)", f"LOGLINE: {ep.logline}", ""]
    for e in ep.events:
        if e.kind == "camera" and not include_camera:
            continue
        stamp = f"[{e.id} {fmt_ts(e.start)}-{fmt_ts(e.end)}]"
        if e.kind == "dialogue":
            emo = f" ({e.emotion})" if e.emotion else ""
            dl = f" [{e.delivery}]" if e.delivery else ""
            out.append(f'{stamp} {e.speaker}{dl}{emo}: "{e.text}"')
        else:
            out.append(f"{stamp} {e.kind.upper()}: {e.text}")
    return "\n".join(out)


def save_parsed(episodes: list[Episode], mapping: dict[str, str], path: Path) -> None:
    payload = {"speaker_mapping": mapping, "episodes": [asdict(ep) for ep in episodes]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
