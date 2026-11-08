"""
Usage:
    python parse_show.py "data/<show folder>" [--aliases aliases.json] [--show-episode 1]

Parses every episode, writes outputs/parsed_show.json, prints a sanity report.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

from promo_finder.parser import parse_show, render_for_llm, save_parsed, fmt_ts


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("show_dir", type=Path)
    ap.add_argument("--aliases", type=Path, default=None, help="JSON map of speaker aliases")
    ap.add_argument("--out", type=Path, default=Path("outputs/parsed_show.json"))
    ap.add_argument("--show-episode", type=int, default=None, help="print the LLM view of one episode")
    args = ap.parse_args()

    episodes, mapping = parse_show(args.show_dir, args.aliases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_parsed(episodes, mapping, args.out)

    all_events = [e for ep in episodes for e in ep.events]
    kinds = Counter(e.kind for e in all_events)
    speakers = Counter(e.speaker for e in all_events if e.speaker)
    durations = [ep.duration for ep in episodes]
    warnings = [(ep.number, w) for ep in episodes for w in ep.warnings]

    print(f"Episodes parsed : {len(episodes)}")
    print(f"Scenes          : {sum(len(ep.scenes) for ep in episodes)}")
    print(f"Events          : {len(all_events)}  {dict(kinds)}")
    print(f"Runtime         : {fmt_ts(sum(durations))} total, "
          f"{fmt_ts(min(durations))}-{fmt_ts(max(durations))} per episode")
    print(f"Speaker merges  : {mapping}")
    print(f"Top speakers    : {speakers.most_common(8)}")
    print(f"Warnings        : {len(warnings)}")
    for n, w in warnings[:15]:
        print(f"   ep{n}: {w}")
    print(f"Saved to        : {args.out}")

    if args.show_episode:
        ep = next(e for e in episodes if e.number == args.show_episode)
        print("\n" + "=" * 70 + "\n" + render_for_llm(ep))


if __name__ == "__main__":
    main()
