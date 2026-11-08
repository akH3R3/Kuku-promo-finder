"""
Rank the candidates from a find_promos.py run, remove duplicates, and build cross-episode montages.

  python rank_promos.py "data/<show>" outputs/runs/full_v1 --model gemini-3.8-flash
  python rank_promos.py "data/<show>" outputs/runs/full_v1 --no-montages

Writes into the run folder: ranked.csv, ranked.json, report.md, rank_raw.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from promo_finder.parser import parse_show, fmt_ts
from promo_finder.llm import LLM, load_dotenv
from promo_finder.ranker import (promo_from_row, dedup_overlaps, build_rank_prompt, check_ranking,
                                 find_montages, seg_text, RANK_SYSTEM)

TOP_FOR_MONTAGE = 12


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("show_dir", type=Path)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--provider", default="gemini", choices=["gemini", "openai"])
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--aliases", type=Path, default=None)
    ap.add_argument("--no-montages", action="store_true")
    ap.add_argument("--thinking", default=None, choices=["minimal", "low", "medium", "high"])
    args = ap.parse_args()
    load_dotenv()

    episodes, _ = parse_show(args.show_dir, args.aliases)
    eps = {e.number: e for e in episodes}
    show_title = args.show_dir.name
    rows = json.loads((args.run_dir / "candidates.json").read_text(encoding="utf-8"))
    ok = [promo_from_row(r) for r in rows if r["status"] == "ok"]
    cands = [(f"C{i:03d}", p) for i, p in enumerate(ok, 1)]
    by_id = dict(cands)
    print(f"{len(cands)} valid candidates from {len({p.episode for p in ok})} episodes")

    # 1. code: overlap dedup
    kept, overlap_dropped = dedup_overlaps(cands)
    print(f"Overlap dedup (code): dropped {len(overlap_dropped)}, {len(kept)} remain")

    # 2. LLM: cluster + rank
    llm = LLM(args.provider, args.model, thinking=args.thinking)
    print("Clustering and ranking (1 LLM call) ...")
    data, raw = llm.complete_json(RANK_SYSTEM, build_rank_prompt(show_title, episodes, kept)
                                  + "\n\nCluster, rank and tier these clips. JSON only.")
    ranking, story_dupes, issues = check_ranking(data, [cid for cid, _ in kept])
    clusters = data.get("clusters", [])
    print(f"Story-beat dedup (LLM): {len(story_dupes)} duplicates folded into {len(ranking)} distinct promos")
    for i in issues:
        print(f"  checker: {i}")

    # 3+4. montages
    montages, montage_log = [], None
    if not args.no_montages:
        top = [(r["id"], by_id[r["id"]]) for r in ranking[:TOP_FOR_MONTAGE]]
        print(f"Building montages from the top {len(top)} clips (1-2 LLM calls) ...")
        montages, montage_log = find_montages(llm, show_title, eps, top)
        print(f"Montages: {sum(m.status == 'ok' for m in montages)} valid, "
              f"{sum(m.status != 'ok' for m in montages)} rejected")

    # ---------- outputs ----------
    out_rows = []
    for rank, r in enumerate(ranking, 1):
        p = by_id[r["id"]]
        dupes = [d for d, w in story_dupes.items() if w == r["id"]]
        out_rows.append({"rank": rank, "id": r["id"], "tier": r["tier"], "type": "single",
                         "rank_reason": r["reason"], "same_beat_as": dupes, **p.to_row()})
    for i, m in enumerate(montages, 1):
        out_rows.append({"rank": None, "id": f"M{i:02d}", "tier": "montage" if m.status == "ok" else "rejected",
                         "type": "montage", "rank_reason": "", "same_beat_as": [], **m.to_row()})

    (args.run_dir / "ranked.json").write_text(json.dumps(
        {"ranking": out_rows, "overlap_dropped": overlap_dropped, "story_duplicates": story_dupes,
         "clusters": clusters, "checker_issues": issues}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.run_dir / "rank_raw.json").write_text(json.dumps(
        {"model": args.model, "rank_raw_response": raw, "montage": montage_log}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    with open(args.run_dir / "ranked.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "tier", "type", "segments", "duration_s", "hook", "cliffhanger", "why",
                    "rank_reason", "same_beat_as", "warnings", "problems"])
        for r in out_rows:
            segs = " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in r["segments"])
            w.writerow([r["rank"], r["id"], r["tier"], r["type"], segs, r["duration"], r["hook"], r["cliffhanger"],
                        r["why"], r["rank_reason"], ", ".join(r["same_beat_as"]), " | ".join(r["warnings"]),
                        " | ".join(r["problems"])])

    write_report(args.run_dir / "report.md", show_title, len(cands), overlap_dropped, story_dupes,
                 issues, out_rows, by_id)
    print(f"\nWrote ranked.csv, ranked.json, report.md, rank_raw.json in {args.run_dir}/")


def write_report(path, show, n_cands, overlap_dropped, story_dupes, issues, rows, by_id):
    singles = [r for r in rows if r["type"] == "single"]
    montages = [r for r in rows if r["type"] == "montage"]

    def segs(r):
        return " + ".join(f"Ep{s['episode']} {s['start_ts']}–{s['end_ts']}" for s in r["segments"])

    L = [f"# Promo report: {show}", "",
         f"{n_cands} valid candidates → {len(overlap_dropped)} dropped as overlapping clips (code) → "
         f"{len(story_dupes)} folded as the same story beat (LLM) → **{len(singles)} distinct promos**, "
         f"plus {sum(m['tier'] == 'montage' for m in montages)} montages.", ""]
    for tier, title in (("post_first", "Post first"), ("strong", "Strong"), ("filler", "Filler")):
        group = [r for r in singles if r["tier"] == tier]
        if not group:
            continue
        L += [f"## {title} ({len(group)})", ""]
        for r in group:
            L += [f"**#{r['rank']} {r['title']}** ({r['id']}) · {segs(r)} · {r['duration']:.0f}s",
                  f"- Hook: {r['hook']}", f"- Cliffhanger: {r['cliffhanger']}", f"- Why: {r['why']}",
                  f"- Rank reason: {r['rank_reason']}"]
            if r["same_beat_as"]:
                L.append(f"- Same beat as: {', '.join(r['same_beat_as'])} (dropped)")
            if r["warnings"]:
                L.append(f"- ⚠ Editor check: {'; '.join(r['warnings'])}")
            L.append("")
    if montages:
        L += ["## Cross-episode montages", ""]
        for r in montages:
            status = "" if r["tier"] == "montage" else f" — REJECTED: {'; '.join(r['problems'])}"
            L += [f"**{r['id']} {r['title']}** · {segs(r)} · {r['duration']:.0f}s{status}",
                  f"- Hook: {r['hook']}", f"- Cliffhanger: {r['cliffhanger']}", f"- Why: {r['why']}", ""]
    L += ["## Removed duplicates", ""]
    for cid, why in overlap_dropped.items():
        L.append(f"- {cid} ({by_id[cid].title}): {why}")
    for loser, winner in story_dupes.items():
        L.append(f"- {loser} ({by_id[loser].title}): same story beat as {winner}")
    if issues:
        L += ["", "## Checker notes on the ranking call", ""] + [f"- {i}" for i in issues]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
