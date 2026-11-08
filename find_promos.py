"""
Find promo candidates in every episode of a show.

Examples:
  python find_promos.py SHOW_DIR --model gemini-3.8-flash --thinking low --batch 5
  python find_promos.py SHOW_DIR --episodes 1-5
  python find_promos.py SHOW_DIR --provider baseline            # no LLM, comparison run
  python find_promos.py --list-models --provider gemini
  # Any OpenAI-compatible API (e.g. Groq), set OPENAI_API_KEY and OPENAI_BASE_URL in .env:
  python find_promos.py SHOW_DIR --provider openai --model llama-3.3-70b-versatile
"""
import argparse
import sys
import csv
import json
import time
from pathlib import Path

from promo_finder.parser import parse_show, fmt_ts
from promo_finder.candidates import find_candidates, find_candidates_batch, baseline_candidates
from promo_finder.llm import LLM, load_dotenv, QuotaExhausted


def episode_range(spec: str | None, available: list[int]) -> list[int]:
    if not spec:
        return available
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return [n for n in out if n in available]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles
    ap = argparse.ArgumentParser()
    ap.add_argument("show_dir", type=Path, nargs="?")
    ap.add_argument("--provider", default="gemini", choices=["gemini", "openai", "baseline"])
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--episodes", help="e.g. 1-5 or 1,4,9-12 (default: all)")
    ap.add_argument("--aliases", type=Path, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-repair", action="store_true", help="skip the automatic repair round")
    ap.add_argument("--retries", type=int, default=5,
                    help="tries per request; on a tiny daily quota use 2 (failed tries count against it)")
    ap.add_argument("--rpm", type=float, default=9, help="max requests per minute (free tier: keep low)")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--thinking", default=None, choices=["minimal", "low", "medium", "high"],
                    help="Gemini 3 thinking level; 'low' is much faster (gemini-3.8-flash rejects 'minimal')")
    ap.add_argument("--batch", type=int, default=1,
                    help="episodes per request (5 fits a 50-episode show in ~12 requests on a 20/day free tier)")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()
    load_dotenv()

    if args.list_models:
        for m in LLM(args.provider, args.model).list_models():
            print(m)
        return
    if not args.show_dir:
        ap.error("show_dir is required")

    episodes, _ = parse_show(args.show_dir, args.aliases)
    wanted = episode_range(args.episodes, [ep.number for ep in episodes])
    model_name = "baseline" if args.provider == "baseline" else args.model
    run = Path("outputs/runs") / (args.run_name or f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}")
    (run / "raw").mkdir(parents=True, exist_ok=True)

    llm = None if args.provider == "baseline" else \
        LLM(args.provider, args.model, min_interval_s=60 / args.rpm, temperature=args.temperature,
            thinking=args.thinking, max_retries=max(1, args.retries))

    all_promos, stopped, failed = [], None, []
    todo = [ep for ep in episodes if ep.number in wanted]
    size = 1 if llm is None else max(1, args.batch)
    groups = [todo[i:i + size] for i in range(0, len(todo), size)]
    for group in groups:
        nums = [ep.number for ep in group]
        label = f"Episode {nums[0]:>2}" if len(nums) == 1 else f"Episodes {nums[0]:>2}-{nums[-1]:<2}"
        print(f"{label} ...", end=" ", flush=True)
        name = f"ep{nums[0]:02d}" if size == 1 else f"batch_ep{nums[0]:02d}-{nums[-1]:02d}"
        try:
            if llm is None:
                promos, log = baseline_candidates(group[0]), {"episode": nums[0], "model": "baseline"}
            elif size == 1:
                promos, log = find_candidates(group[0], llm, repair=not args.no_repair)
            else:
                promos, log = find_candidates_batch(group, llm, repair=not args.no_repair)
        except QuotaExhausted as e:
            print(f"\n\nSTOPPED: daily quota used up.\n  {e}\n"
                  "  Finished episodes are cached: run the SAME command again after the quota resets\n"
                  "  (midnight Pacific time) and it resumes here. Or switch --model (each model has its own quota).")
            stopped = nums[0]
            break
        except Exception as e:  # one bad batch must not kill a weekly run
            failed += nums
            print(f"FAILED: {e}")
            (run / "raw" / f"{name}.json").write_text(json.dumps({"error": str(e)}), encoding="utf-8")
            continue
        (run / "raw" / f"{name}.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        ok = [p for p in promos if p.status == "ok"]
        per_ep = ", ".join(f"ep{n}:{sum(p.episode == n for p in ok)}" for n in nums) if len(nums) > 1 else ""
        print(f"{len(ok)} ok, {len(promos) - len(ok)} rejected {('(' + per_ep + ')') if per_ep else ''}")
        all_promos += promos

    rows = [p.to_row() for p in all_promos]
    # Partial re-runs (e.g. --episodes 31-50) ADD to an existing run instead of overwriting it:
    # keep earlier rows for every episode this invocation did not successfully process.
    done_now = {ep.number for ep in todo} - set(failed) - (
        {ep.number for ep in todo if stopped is not None and ep.number >= stopped})
    prev_file = run / "candidates.json"
    if prev_file.exists():
        prev = json.loads(prev_file.read_text(encoding="utf-8"))
        kept = [r for r in prev if r["episode"] not in done_now]
        if kept:
            print(f"Keeping {len(kept)} earlier candidates from episodes not re-run in this invocation")
        rows = sorted(kept + rows, key=lambda r: r["episode"])
    (run / "candidates.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(run / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "episode", "title", "segments", "duration_s", "hook", "cliffhanger", "why",
                    "hook_score", "cliff_score", "standalone_score", "repaired", "problems", "warnings"])
        for r in rows:
            segs = " + ".join(f"Ep{s['episode']} {s['start_ts']}-{s['end_ts']}" for s in r["segments"])
            sc = r["scores"]
            w.writerow([r["status"], r["episode"], r["title"], segs, r["duration"], r["hook"],
                        r["cliffhanger"], r["why"], sc.get("hook"), sc.get("cliffhanger"),
                        sc.get("standalone"), r["repaired"], " | ".join(r["problems"]),
                        " | ".join(r["warnings"])])

    ok = [r for r in rows if r["status"] == "ok"]
    covered = sorted({r["episode"] for r in rows} | done_now)
    summary = {
        "model": model_name, "episodes_requested_now": len(wanted),
        "episodes_covered_in_file": len(covered), "candidates_total": len(rows),
        "candidates_ok": len(ok), "candidates_rejected": len(rows) - len(ok),
        "repaired_ok": sum(r["repaired"] for r in ok),
        "thinking": args.thinking, "batch": args.batch, "failed_episodes": failed,
        "stopped_at_episode_quota": stopped,
        "complete": stopped is None and not failed,
    }
    if llm:
        u = llm.usage
        summary.update({"api_calls": u.calls, "cache_hits": u.cached, "input_tokens": u.input_tokens,
                        "output_tokens": u.output_tokens, "api_seconds": round(u.seconds, 1)})
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nOutputs in {run}/")


if __name__ == "__main__":
    main()