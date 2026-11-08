"""Offline test of the ranking stage on 7 real picks from a Gemini run (episodes 1-3),
with a fake model that makes realistic mistakes.   Run:  python -m tests.test_ranker "data/<show>"
"""
import json, sys, tempfile
from pathlib import Path
from unittest import mock
from promo_finder.parser import parse_show
from promo_finder.candidates import Promo, Segment, validate
from promo_finder import llm as llm_mod
import rank_promos

# (episode, title, start_event, end_event): the 7 valid picks gemini-3.8-flash made for episodes 1-3
REAL_PICKS = [(1, "Altar Swap Betrayal", "E01-022", "E01-052"),
              (1, "Tolerated Daughter's Revenge Awakening", "E01-072", "E01-094"),
              (2, "Left at the Altar", "E02-001", "E02-032"),
              (2, "The Bride Strikes Back", "E02-033", "E02-068"),
              (3, "Stranger's Intervention and Shock Proposal", "E03-070", "E03-116"),
              (3, "Violent Eviction to Lethal Rescue", "E03-046", "E03-074"),
              (3, "Family Defiance to Altar Proposal", "E03-082", "E03-116")]

def build_candidates(eps):
    rows = []
    for n, title, a, b in REAL_PICKS:
        ep = eps[n]
        ids = [e.id for e in ep.events]
        inside = [e for e in ep.events[ids.index(a):ids.index(b) + 1] if e.kind != "camera"]
        p = Promo(n, title, [Segment(n, a, b)], inside[0].id, inside[0].text, inside[-1].id, inside[-1].text,
                  "w", "none", {"hook": 5, "cliffhanger": 5, "standalone": 5}, "gemini-3.8-flash")
        rows.append(validate(p, ep).to_row())
    return rows

class FakeLLM:
    model = "fake"
    def __init__(self, *a, **k): self.calls = []
    def complete_json(self, system, user):
        self.calls.append(user)
        if "head of user acquisition" in system:
            d = {"clusters": [{"members": ["C001", "C003"], "best": "C001", "beat": "jilted at the altar"},
                              {"members": ["C002"], "best": "C002", "beat": "revenge stare"},
                              {"members": ["C004"], "best": "C004", "beat": "bride strikes back"},
                              {"members": ["C005"], "best": "C005", "beat": "punch and proposal"}],
                 "ranking": [{"id": "C005", "tier": "post_first", "reason": "punch in second 1"},
                             {"id": "C001", "tier": "post_first", "reason": "betrayal, zero context"},
                             {"id": "C003", "tier": "post_first", "reason": "should be ignored: a dupe"},
                             {"id": "C099", "tier": "post_first", "reason": "invented id"},
                             {"id": "C002", "tier": "post_first", "reason": "revenge"},
                             {"id": "C004", "tier": "post_first", "reason": "x"},
                             {"id": "C005", "tier": "strong", "reason": "listed twice"}]}
                 # C006 forgotten on purpose
        elif "Your previous answer" not in user:   # first montage call: one good, one broken
            d = {"candidates": [
                {"title": "Jilted to proposed", "segments": [{"start_event": "E01-024", "end_event": "E01-030"},
                                                             {"start_event": "E03-105", "end_event": "E03-116"}],
                 "hook_event": "E01-024", "hook": "h", "cliffhanger_event": "E03-114", "cliffhanger": "c",
                 "why": "w", "context_needed": "none", "scores": {}},
                {"title": "Too long", "segments": [{"start_event": "E01-022", "end_event": "E01-094"},
                                                   {"start_event": "E03-070", "end_event": "E03-116"}],
                 "hook_event": "E01-022", "hook": "h", "cliffhanger_event": "E03-114", "cliffhanger": "c",
                 "why": "w", "context_needed": "none", "scores": {}}]}
        else:                                      # repair: returns only the fix for the broken one
            d = {"candidates": [
                {"title": "Too long (fixed)", "segments": [{"start_event": "E01-072", "end_event": "E01-086"},
                                                         {"start_event": "E03-070", "end_event": "E03-074"}],
                 "hook_event": "E01-072", "hook": "h", "cliffhanger_event": "E03-074", "cliffhanger": "c",
                 "why": "w", "context_needed": "none", "scores": {}}]}
        raw = json.dumps(d)
        return json.loads(raw), raw

def main(show):
    eps = {e.number: e for e in parse_show(Path(show))[0]}
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        (run / "candidates.json").write_text(json.dumps(build_candidates(eps)), encoding="utf-8")
        with mock.patch.object(rank_promos, "LLM", FakeLLM), \
             mock.patch.object(sys, "argv", ["rank_promos.py", show, str(run)]):
            rank_promos.main()
        out = json.loads((run / "ranked.json").read_text(encoding="utf-8"))
        report = (run / "report.md").read_text(encoding="utf-8")

    singles = [r for r in out["ranking"] if r["type"] == "single"]
    order = [r["id"] for r in singles]
    print("\nfinal order:", [(r["id"], r["tier"]) for r in singles])
    print("overlap dropped:", out["overlap_dropped"])
    print("story dupes:", out["story_duplicates"])
    assert "C007" in out["overlap_dropped"], "#7 sits inside #5: code must drop it"
    assert out["story_duplicates"] == {"C003": "C001"}, "Ep2 replay folded into Ep1"
    assert "C003" not in order and "C099" not in order and order.count("C005") == 1
    assert order[-1] == "C006", "forgotten clip appended, not lost"
    assert sum(r["tier"] == "post_first" for r in singles) <= 5
    assert any("unknown ids" in i for i in out["checker_issues"])
    monts = [r for r in out["ranking"] if r["type"] == "montage"]
    print("montages:", [(r["id"], r["title"], r["tier"], r["duration"], r["problems"]) for r in monts])
    assert [r["title"] for r in monts if r["tier"] == "montage"] == ["Jilted to proposed", "Too long (fixed)"], \
        "valid montage must survive the repair round, and the fix must be added"
    assert any(r["tier"] == "rejected" and r["title"] == "Too long" for r in monts), "broken original kept as evidence"
    print("All ranker checks passed.")

if __name__ == "__main__":
    main(sys.argv[1])
