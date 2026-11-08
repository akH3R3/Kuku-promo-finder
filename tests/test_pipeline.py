"""Offline tests: a fake LLM that makes realistic mistakes, to prove the checker + repair loop work.
Run:  python -m tests.test_pipeline "data/<show folder>"
"""
import json, sys
from pathlib import Path
from promo_finder.parser import parse_show
from promo_finder.candidates import find_candidates
from promo_finder.llm import extract_json


class FakeLLM:
    model = "fake"
    def __init__(self, first, second):
        self.replies, self.prompts = [first, second], []
    def complete_json(self, system, user):
        self.prompts.append(user)
        raw = json.dumps(self.replies.pop(0))
        return json.loads(raw), raw


def cand(title, segs, hook, cliff, standalone=4):
    return {"title": title, "segments": [{"start_event": a, "end_event": b} for a, b in segs],
            "hook_event": hook, "hook": "h", "cliffhanger_event": cliff, "cliffhanger": "c",
            "why": "w", "context_needed": "none", "scores": {"hook": 4, "cliffhanger": 4, "standalone": standalone}}


def main(show):
    eps, _ = parse_show(Path(show))
    ep1 = eps[0]
    first = {"candidates": [
        cand("good: sister steals groom", [("E01-027", "E01-047")], "E01-027", "E01-047"),
        cand("invented id", [("E01-027", "E01-999")], "E01-027", "E01-999"),
        cand("too long", [("E01-001", "E01-080")], "E01-001", "E01-080"),
        cand("hook buried", [("E01-001", "E01-030")], "E01-027", "E01-029"),
    ]}
    second = {"candidates": [
        cand("good: sister steals groom", [("E01-027", "E01-047")], "E01-027", "E01-047"),
        cand("fixed: two-segment cut", [("E01-008", "E01-017"), ("E01-041", "E01-047")], "E01-008", "E01-047"),
    ]}
    llm = FakeLLM(first, second)
    promos, log = find_candidates(ep1, llm)

    for p in promos:
        segs = " + ".join(f"{s.start:.1f}-{s.end:.1f}" for s in p.segments)
        print(f"[{p.status:8}] repaired={p.repaired!s:5} {p.title:28} {segs:22} {p.duration:5.1f}s  {p.problems}")

    ok = [p for p in promos if p.status == "ok"]
    assert len(llm.prompts) == 2, "repair round should have been triggered"
    assert "E01-999" in llm.prompts[1] and "outside 20-60s" in llm.prompts[1], "violations sent back to model"
    assert {p.title for p in ok} == {"good: sister steals groom", "fixed: two-segment cut"}
    assert len([p for p in promos if p.status == "rejected"]) == 3, "original failures kept as evidence"
    # no cut lands mid-dialogue
    for p in ok:
        for s in p.segments:
            for e in ep1.events:
                if e.kind == "dialogue":
                    assert not (e.start < s.start < e.end) and not (e.start < s.end < e.end)
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 2} hope it helps') == {"a": 2}
    print("\nAll checks passed.")


if __name__ == "__main__":
    main(sys.argv[1])
