"""Offline: batched requests (several episodes per call).   Run:  python -m tests.test_batch "data/<show>" """
import json, sys, tempfile, os
from pathlib import Path
from unittest import mock
import find_promos

def c(segs, hook, cliff, title, episode=None):
    d = {"title": title, "segments": [{"start_event": a, "end_event": b} for a, b in segs], "hook_event": hook,
         "hook": "h", "cliffhanger_event": cliff, "cliffhanger": "c", "why": "w", "context_needed": "none",
         "scores": {"hook": 4, "cliffhanger": 4, "standalone": 4}}
    if episode: d["episode"] = episode
    return d

class FakeLLM:
    calls = []
    def __init__(self, *a, **k): self.model = "fake"; self.usage = type("U", (), {"calls": 0, "cached": 0,
                                  "input_tokens": 0, "output_tokens": 0, "seconds": 0})()
    def complete_json(self, system, user):
        FakeLLM.calls.append((system, user))
        if "Your previous answer" in user:
            d = {"candidates": [c([("E02-033", "E02-068")], "E02-033", "E02-068", "fixed ep2", 2)]}
        elif "EPISODE 1 " in user:      # batch 1: episodes 1 and 2
            d = {"candidates": [
                c([("E01-022", "E01-052")], "E01-024", "E01-052", "good ep1", 1),
                c([("E02-001", "E02-032")], "E02-001", "E02-032", "good ep2, no episode field"),
                c([("E01-022", "E01-030"), ("E02-001", "E02-010")], "E01-024", "E02-010", "mixes episodes", 1),
                c([("E02-001", "E02-068")], "E02-001", "E02-068", "too long", 2)]}
        else:                           # batch 2: episode 3, nothing promo-worthy
            d = {"candidates": []}
        raw = json.dumps(d); return json.loads(raw), raw

def main(show):
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        show_abs = str(Path(show).resolve()); os.chdir(tmp)
        try:
            with mock.patch.object(find_promos, "LLM", FakeLLM), mock.patch.object(sys, "argv",
                    ["find_promos.py", show_abs, "--episodes", "1-3", "--batch", "2", "--run-name", "t"]):
                find_promos.main()
            rows = json.loads(Path("outputs/runs/t/candidates.json").read_text(encoding="utf-8"))
            raws = sorted(p.name for p in Path("outputs/runs/t/raw").iterdir())
        finally:
            os.chdir(cwd)
    ok = sorted(r["title"] for r in rows if r["status"] == "ok")
    rej = {r["title"]: r["problems"] for r in rows if r["status"] != "ok"}
    print("\nok:", ok, "\nrejected:", rej, "\nraw files:", raws, "\nAPI calls:", len(FakeLLM.calls))
    assert ok == ["fixed ep2", "good ep1", "good ep2, no episode field"]
    assert "a single promo must come from one episode" in rej["mixes episodes"]
    assert len(FakeLLM.calls) == 3, "2 batches + 1 repair"
    assert "SEVERAL episodes" in FakeLLM.calls[0][0]
    print("All batch checks passed.")

if __name__ == "__main__":
    main(sys.argv[1])
