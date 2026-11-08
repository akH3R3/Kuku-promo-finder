"""Offline: simulate Google's failure modes.   Run:  python -m tests.test_llm_errors"""
import io, json, os, socket, tempfile, urllib.error, urllib.request
from pathlib import Path
from unittest import mock
from promo_finder import llm as L

OK = {"candidates": [{"content": {"parts": [{"text": '{"candidates": []}'}]}, "finishReason": "STOP"}]}
PER_MINUTE = {"error": {"code": 429, "message": "Resource exhausted", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "10"}]},
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}]}}
PER_DAY = {"error": {"code": 429, "message": "You exceeded your current quota", "details": [
    {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}]}}

def http_err(code, body):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(json.dumps(body).encode()))

def run(script, **kw):
    script, sent, sleeps = list(script), [], []
    def urlopen(req, timeout=0):
        sent.append(json.loads(req.data))
        step = script.pop(0)
        if isinstance(step, Exception):
            raise step
        return io.BytesIO(json.dumps(step).encode())
    os.environ["GEMINI_API_KEY"] = "k"
    with mock.patch.object(urllib.request, "urlopen", urlopen), mock.patch.object(L.time, "sleep", sleeps.append):
        llm = L.LLM("gemini", "gemini-3.8-flash", min_interval_s=0, cache_dir=Path(tempfile.mkdtemp()), **kw)
        try:
            return llm.complete_json("sys", "user"), sent, sleeps
        except Exception as e:
            return e, sent, sleeps

r, sent, sleeps = run([socket.timeout("The read operation timed out"), OK])
print("timeout then ok  :", r[0] if isinstance(r, tuple) else r, "| waits", sleeps); assert isinstance(r, tuple)

r, sent, sleeps = run([http_err(503, {"error": {"message": "overloaded"}}), http_err(429, PER_MINUTE), OK])
print("503, 429/min, ok :", r[0], "| waits", sleeps); assert sleeps == [15, 39]

r, sent, sleeps = run([http_err(429, PER_DAY), OK])
print("429 per day      :", type(r).__name__, "->", r, "| calls made:", len(sent))
assert isinstance(r, L.QuotaExhausted) and len(sent) == 1, "must stop at once, not retry"

r, sent, _ = run([OK], thinking="low")
print("thinking=low     :", sent[0]["generationConfig"]); assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}

# cache keys: unchanged when no settings (existing caches stay valid), different with thinking
k = lambda **kw: L.LLM("gemini", "m", **kw)
import hashlib
old = hashlib.sha256("gemini|m|s|u".encode()).hexdigest()[:24]
with mock.patch.object(L.Path, "exists", lambda self: print("   cache file:", self.name) or False), \
     mock.patch.object(L.LLM, "_call", side_effect=RuntimeError("stop")):
    for kw in ({}, {"thinking": "low"}):
        try: k(min_interval_s=0, max_retries=1, **kw).complete_json("s", "u")
        except Exception: pass
print("old-style key    :", old)
print("\nAll LLM error-handling checks passed.")
