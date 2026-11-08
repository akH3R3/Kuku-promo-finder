"""
Model-agnostic LLM layer. Swapping models = changing two CLI flags (--provider, --model).

Providers:
  gemini   Google Gemini REST API (key: GEMINI_API_KEY)
  openai   Any OpenAI-compatible API: OpenAI, Groq, OpenRouter, Together, local Ollama...
           (key: OPENAI_API_KEY, base URL: OPENAI_BASE_URL)

Everything uses the Python standard library (urllib), so there is nothing to pip install.
Every call is cached on disk by (provider, model, prompt) so re-runs cost no quota.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader so keys never live in code.
    Tolerates Windows quirks: Notepad saving as '.env.txt', BOMs, PowerShell's UTF-16."""
    if not path.exists() and Path(".env.txt").exists():
        path = Path(".env.txt")
    if path.exists():
        data = path.read_bytes()
        text = data.decode("utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else data.decode("utf-8-sig")
        for line in text.splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def extract_json(text: str):
    """Models sometimes wrap JSON in ```json fences or add a sentence. Pull out the JSON."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


class QuotaExhausted(Exception):
    """Daily/project quota is used up: retrying is pointless, stop the run cleanly."""


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0   # includes "thinking" tokens where the API reports them
    seconds: float = 0.0


@dataclass
class LLM:
    provider: str
    model: str
    min_interval_s: float = 6.5        # ~9 requests/min: stays under free-tier RPM limits
    max_retries: int = 5
    temperature: float | None = None   # None = provider default (Gemini 3 wants 1.0)
    thinking: str | None = None        # Gemini 3 thinkingLevel: minimal|low|medium|high (None = default)
    timeout_s: int = 300
    cache_dir: Path = Path("outputs/cache")
    usage: Usage = field(default_factory=Usage)
    _last_call: float = 0.0

    # ---- public ----
    def complete_json(self, system: str, user: str) -> tuple[dict, str]:
        """Returns (parsed_json, raw_text). Raises after retries are exhausted."""
        settings = ""
        if self.thinking:
            settings += f"|thinking={self.thinking}"
        if self.temperature is not None:
            settings += f"|temp={self.temperature}"
        key = hashlib.sha256(f"{self.provider}|{self.model}{settings}|{system}|{user}".encode()).hexdigest()[:24]
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            raw = json.loads(cache_file.read_text(encoding="utf-8"))["raw"]
            self.usage.cached += 1
            return extract_json(raw), raw

        for attempt in range(self.max_retries):
            wait = self.min_interval_s - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            t0 = time.time()
            try:
                raw = self._call(system, user)
                parsed = extract_json(raw)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")
                if e.code == 429 and _is_daily_quota(body):
                    raise QuotaExhausted(_quota_summary(body)) from e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    backoff = _retry_delay(body) or 15 * (attempt + 1)
                    print(f"    {e.code} from API, retrying in {backoff}s ...")
                    time.sleep(backoff)
                    continue
                raise RuntimeError(f"{self.provider} HTTP {e.code}: {_quota_summary(body)}") from e
            except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
                if attempt < self.max_retries - 1:
                    print(f"    network/timeout ({e.__class__.__name__}), retrying in 20s ...")
                    time.sleep(20)
                    continue
                raise RuntimeError(f"network error after {self.max_retries} tries: {e}") from e
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                if attempt < self.max_retries - 1:
                    print(f"    unparseable response ({e.__class__.__name__}), retrying ...")
                    continue
                raise
            self.usage.calls += 1
            self.usage.seconds += time.time() - t0
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"model": self.model, "raw": raw}, ensure_ascii=False), encoding="utf-8")
            return parsed, raw
        raise RuntimeError("retries exhausted")

    def list_models(self) -> list[str]:
        if self.provider == "gemini":
            data = self._get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                             {"x-goog-api-key": self._key("GEMINI_API_KEY")})
            return sorted(m["name"].removeprefix("models/") for m in data.get("models", [])
                          if "generateContent" in m.get("supportedGenerationMethods", []))
        data = self._get(f"{self._openai_base()}/models",
                         {"Authorization": f"Bearer {self._key('OPENAI_API_KEY')}"})
        return sorted(m["id"] for m in data.get("data", []))

    # ---- providers ----
    def _call(self, system: str, user: str) -> str:
        if self.provider == "gemini":
            return self._gemini(system, user)
        if self.provider == "openai":
            return self._openai(system, user)
        raise ValueError(f"Unknown provider '{self.provider}'")

    def _gemini(self, system: str, user: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        gen_cfg = {"responseMimeType": "application/json"}
        if self.temperature is not None:
            gen_cfg["temperature"] = self.temperature
        if self.thinking:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": self.thinking}
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": gen_cfg,
        }
        data = self._post(url, body, {"x-goog-api-key": self._key("GEMINI_API_KEY")}, self.timeout_s)
        meta = data.get("usageMetadata", {})
        self.usage.input_tokens += meta.get("promptTokenCount", 0)
        self.usage.output_tokens += meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
        cand = data["candidates"][0]
        if cand.get("finishReason") not in (None, "STOP"):
            print(f"    warning: finishReason={cand.get('finishReason')}")
        # Skip "thought" parts some models return; keep only the answer text
        return "".join(p.get("text", "") for p in cand["content"]["parts"] if not p.get("thought"))

    def _openai(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        data = self._post(f"{self._openai_base()}/chat/completions", body,
                          {"Authorization": f"Bearer {self._key('OPENAI_API_KEY')}"}, self.timeout_s)
        u = data.get("usage", {})
        self.usage.input_tokens += u.get("prompt_tokens", 0)
        self.usage.output_tokens += u.get("completion_tokens", 0)
        return data["choices"][0]["message"]["content"]

    # ---- plumbing ----
    @staticmethod
    def _key(name: str) -> str:
        k = os.environ.get(name)
        if not k:
            raise SystemExit(f"Missing {name}. Put it in a .env file:  {name}=your-key")
        return k

    @staticmethod
    def _openai_base() -> str:
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

    @staticmethod
    def _post(url: str, body: dict, headers: dict, timeout: int = 300) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def _get(url: str, headers: dict) -> dict:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())


# ---------- reading rate-limit errors ----------

def _is_daily_quota(body: str) -> bool:
    b = body.lower()
    return "perday" in b or "per day" in b or "per_day" in b


def _retry_delay(body: str) -> int | None:
    """Google puts e.g. "retryDelay": "37s" in 429 errors; honour it (+2s margin)."""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+)(?:\.\d+)?s"', body)
    return int(m.group(1)) + 2 if m else None


def _quota_summary(body: str) -> str:
    """Pull the human-readable message and the quota id out of an error body."""
    try:
        err = json.loads(body).get("error", {})
        ids = re.findall(r'"quotaId"\s*:\s*"([^"]+)"', body)
        vals = re.findall(r'"quotaValue"\s*:\s*"([^"]+)"', body)
        extra = f" [quota: {', '.join(ids)}{' = ' + ', '.join(vals) if vals else ''}]" if ids else ""
        return f"{err.get('message', '')[:400]}{extra}"
    except (json.JSONDecodeError, AttributeError):
        return body[:400]
