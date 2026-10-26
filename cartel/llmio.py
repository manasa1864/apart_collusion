"""LLM access for the cartel study: rate limiting, a daily budget, a response cache, and resumable runs.

Why a cache gives free resume: env.run_episode is deterministic given the seed and the agents' answers. Every model answer is stored under a
key (episode, round, agent, call kind). If a run stops half-way (daily budget, rate limit, Ctrl-C, crash), the next run replays the episode:
cached answers cost nothing and it continues with real calls exactly where it stopped.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime, timezone


class BudgetExceeded(Exception):
    """Raised when a call/token budget is used up. The runner catches it, saves progress and stops cleanly.
    kind: 'ledger' (our own daily cap, resets at UTC midnight), 'provider' (the API said its daily limit is hit; reset time unknown),
    'session' (the --session-calls cap for this run)."""

    def __init__(self, msg, kind="ledger", model=None):
        super().__init__(msg)
        self.kind, self.model = kind, model


# ----------------------------------------------------------------------------- rate limiter
class RateLimiter:
    """Sliding 60 s window on requests and tokens (free tiers cap both per minute)."""

    def __init__(self, rpm=30, tpm=8000, clock=time.monotonic, sleep=time.sleep):
        self.rpm, self.tpm, self.clock, self.sleep = rpm, tpm, clock, sleep
        self.req, self.tok = deque(), deque()          # (time, amount)

    def _trim(self, now):
        for dq in (self.req, self.tok):
            while dq and now - dq[0][0] >= 60:
                dq.popleft()

    def wait(self, est_tokens):
        """Block until one more request of ~est_tokens fits. Returns seconds slept."""
        slept = 0.0
        est = min(est_tokens, self.tpm)                # a single call larger than the cap would wait forever
        while True:
            now = self.clock()
            self._trim(now)
            used_t = sum(a for _, a in self.tok)
            if len(self.req) < self.rpm and used_t + est <= self.tpm:
                self.req.append((now, 1))
                self.tok.append((now, est))
                return slept
            nxt = min([dq[0][0] for dq in (self.req, self.tok) if dq] or [now])
            d = max(0.5, 60 - (now - nxt) + 0.05)
            self.sleep(d)
            slept += d

    def correct(self, est, actual):
        """Replace the estimate of the latest request with the real token count."""
        if self.tok:
            t, _ = self.tok[-1]
            self.tok[-1] = (t, actual)


# ----------------------------------------------------------------------------- ledger (calls and tokens per UTC day per model)
class Ledger:
    def __init__(self, path, clock=lambda: datetime.now(timezone.utc)):
        self.path, self.clock = path, clock
        self.data = json.load(open(path)) if path and os.path.exists(path) else {}

    def _row(self, model):
        d = self.clock().strftime("%Y-%m-%d")
        return self.data.setdefault(d, {}).setdefault(model, {"calls": 0, "tokens": 0})

    def today(self, model):
        return dict(self._row(model))

    def add(self, model, tokens):
        r = self._row(model)
        r["calls"] += 1
        r["tokens"] += int(tokens)
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f, indent=1)
            for attempt in range(5):
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:      # Windows: another process (editor, sync, antivirus) has the file open
                    time.sleep(0.2 * (attempt + 1))
            with open(self.path, "w") as f:  # last resort: write in place rather than lose the run
                json.dump(self.data, f, indent=1)


# ----------------------------------------------------------------------------- cache (jsonl, append only)
class Cache:
    def __init__(self, path):
        self.path, self.mem, self._lock = path, {}, threading.Lock()
        if path and os.path.exists(path):
            for line in open(path):
                if line.strip():
                    try:
                        r = json.loads(line)
                        self.mem[r["key"]] = r["text"]
                    except json.JSONDecodeError:
                        pass                           # a half-written last line from a crash

    def get(self, key):
        return self.mem.get(key)

    def put(self, key, text):
        with self._lock:
            self.mem[key] = text
            if self.path:
                with open(self.path, "a") as f:
                    f.write(json.dumps(dict(key=key, text=text)) + "\n")


# ----------------------------------------------------------------------------- transport (OpenAI-compatible, e.g. Groq)
def http_transport(base_url, api_key, timeout=180):
    def call(payload):
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json",
                                              "User-Agent": "cartel-study/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            err = RuntimeError(f"HTTP {e.code}: {body}")
            err.status, err.retry_after = e.code, e.headers.get("retry-after")
            raise err
    return call


class RateLimited(Exception):
    pass


class LLM:
    """complete(system, user, key=...) -> text. Cached by key; rate limited; counted against the budgets."""

    def __init__(self, model, transport, cache=None, ledger=None, limiter=None, daily_calls=None, daily_tokens=None,
                 session_calls=None, max_tokens=500, temperature=1.0, reasoning_effort=None, json_mode=True,
                 sleep=time.sleep, retries=10, session_counter=None, fallbacks=None):
        self.model, self.transport, self.cache = model, transport, cache or Cache(None)
        self.ledger, self.limiter = ledger or Ledger(None), limiter or RateLimiter()
        # slots = API keys in order of preference. The first one that is not dead (bad key / provider daily limit) and is under our own
        # daily budget is used. Each extra key has its own rate limiter and its own ledger row ("<model>@key2"); the first keeps the plain model name.
        lim = self.limiter
        self.slots = [dict(name=model, transport=transport, limiter=lim, dead=0.0)]
        for pos, tr in enumerate(fallbacks or [], start=2):
            n, tr = tr if isinstance(tr, tuple) else (pos, tr)      # (key number, transport), or a bare transport numbered by position
            self.slots.append(dict(name=f"{model}@key{n}", transport=tr, dead=0.0,
                                   limiter=RateLimiter(lim.rpm, lim.tpm, clock=lim.clock, sleep=lim.sleep)))
        self.cur = self.slots[0]
        self.daily_calls, self.daily_tokens, self.session_calls = daily_calls, daily_tokens, session_calls
        self.max_tokens, self.temperature, self.reasoning_effort, self.json_mode = max_tokens, temperature, reasoning_effort, json_mode
        self.sleep, self.retries = sleep, retries
        self.calls_made = self.cache_hits = self.slept = 0
        self.session_counter = session_counter if session_counter is not None else {"calls": 0}   # shared across models
        self._lock = threading.Lock()

    def _check_budget(self):
        """Pick the first usable key (sets self.cur) or raise BudgetExceeded."""
        err = None
        for slot in self.slots:
            if slot["dead"] > time.time():
                continue
            t, name = self.ledger.today(slot["name"]), slot["name"]
            if self.daily_calls is not None and t["calls"] >= self.daily_calls:
                err = BudgetExceeded(f"{name}: daily call budget reached ({t['calls']}/{self.daily_calls})", "ledger", self.model)
            elif self.daily_tokens is not None and t["tokens"] >= self.daily_tokens:
                err = BudgetExceeded(f"{name}: daily token budget reached ({t['tokens']}/{self.daily_tokens})", "ledger", self.model)
            else:
                err = None
                self.cur = slot
                break
        else:
            if err is None:                                    # every key is dead
                if all(sl.get("why") == "auth" for sl in self.slots):    # no point sleeping until tomorrow: the keys themselves are bad
                    err = BudgetExceeded(f"every API key was rejected by the provider (HTTP 401/403): the keys are invalid, revoked or expired. "
                                         f"Create new keys, set them, and run again", "session", self.model)
                else:
                    err = BudgetExceeded(f"{self.model}: every API key hit the provider's daily limit or was rejected", "provider", self.model)
        if err is not None:
            raise err
        if self.session_calls is not None and self.session_counter["calls"] >= self.session_calls:
            raise BudgetExceeded(f"session call budget reached ({self.session_counter['calls']}/{self.session_calls})", "session", self.model)

    def complete(self, system, user, key=None, max_tokens=None):
        if key is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit
        with self._lock:
            self._check_budget()
            mt = max_tokens or self.max_tokens
            est = (len(system) + len(user)) // 3 + mt
            slot = self.cur
            self.slept += slot["limiter"].wait(est)
            payload = dict(model=self.model, temperature=self.temperature, max_tokens=mt,
                           messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            if self.json_mode:
                payload["response_format"] = {"type": "json_object"}
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
            resp, last, jv = None, None, 0
            for attempt in range(self.retries):
                try:
                    resp = slot["transport"](payload)
                    break
                except Exception as e:                            # noqa: BLE001
                    last, status = e, getattr(e, "status", None)
                    if status == 429 or (status and status >= 500) or (status is None and isinstance(e, OSError)):   # OSError: timeout / dropped connection
                        ra = getattr(e, "retry_after", None)
                        wait = float(ra) if ra and str(ra).replace(".", "", 1).isdigit() else min(60, 2 ** attempt * 2)
                        m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(e))        # Groq says how long to wait: "in 38.4s" / "in 2m3.4s"
                        if m:
                            wait = max(wait, int(m.group(1) or 0) * 60 + float(m.group(2)) + 1)
                        if status == 429 and "per day" in str(e).lower():
                            slot["dead"], slot["why"] = time.time() + 3600, "daily"   # provider's daily limit on this key: try the next key, else stop
                            try:
                                self._check_budget()
                            except BudgetExceeded as be:
                                raise BudgetExceeded(f"{self.model}: provider says the daily limit is reached: {e}", "provider", self.model) from be
                            slot = self.cur
                            print(f"  [{slot['name']}] previous key hit its daily limit; switching key", flush=True)
                            continue
                        self.sleep(wait)
                        self.slept += wait
                    elif status == 400 and "json_validate_failed" in str(e):
                        # the model returned empty/invalid JSON (a reasoning model can spend all max_tokens thinking).
                        jv += 1
                        if jv == 2:                                # 2nd try: no JSON mode, more room to think
                            payload.pop("response_format", None)
                            payload["max_tokens"] = max(mt, 900)       # qwen's output-tokens-per-minute cap is 1000: 1200 is rejected
                        elif jv >= 3:                              # give up on this call: empty text -> counted as a parse failure
                            resp = {"choices": [{"message": {"content": ""}}], "usage": {}}
                            break
                    elif status == 400 and self.json_mode and "response_format" in str(e):
                        payload.pop("response_format", None)      # model without JSON mode: retry without it
                        self.json_mode = False
                    elif status in (401, 403) and len(self.slots) > 1:
                        slot["dead"], slot["why"] = time.time() + 3600, "auth"  # rejected key: warn loudly and fall back to the next key
                        print(f"  WARNING: key for {slot['name']} was rejected (HTTP {status}); falling back", flush=True)
                        self._check_budget()                       # raises if no key is left
                        slot = self.cur
                    else:
                        raise
            if resp is None:
                raise RateLimited(f"gave up after {self.retries} attempts: {last}")
            text = (resp["choices"][0]["message"].get("content") or "")
            used = (resp.get("usage") or {}).get("total_tokens") or est
            slot["limiter"].correct(est, used)
            self.ledger.add(slot["name"], used)
            self.calls_made += 1
            self.session_counter["calls"] += 1
        if key is not None:
            self.cache.put(key, text)
        return text
