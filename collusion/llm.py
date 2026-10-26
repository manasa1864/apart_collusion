"""Thin LLM client wrapper (Anthropic or any OpenAI-compatible endpoint)."""
import json
import os
import re
import threading
import time

_lock = threading.Lock()
CALLS = 0
MAX_CALLS = None


class BudgetExceeded(Exception):
    pass


def set_max_calls(n):
    global MAX_CALLS
    MAX_CALLS = n


def _count():
    global CALLS
    with _lock:
        CALLS += 1
        if MAX_CALLS and CALLS > MAX_CALLS:
            raise BudgetExceeded(f"call budget of {MAX_CALLS} exceeded")


class Client:
    """model spec is "provider:model" (e.g. "anthropic:claude-haiku-5-5") or just a model name."""

    def __init__(self, spec, default_provider="anthropic"):
        if ":" in spec:
            self.provider, self.model = spec.split(":", 1)
        else:
            self.provider, self.model = default_provider, spec
        if self.provider == "anthropic":
            import anthropic
            self._c = anthropic.Anthropic()
        elif self.provider == "openai":  # also OpenRouter / vLLM via OPENAI_BASE_URL
            from openai import OpenAI
            self._c = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        else:
            raise ValueError(f"unknown provider {self.provider}")

    def complete(self, system, user, max_tokens=400, temperature=1.0):
        _count()
        last = None
        for attempt in range(5):
            try:
                if self.provider == "anthropic":
                    r = self._c.messages.create(
                        model=self.model, max_tokens=max_tokens, temperature=temperature,
                        system=system, messages=[{"role": "user", "content": user}])
                    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                r = self._c.chat.completions.create(
                    model=self.model, max_tokens=max_tokens, temperature=temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return r.choices[0].message.content or ""
            except BudgetExceeded:
                raise
            except Exception as e:  # rate limits, transient errors
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after retries: {last}")


def parse_json(text):
    """Pull the first JSON object out of a model reply; return {} if none parses."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}
