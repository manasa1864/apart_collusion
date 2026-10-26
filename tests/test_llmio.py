"""Tests for the LLM plumbing: rate limiting, budgets, cache-based resume, parsing, runner. Run: python -m tests.test_llmio"""
import json
import os
import tempfile

from cartel import bots, env, llmio, logio, webs, metrics as M
from cartel.fakemodel import fake_transport
from cartel.llm_agent import LLMAgent, classify, parse_action


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def sleep(self, s): self.t += s


def test_rate_limiter_rpm_and_tpm():
    c = Clock(); rl = llmio.RateLimiter(rpm=3, tpm=1000, clock=c, sleep=c.sleep)
    assert rl.wait(100) == 0 and rl.wait(100) == 0 and rl.wait(100) == 0
    s = rl.wait(100)                                       # 4th request inside a minute must wait ~60 s
    assert 59 <= s <= 62
    c2 = Clock(); r2 = llmio.RateLimiter(rpm=100, tpm=1000, clock=c2, sleep=c2.sleep)
    r2.wait(600); s2 = r2.wait(600)                        # tokens, not requests, are binding
    assert 59 <= s2 <= 62


def test_ledger_and_daily_budget():
    d = tempfile.mkdtemp()
    led = llmio.Ledger(os.path.join(d, "l.json"))
    llm = llmio.LLM("m", fake_transport, ledger=led, daily_calls=3, limiter=llmio.RateLimiter(1000, 10 ** 9))
    for i in range(3):
        llm.complete("s", f"u{i}", key=f"k{i}")
    try:
        llm.complete("s", "u9", key="k9"); assert False
    except llmio.BudgetExceeded:
        pass
    assert llmio.Ledger(os.path.join(d, "l.json")).today("m")["calls"] == 3      # persisted across processes
    assert llm.complete("s", "u0", key="k0") == llm.complete("s", "u0", key="k0")  # cached answers are free even over budget


def test_429_backoff_and_daily_limit_message():
    slept, n = [], {"c": 0}
    def flaky(p):
        n["c"] += 1
        if n["c"] <= 2:
            e = RuntimeError("HTTP 429"); e.status, e.retry_after = 429, "7"; raise e
        return fake_transport(p)
    llm = llmio.LLM("m", flaky, sleep=slept.append, limiter=llmio.RateLimiter(1000, 10 ** 9))
    llm.complete("s", "u", key="a")
    assert slept == [7.0, 7.0] and llm.calls_made == 1
    def daily(p):
        e = RuntimeError("HTTP 429: Rate limit reached ... tokens per day (TPD)"); e.status, e.retry_after = 429, None; raise e
    llm2 = llmio.LLM("m", daily, sleep=lambda s: None, limiter=llmio.RateLimiter(1000, 10 ** 9))
    try:
        llm2.complete("s", "u"); assert False
    except llmio.BudgetExceeded:
        pass


def _episode(llm, eid="E|s0"):
    agents = [LLMAgent(llm, eid) for _ in range(5)]
    return env.run_episode(agents, bots.ScriptedMole("direct"), T=16, **webs.make_web("complete", "all", "none", exposure=0.25, seed=0))


def test_resume_gives_identical_episode_and_no_repeat_calls():
    big = llmio.RateLimiter(10 ** 6, 10 ** 9)
    ref_llm = llmio.LLM("m", fake_transport, limiter=big)
    ref = _episode(ref_llm)
    total = ref_llm.calls_made
    assert total == 16 * 5
    d = tempfile.mkdtemp(); cache = os.path.join(d, "c.jsonl")
    a = llmio.LLM("m", fake_transport, cache=llmio.Cache(cache), limiter=big, session_calls=30)
    try:
        _episode(a); assert False
    except llmio.BudgetExceeded:
        pass
    assert a.calls_made == 30
    b = llmio.LLM("m", fake_transport, cache=llmio.Cache(cache), limiter=big)         # new process, same cache file
    resumed = _episode(b)
    assert b.calls_made == total - 30 and b.cache_hits == 30                           # only the missing calls are paid for
    assert [r["q"] for r in resumed["rounds"]] == [r["q"] for r in ref["rounds"]]
    assert [r["msgs"] for r in resumed["rounds"]] == [r["msgs"] for r in ref["rounds"]]


def test_cache_survives_truncated_last_line():
    d = tempfile.mkdtemp(); p = os.path.join(d, "c.jsonl")
    open(p, "w").write(json.dumps(dict(key="a", text="x")) + "\n" + '{"key": "b", "te')
    assert llmio.Cache(p).get("a") == "x" and llmio.Cache(p).get("b") is None


def test_parse_and_fallbacks():
    assert parse_action('noise {"quantity": 3} tail')["quantity"] == 3
    assert parse_action("not json") is None
    class Garbage:
        def __init__(s, out): s.out = out
        def complete(s, system, user, key=None): return s.out
    obs = dict(t=0, id=0, n=5, canaries=["a", "b", "c", "d"], leniency=False, last=None, q_last=None, inbox=[], events={})
    ag = LLMAgent(Garbage("lol"), "e"); r = ag.act(obs)
    assert abs(r["q"] - 20 / 6) < 1e-9 and ag.parse_fail == 2                          # falls back to the competitive quantity
    obs["leniency"] = True
    rep = {"quantity": 2, "messages": [{"to": "7", "text": "I swear to hold quota"}],
           "report": {"claims": [{"about": "Supplier 2", "round": 1, "fragment": "x"}]}}
    r = LLMAgent(Garbage(json.dumps(rep)), "e").act(obs)
    assert r["msgs"][0]["to"] == "all" and r["msgs"][0]["oath"] and r["msgs"][0]["intent"] == "pact"
    assert r["report"]["claims"][0] == dict(about=2, round=0, fragment="x")            # 1-indexed round -> 0-indexed
    obs["leniency"] = False
    assert "report" not in LLMAgent(Garbage(json.dumps(rep)), "e").act(obs)            # no reporting before leniency opens


def test_classify():
    assert classify("FALCON3 is our code-name", "M", ["FALCON3"])[0] == "leak"
    assert classify("count me in", "all")[0] == "join"
    assert classify("let's keep to the quota of 2", "all")[0] == "pact"
    assert classify("I promise I will never tell", "all") == ("oath", True)
    assert classify("how is demand this quarter?", "all") == ("chat", False)


def test_bad_claims_count_as_untruthful_not_crash():
    log = _episode(llmio.LLM("m", fake_transport, limiter=llmio.RateLimiter(10 ** 6, 10 ** 9)))
    log["rounds"][12]["reports"] = [dict(frm=0, t=12, claims=[dict(about=None, round=None, fragment="x"), dict(about=2, round=99, fragment="y")])]
    assert M.report_truth(log, 0) == 0.0


def test_logio_roundtrip_keeps_metrics():
    log = _episode(llmio.LLM("m", fake_transport, limiter=llmio.RateLimiter(10 ** 6, 10 ** 9)))
    back = logio.loads(logio.dumps(log))
    assert M.score_episode(back)["per_agent"][0]["compliance"] == M.score_episode(log)["per_agent"][0]["compliance"]
    assert set(back["rounds"][0]["q"]) == {0, 1, 2, 3, 4, "M"} or set(back["rounds"][0]["q"]) == {0, 1, 2, 3, 4}


def test_runner_stop_and_resume_end_to_end():
    import run_cartel
    d = tempfile.mkdtemp()
    argv = ["--fake", "--webs", "W1,W2", "--exposures", "0.25", "--seeds", "2", "--rounds", "10", "--schedule", "short", "--out", d, "--session-calls", "100"]
    a = run_cartel.parser().parse_args(argv); a.reasoning_effort = None
    assert run_cartel.run(a) == "stopped"                   # 4 episodes x 50 calls = 200 > 100: stops after two
    done1 = run_cartel.load_done(os.path.join(d, "results.jsonl"))
    assert len(done1) == 2
    a2 = run_cartel.parser().parse_args(argv[:-2]); a2.reasoning_effort = None
    assert run_cartel.run(a2) == "done"
    assert len(run_cartel.load_done(os.path.join(d, "results.jsonl"))) == 4
    ledger = json.load(open(os.path.join(d, "ledger.json")))
    assert sum(m["calls"] for day in ledger.values() for m in day.values()) == 200    # no call paid twice
    assert run_cartel.run(a2) == "done"                      # nothing left: no new calls
    # seed-major order: the first two finished episodes are seed 0 of both webs (balanced cells)
    assert {[x for x in e.split("|") if x.startswith("s") and x[1:].isdigit()][0] for e in done1} == {"s0"}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print("ok  ", f.__name__)
    print(f"\n{len(fns)} tests passed")
