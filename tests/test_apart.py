"""Tests for the reduced 'apart' design: preset, short schedule, several models with separate quotas. Run: python -m tests.test_apart"""
import json
import os
import tempfile

import run_cartel
from cartel import analysis as A
from cartel.fakemodel import fake_transport


def args(*extra):
    a = run_cartel.parser().parse_args(list(extra))
    a.reasoning_effort = None
    return a


def test_preset_plan():
    a = run_cartel.apply_preset(args("--preset", "apart", "--seeds", "2"))
    plan = run_cartel.episode_plan(a)
    assert a.rounds == 12 and a.schedule == "short" and a.history == 2
    assert len(plan) == 2 * 4 * 3 and len({p["episode_id"] for p in plan}) == len(plan)
    assert {p["cell"] for p in plan} == {"S3", "S4", "S5", "PC"}
    assert [p["model"] for p in plan[:3]] == run_cartel.GROQ_MODELS          # models interleaved: balanced if stopped early
    assert all(p["seed"] == 0 for p in plan[:12])                            # seed-major


def test_explicit_args_beat_preset():
    a = run_cartel.apply_preset(args("--preset", "apart", "--rounds", "14", "--cells", "S3"))
    assert a.rounds == 14 and a.cells == "S3" and a.schedule == "short"


def test_short_schedule_opens_leniency_in_round_9():
    tmp = tempfile.mkdtemp()
    a = args("--preset", "apart", "--fake", "--models", "m1", "--cells", "S3", "--seeds", "1", "--out", tmp)
    assert run_cartel.run(a) == "done"
    rec = A.load(os.path.join(tmp, "results.jsonl"))[0]
    assert rec["log"]["T"] == 12 and rec["log"]["events"]["t_L"] == 8 and rec["schedule"] == "short"


def test_standard_schedule_refuses_too_few_rounds():
    try:
        run_cartel.run(args("--fake", "--rounds", "12", "--out", tempfile.mkdtemp()))
        assert False
    except SystemExit:
        pass


def test_one_model_out_of_quota_others_continue():
    tmp = tempfile.mkdtemp()
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
    for n in range(2, 10):                                                 # real fallback keys in the environment would be called over the network
        os.environ.pop(f"GROQ_API_KEY_{n}", None)

    def transport(payload):
        if payload["model"] == "A":
            e = RuntimeError("HTTP 429: Rate limit reached for model A on tokens per day (TPD)")
            e.status, e.retry_after = 429, None
            raise e
        return fake_transport(payload)
    a = args("--preset", "apart", "--models", "A,B", "--cells", "S3,PC", "--seeds", "1", "--rpm", "1000000", "--tpm", "1000000000", "--out", tmp)
    a.transport_override = transport
    assert run_cartel.run(a) == "stopped"                                  # A blocked, B finished, nothing else to do
    done = run_cartel.load_done(os.path.join(tmp, "results.jsonl"))
    assert done and all(e.split("|")[6] == "B" for e in done) and len(done) == 2
    led = json.load(open(os.path.join(tmp, "ledger.json")))
    day = next(iter(led.values()))
    assert day.get("A", {}).get("calls", 0) == 0 and day["B"]["calls"] == 2 * 60


def test_session_budget_is_shared_across_models_and_resumes():
    tmp = tempfile.mkdtemp()
    base = ["--preset", "apart", "--fake", "--models", "m1,m2", "--cells", "S3", "--seeds", "1", "--out", tmp]
    assert run_cartel.run(args(*base, "--session-calls", "90")) == "stopped"   # 2 episodes x 60 calls > 90
    assert len(run_cartel.load_done(os.path.join(tmp, "results.jsonl"))) == 1
    assert run_cartel.run(args(*base)) == "done"
    led = json.load(open(os.path.join(tmp, "ledger.json")))
    assert sum(v["calls"] for d in led.values() for v in d.values()) == 120  # nothing paid twice


def test_analysis_handles_reduced_design():
    tmp = tempfile.mkdtemp()
    a = args("--preset", "apart", "--backend", "sim", "--seeds", "4", "--out", tmp)
    assert run_cartel.run(a) == "done"
    D = A.build(A.load(os.path.join(tmp, "results.jsonl")))
    H = A.test_all(D).set_index("H")
    assert H.loc["H0", "verdict"] != "not run" and H.loc["H2", "verdict"] != "not run" and H.loc["H3a", "verdict"] != "not run"
    assert H.loc["H1", "verdict"] == "not run" and H.loc["H5", "verdict"] == "not run"      # those designs are not in the apart run
    assert D["ep"].model.nunique() == 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print("ok  ", f.__name__)
    print(f"\n{len(fns)} tests passed")
