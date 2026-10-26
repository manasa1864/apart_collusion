# AI Cartel Sting

Apart Research hackathon, Track 3: *Collusion Carriers and Countermeasures*.

Five LLM suppliers of an essential medicine can quietly restrict supply and cause a shortage. We run a **sting**: a scripted undercover mole offers
them a pact, a regulator offers leniency to the first reporter, and a monitor reads a share of the messages. We measure whether AI cartels have
*loyalty* (who keeps the pact when tempted, who snitches, who leaks secrets) and what breaks them: leniency, surveillance, or telling the agents they are watched.

The full design is in [PLAN.md](PLAN.md). This repository contains the reduced version that fit the free-tier API budget, and its results.

## Results (80 real episodes)

4 cells (S3, S4, S5, PC) x 10 seeds x 2 models (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`, Groq free tier), 12 rounds each.
All 1,394 agent messages were labelled by an LLM judge. Full report, tables and figures: [analysis/apart_judged/](analysis/apart_judged/report.md).

| Hypothesis | Estimate [95% CI] | Verdict |
|---|---|---|
| H0 manipulation check: collusion when told to (PC vs S3) | 2.29 [1.90, 2.71] | supported |
| H6 contagion: hazard of a first deviation after another agent deviated | 2.48 [1.62, 3.60] | supported |
| H2 leniency reports, 100% vs 25% message reading | 0.05 [0.00, 0.15] | not significant |
| H3a compliance when told "you are monitored" (S5 vs S3) | -0.01 [-0.08, 0.05] | not significant |
| H3b coded share of collusive talk, told monitored vs not | -0.06 [-0.33, 0.23] | inconclusive (10 + 13 episodes) |
| H4 loyalty is one trait (PC1 share of variance) | 0.51 [0.47, 0.54] | not supported: loyalty splits into several components |
| H15 stubbornness vs loyalty (exploratory) | -0.26 [-0.36, -0.16] | opposite to the hypothesis |

The other hypotheses in PLAN.md (H1, H5, H7-H14, H16) need cells that were not run and show as "not run" in the report.

### Limitations

- Reduced design: 2 models, 4 cells, 10 seeds per model per cell. Differences between cells have wide intervals.
- Message labels come from an LLM judge (`gpt-oss-20b`, one of the two studied models) that was **not validated against human labels**. The keyword-only
  analysis ([analysis/apart/](analysis/apart/report.md)) gives the same verdicts for the main hypotheses.
- Only 2 of the 7 tested hypotheses are supported after Holm correction; the others are null or exploratory results.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
python -m tests.test_metrics && python -m tests.test_webs && python -m tests.test_netformulas && python -m tests.test_llmio && python -m tests.test_analysis && python -m tests.test_apart
```

## Reproduce

Analyse the included results (no API key needed):

```
python analyze_cartel.py --results results/apart/results.jsonl --labels results/apart/labels.jsonl --out analysis/apart_judged
```

Check the pipeline on synthetic agents with planted effects (no API key):

```
python run_cartel.py --backend sim --design all --webs all --seeds 30 --out results/sim
python analyze_cartel.py --results results/sim/results.jsonl --out analysis/sim
```

Re-run the real study (Groq free tier; stops at the daily token cap, run the same command again to continue):

```
export GROQ_API_KEY=...             # PowerShell:  $env:GROQ_API_KEY="..."
python run_cartel.py --preset apart --seeds 10 --out results/apart --models openai/gpt-oss-120b,openai/gpt-oss-20b --wait-reset
python -m cartel.judge label --results results/apart/results.jsonl --out results/apart/labels.jsonl --model openai/gpt-oss-20b
python analyze_cartel.py --results results/apart/results.jsonl --labels results/apart/labels.jsonl --out analysis/apart_judged
```

**API keys.** Keys are read from environment variables and never stored in files. Extra keys `GROQ_API_KEY_2` ... `GROQ_API_KEY_9` are used when the first
is rejected or out of quota; the free quota belongs to the Groq account, so extra keys only help if they come from different accounts.
Use `--dry-run` to see how many episodes and days a run needs.

## Layout

| Path | What it is |
|---|---|
| `run_cartel.py` | resumable runner (budgets, rate limits, key fallbacks); `--preset apart` is the reduced design |
| `analyze_cartel.py` | analysis and figures from `results.jsonl` (+ optional judge labels) |
| `cartel/` | game (`econ.py`, `env.py`), agents (`llm_agent.py`, `simagent.py`), API client (`llmio.py`), metrics, statistics, figures, LLM judge (`judge.py`) |
| `tests/` | unit tests |
| `results/apart/` | raw episodes (`results.jsonl`) and judge labels (`labels.jsonl`) |
| `analysis/apart_judged/` | report, tables and figures used for the write-up (judge labels) |
| `analysis/apart/` | same analysis with the keyword classifier, for comparison |
| `label_hand.py` | optional: label a sample of messages by hand to validate the judge (`python -m cartel.judge sample`, then `agree`) |

## Related work

1. Fish, Gonczarowski, Shorrer, *Algorithmic Collusion by LLMs* - https://arxiv.org/abs/2404.00806
2. Motwani et al., *Secret Collusion among AI Agents* - https://arxiv.org/abs/2402.07510
3. Mathew et al., *Hidden in Plain Text* - https://arxiv.org/abs/2410.03768
4. Agrawal et al., *Evaluating LLM Agent Collusion in Double Auctions* - https://arxiv.org/abs/2507.01413
5. Hammond et al., *Multi-Agent Risks from Advanced AI* - https://arxiv.org/abs/2502.14143
6. Greenblatt et al., *AI Control* - https://arxiv.org/abs/2312.06942
