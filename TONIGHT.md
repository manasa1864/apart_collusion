# What to run tonight (Apart version)

Everything below runs from inside the `apart_research` folder. Commands for Windows PowerShell are given where they differ.

## 1. Set up (5 minutes)
```
pip install numpy scipy pandas matplotlib
python -m tests.test_apart          # should end with "7 tests passed"
python -m tests.test_llmio          # "10 tests passed"
```

## 2. Put your Groq key in the terminal (never in a file)
```
export GROQ_API_KEY=gsk_...              # Mac / Linux
$env:GROQ_API_KEY="gsk_..."              # Windows PowerShell
```

## 3. Pilot: one game, one model (about 10 minutes, ~60 calls)
```
python run_cartel.py --preset apart --cells PC --seeds 1 --models openai/gpt-oss-120b --out results/apart
python -m cartel.show results/apart
```
Use `results/apart` as the folder on purpose: the pilot game then counts towards the main run.

Check four things in the `cartel.show` output:
1. **Tokens per call** (top lines). This decides how many games fit before the deadline.
2. **Parse failures** close to 0. If it is above ~5, tell me: the prompt or `--max-tokens` needs adjusting.
3. **Quantities** change between rounds and the agents send messages. If everyone always produces the same number and never talks, tell me.
4. **Positive control** (this pilot is the PC cell): agents should talk about holding output at 2. If they refuse even when told to, that is a finding, but tell me before the long run.

If a model name gives an error like "model not found", Groq has renamed it: run the main step with only the models that work, e.g. `--models openai/gpt-oss-120b,openai/gpt-oss-20b`.

## 4. See how much fits (uses the measured tokens/call)
```
python run_cartel.py --preset apart --seeds 10 --out results/apart --dry-run
```

## 5. Start the overnight run
```
python run_cartel.py --preset apart --seeds 10 --out results/apart --wait-reset
```
- It runs S3, S4, S5 and PC, one game per model in turn, seed by seed, so whatever finishes is balanced.
- When one model's daily quota is used up it continues with the others. When all are used up it sleeps (until UTC midnight = 5:30 am IST for our own cap; it re-checks hourly if Groq itself refused) and carries on.
- Keep the laptop plugged in, awake and online. Mac: put `caffeinate -i ` in front of the command. Windows: Settings > Power > Sleep: Never.
- You can stop it at any time with Ctrl+C and start it again with the same command; nothing is lost or paid twice.
- `--seeds 10` is a ceiling, not a promise: it simply keeps going until you stop it or it finishes.

## 6. Any time from tomorrow: results, figures, report
```
python analyze_cartel.py --results results/apart/results.jsonl --out analysis/apart
```
Open `analysis/apart/report.md`. Figures are in `analysis/apart/figures/` (PNG and PDF), tables in `analysis/apart/tables/` (CSV, Markdown, LaTeX). It works on partial data, so run it as often as you like.

## 7. On the 25th (after the main run has used its quota): label messages
The judge uses tokens from `openai/gpt-oss-20b`, so run it once the games are done or on the last day:
```
python -m cartel.judge label --results results/apart/results.jsonl --out results/apart/labels.jsonl --model openai/gpt-oss-20b
python -m cartel.judge sample --results results/apart/results.jsonl --labels results/apart/labels.jsonl --n 50 --out hand.csv
# fill the 'human' column of hand.csv yourself (one of: pact, coded, join, oath, vetting, accusation, leak, relay, report_talk, chat)
python -m cartel.judge agree --hand hand.csv
python analyze_cartel.py --results results/apart/results.jsonl --labels results/apart/labels.jsonl --out analysis/apart
```
Without judge labels the analysis still runs, but "coded talk", "vetting" and "accusation" measures stay empty (the keyword classifier cannot see them).

## What to expect by the 26th
Rough, until the pilot measures tokens/call: at ~800 tokens per call a 12-round game costs ~48K tokens, so ~4 games per model per day, ~12 per day across three models, ~25-35 games by the morning of the 26th (about 6-9 per cell, 2-3 per cell per model). That is enough for large effects (does the positive control collude, does being told you're watched change talk) and too little for small ones: report those as preliminary and point to the power figure (F19).
