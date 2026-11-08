# Kuku Promo Finder

Reads microdrama screenplays and finds 20-60s promo clips: hook in the first 3 seconds,
cut on a cliffhanger, understandable to a stranger.

**Core design:** the LLM judges drama but may only answer with event IDs (`E07-012`).
Code turns IDs into timestamps and enforces every checkable rule, so the model cannot
invent a timestamp. Rule violations are sent back to the model once for repair; anything
still broken is kept in the output as `rejected` with the reason.

## Setup (Python 3.10+, no pip installs needed)
1. Get a free key at https://aistudio.google.com/apikey
2. `cp .env.example .env` and paste the key in.
3. Put the show folder (the `episode-*.txt` files) inside `data/`.

## Run
```bash
python parse_show.py "data/<show>" --show-episode 1        # parse + sanity report
python find_promos.py --list-models --provider gemini        # which models your key can use
python find_promos.py "data/<show>" --episodes 1-3           # quick test run
python find_promos.py "data/<show>" --thinking low --batch 5  # all episodes, 5 per request (~12 requests)
python find_promos.py "data/<show>" --provider baseline      # rule-based comparison, no LLM
python rank_promos.py "data/<show>" outputs/runs/<run>        # dedup + rank + montages
python -m tests.test_pipeline "data/<show>"                  # offline tests
python -m tests.test_ranker "data/<show>"
python -m tests.test_llm_errors
```

## Pipeline
1. **Parse** (code): scripts → timestamped events with IDs.
2. **Find** (LLM per episode): 1-3 candidates as event IDs → **check** (code): length, hook in first 3s,
   cut near cliffhanger, no mid-line cuts → one **repair** round for broken ones.
3. **Rank** (`rank_promos.py`): code drops overlapping clips; one LLM call clusters clips showing the same
   story beat (episodes replay the previous cliffhanger) and ranks the winners comparatively, ignoring
   the inflated self-scores; code checks the ranking (unknown IDs, forgotten clips, too many "post first").
4. **Montages**: LLM joins moments from the top clips across episodes; same checker + repair.
5. **Human**: editor reviews `report.md` (ranked list with warnings) instead of watching every episode.
Swap models with `--provider` / `--model`. Any OpenAI-compatible API works via `--provider openai`.

## Output (`outputs/runs/<run>/`)
- `candidates.csv` / `.json`: every candidate, with segments, timestamps, hook, cliffhanger, scores, status
- `raw/epNN.json`: the model's unedited responses (including the repair round)
- `summary.json`: counts, API calls, tokens
- after ranking: `report.md` (read this first), `ranked.csv`, `ranked.json`, `rank_raw.json`

Responses are cached in `outputs/cache/`, so re-runs are free and instant.

## Note
`data/` is git-ignored: the scripts are Kuku's confidential content, don't push them.
