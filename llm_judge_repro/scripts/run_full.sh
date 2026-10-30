#!/usr/bin/env bash
# Full study on one GPU machine. Run from the repo root.  Read the PAUSE points: they are the gates from the roadmap.
set -euo pipefail
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}          # tokenizer download for the length filter
# Ollama must already be running: bash scripts/ollama_setup.sh
RUN=${RUN:-full-$(date +%Y%m%d-%H%M)}
J="python -m judgestudy --run-id"
JUDGES="gpt-oss-20b qwen3-8b"                       # the two core judges (gpt-oss-120b is an optional stretch judge: ~65 GB download)

# --- 0. data (download once; see README) ---
$J "$RUN-data" inspect
$J "$RUN-data" prepare                                  # writes the eligibility funnel; check the Foundations gate in gate_decisions.md
read -p "PAUSE 1: funnel OK (>=30 eligible pairs)? [enter to continue] "

# --- 1. pilot: 400 dev battles per judge, look at raw outputs and invalid rate ---
for j in $JUDGES; do $J "$RUN-pilot-$j" generate --judge $j --source arena --split dev --limit 400; done
echo "Inspect data/private/raw/<judge>/ and results/*pilot*/tables/generation_quality_arena.csv (invalid rate must be < 1%)."
read -p "PAUSE 2: parser/prompt fine? If not, fix on DEV only, delete data/verdicts/<judge>, rerun pilot. [enter] "

# --- 2. full verdict record (dev first, then test) ---
for j in $JUDGES; do
  for s in arena mt; do
    $J "$RUN-gen-$j" generate --judge $j --source $s --split dev
    $J "$RUN-gen-$j" generate --judge $j --source $s --split test
  done
  $J "$RUN-checks-$j" checks --judge $j --kind both
done

# --- 3. tune on dev, freeze, analyse on test ---
for s in arena mt; do $J "$RUN-analysis" tune --source $s; done
$J "$RUN-analysis" freeze
read -p "PAUSE 3: archive data/processed/FROZEN.json + configs/ + prompts/ on Zenodo, paste the DOI in the paper, then continue. [enter] "
for s in arena mt; do $J "$RUN-analysis" analyze --source $s; done
$J "$RUN-analysis" report

# --- 4. optional sensitivity check: Qwen3 with thinking on, 200 dev battles only (thinking replies are long: watch the ETA in the log) ---
read -p "OPTIONAL 4: run the thinking-mode sensitivity check now? [enter to run, Ctrl+C to skip] "
$J "$RUN-sens" generate --judge qwen3-8b-think --source arena --split dev --limit 200
$J "$RUN-sens" sensitivity --source arena --judge qwen3-8b-think --baseline qwen3-8b
$J "$RUN-analysis" release --out release
echo "Done. Documentation material: results/$RUN-analysis/summary.md"
