#!/usr/bin/env bash
# Start a local Ollama server tuned for this study and pull the three judges.
# No root? Download the tarball into your home directory:
#   curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz && mkdir -p ~/ollama && tar -C ~/ollama -xzf ollama.tgz
#   export PATH=$HOME/ollama/bin:$PATH
set -euo pipefail
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-8}      # concurrent requests per model (KV memory = num_ctx x parallel slots)
export OLLAMA_MAX_LOADED_MODELS=1                          # one judge at a time
export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-30m}
export OLLAMA_MODELS=${OLLAMA_MODELS:-$PWD/.ollama_models} # big downloads go to the project disk, not $HOME
mkdir -p "$OLLAMA_MODELS" results
if curl -s localhost:11434/api/version > /dev/null; then
  echo "An Ollama server is ALREADY running on :11434. Reusing it (its model store and OLLAMA_NUM_PARALLEL cannot be changed from here)."
  echo "To use your own server instead: OLLAMA_HOST=127.0.0.1:11435 bash scripts/ollama_setup.sh   (then export OLLAMA_HOST=127.0.0.1:11435 before running judgestudy)"
else
  nohup ollama serve > results/ollama_server.log 2>&1 &
  sleep 8
fi
for m in gpt-oss:20b qwen3:8b; do ollama pull "$m"; done      # add gpt-oss:120b only if you have ~65 GB of disk and an 80 GB GPU
ollama list
echo "Server log: results/ollama_server.log  (attach it to your documentation). If memory is tight, lower OLLAMA_NUM_PARALLEL and 'workers' in configs/judges.yaml."
