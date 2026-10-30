#!/bin/bash
# Usage: sbatch --export=ALL,JUDGE=gpt-oss-20b,SOURCE=arena,SPLIT=all,RUN=full1 scripts/slurm_generate.sbatch
# One array task per shard; shards write separate files, so the job is restartable (finished calls are skipped).
#SBATCH --job-name=judgegen
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%A_%a.out
set -euo pipefail
module load cuda 2>/dev/null || true
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
python -m judgestudy --run-id "${RUN}-gen-${JUDGE}-shard${SLURM_ARRAY_TASK_ID}" generate \
  --judge "$JUDGE" --source "$SOURCE" --split "${SPLIT:-all}" --shard "$SLURM_ARRAY_TASK_ID" --n-shards 4
