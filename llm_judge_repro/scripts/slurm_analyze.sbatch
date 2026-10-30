#!/bin/bash
# CPU job: bootstrap is embarrassingly parallel. Set sim.n_jobs in configs/study.yaml to the cpus-per-task below.
#SBATCH --job-name=judgeanalyze
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out
set -euo pipefail
source .venv/bin/activate
python -m judgestudy --run-id "${RUN}-analysis" analyze --source arena
python -m judgestudy --run-id "${RUN}-analysis" analyze --source mt
python -m judgestudy --run-id "${RUN}-analysis" report
