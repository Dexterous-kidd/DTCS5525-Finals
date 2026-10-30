# LLM-judge decision reproducibility: complete study code

Implements the Milestone 2 design end to end: data preparation, verdict-record generation with three open-weight
judges (gpt-oss-20b, Qwen3-8B, gpt-oss-120b), six judging protocols, cluster bootstrap, hypotheses H1a/H1b/H2/H3,
MT-Bench replication, ablations, and automatic logging for the paper.

## 1. Set up (GPU machine)
**Default backend: Ollama** (`configs/judges.yaml`). The vLLM configuration is kept in `configs/judges_vllm.yaml`;
to use it, set `judge_config: configs/judges_vllm.yaml` in `configs/study.yaml`.
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # vllm is only needed for the vLLM backend: remove that line if it fails to install
bash scripts/ollama_setup.sh               # starts `ollama serve` with parallel slots, pulls gpt-oss:20b and qwen3:8b
huggingface-cli download lmarena-ai/arena-human-preference-140k --repo-type dataset --local-dir data/raw/arena-human-preference-140k
huggingface-cli download lmsys/mt_bench_human_judgments        --repo-type dataset --local-dir data/raw/mt_bench_human_judgments
python -m pytest -q tests                  # 16 tests, ~10 s
python -m judgestudy smoke                 # whole pipeline on synthetic data + mock judges (no GPU, ~3 min)
```

## 2. Run order (or `scripts/run_full.sh`, which has the gate pauses built in)
| step | command | what it does |
|---|---|---|
| 1 | `python -m judgestudy inspect` | prints the real schema and winner values. **If column names differ, edit `arena.columns` in `configs/study.yaml`.** |
| 2 | `python -m judgestudy prepare` | filters, funnel table, pair selection, dev/test split, MT-Bench items. Writes gate evidence. `--min-battles 120` is the pre-specified contingency. |
| 3 | `python -m judgestudy generate --judge gpt-oss-20b --source arena --split dev --limit 400` | **pilot**. Check `data/private/raw/` text, invalid rate (< 1 %), truncation. |
| 4 | `generate --judge J --source S --split dev` then `--split test` for J in 3 judges, S in arena/mt | full verdict record. Restartable: finished calls are skipped. Shardable: `--shard i --n-shards N`. |
| 5 | `python -m judgestudy checks --judge J` | determinism check (T=0 repeated) and contamination probe. |
| 6 | `python -m judgestudy tune --source arena` (and `mt`) | z* per judge/protocol and probit fits on **dev only**. If coverage 80 % is unreachable it tells you to rerun with `--coverage 0.7`. |
| 7 | `python -m judgestudy freeze` | hashes configs, prompts and tuned parameters into `data/processed/FROZEN.json`. Archive it on Zenodo, put the DOI in the paper. |
| 8 | `python -m judgestudy analyze --source arena` (and `mt`) | all analyses on the **test** split. Refuses to run if frozen files changed. |
| 9 | `python -m judgestudy report` / `release` | `summary.md`; public-safe verdict export (ids and labels only). |

Use `--run-id NAME` (before the subcommand) to append to a run folder; each GPU job should use its own id.

## 3. What gets collected (for documentation)
Everything lands in `results/<run-id>/`:
- `logs/run.log` timestamped log of every stage; vLLM/slurm stdout also lands in `slurm-*.out`.
- `manifest.json` git commit, pip freeze, GPU, SLURM variables, SHA-256 of configs and prompts, full config snapshot.
- `stages.jsonl` duration and status per stage; `results.jsonl` every named result; `gate_decisions.md` evidence per gate.
- `tables/*.csv|md`, `figures/*.png`, `RESULTS_INDEX.csv`, and an auto-assembled `summary.md`.
- Persistent: `data/verdicts/<judge>/` (verdict lines, `header.json` with weight revision, vLLM version and GPU, `efficiency.jsonl`),
  `data/processed/*_funnel.csv`, `data/private/raw/` (raw model text, never released).

Paper section map: eligibility funnel -> Methods/Data; `generation_quality_*`, `efficiency_*` -> Development environment and
compute cost; `item_level_metrics_*` -> baseline agreement table; `decomposition_*` + `H1a/H1b` -> RQ1; `protocol_summary_*`,
`H2*`, `risk_coverage_*` -> RQ2; `H3*`, `probit_params_*` -> RQ3; `anchor_replication`, `ablations_*` -> robustness;
`check/*` entries in `results.jsonl` -> validity checks.

## 4. Design decisions you must state in the paper (they differ from, or sharpen, the written plan)
1. **Disjoint generations per replicate.** The record has only four sampled generations per battle and order. If two replicates
   drew from the same four, they would reuse the same call 25 % of the time and flips would be understated. Each draw therefore
   splits the four generations into two disjoint pairs, one pair per replicate (`sim.gen_sets`).
2. **Invalid verdicts** after one retry are scored as ties and their rate is reported (limit 1 %).
3. **z\* is tuned on the close development pairs** (all pairs for MT-Bench) at budget 32, per judge and protocol.
4. **S3 (non-inferiority on agreement with human-clear decisions) uses all human-clear pairs**, because close pairs rarely have a
   human-clear winner. The number of human-clear close/far pairs is recorded (`human_clear_pairs/*`).
5. **H3 is evaluated for two models.** M0 (the specified ordered probit with independent calls) is the primary test.
   M1 adds a battle random intercept and is labelled exploratory (`H3_RE_*`). Predictions account for the shared battle subset and the
   finite 128-battle pool.
6. **MT-Bench category** is derived from the question id (80 questions, 10 per category, ids 81 to 160).
7. Arena battles are put in a canonical orientation (alphabetically first system = A). Protocol P0 uses the dataset's original
   presentation order (`ds_flip`).

## 5. Things I could not verify here (check before the full run)
- **Neither real backend was executed** (no GPU, no Ollama in the build environment). The Ollama client is tested against a fake local server
  (payload fields, headers, errors); everything else, including checkpointing, sharding, retries and all analyses, ran with mock judges.
  The pilot step exists to catch real-server issues: the gpt-oss output format, and whether the reply `content` contains the verdict tag.
- **Ollama specifics.** (1) Ollama's default context is 4096 tokens; the code always sends `num_ctx = max_model_len` (8192). KV memory grows with
  `num_ctx x OLLAMA_NUM_PARALLEL`, so lower `workers`/`OLLAMA_NUM_PARALLEL` if you run out of GPU memory. (2) Ollama serves quantized GGUF weights
  (the quantization is recorded in `data/verdicts/<judge>/header.json`); say in the paper that the judges are the Ollama builds, not the
  full-precision Hugging Face weights, and cite the digest. (3) `top_k <= 0` in the config means Ollama's default (40) applies. (4) Temperature-zero
  determinism with concurrent requests is not guaranteed; the determinism check measures it. (5) `think: low` for gpt-oss and `think: false` for
  Qwen3 need a recent Ollama (`ollama --version`; update if the pilot shows reasoning text in the answers or empty replies).
- **Prompt templates** in `prompts/` were written from memory of Zheng et al. (2023). Compare them word for word with the paper's appendix.
- **Arena 140k column names** and the language field are configured, not confirmed. `inspect` tells you.
- gpt-oss-120b is optional (about 65 GB of disk and an 80 GB GPU) and is not in the default run. `qwen3-8b-think` is a sensitivity judge: it is skipped by `tune`/`analyze` and compared with `qwen3-8b` by `python -m judgestudy sensitivity`.
- Bootstrap cost: about 30 ms per pair per replicate; with 50 pairs, 1,000 resamples and 8 workers, expect minutes, not hours.
