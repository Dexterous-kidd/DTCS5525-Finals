"""RunContext: one object that collects everything needed for documentation.

Per run it writes (under results/<run_id>/):
  logs/run.log          timestamped log (also echoed to the console)
  manifest.json         config hashes, git commit, python/pip freeze, GPU, SLURM variables
  stages.jsonl          one line per stage: start, duration, status
  results.jsonl         every named result (value, CI, hypothesis verdict ...)
  tables/*.csv|*.md     every table
  figures/*.png         every figure
  gate_decisions.md     auto-generated evidence for each gate (copy into the paper's log)
"""
import json, logging, sys, time, traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from .util import (env_info, git_info, jsonl_append, now, pip_freeze, sha256_file, _json_default)


class RunContext:
    def __init__(self, cfg, run_id=None, label="run", args=None):
        self.cfg = cfg
        root = Path(cfg["paths"]["results"])
        self.run_id = run_id or (time.strftime("%Y%m%d-%H%M%S") + f"-{label}")
        self.dir = root / self.run_id
        for sub in ("logs", "tables", "figures"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.log = self._setup_logging()
        self._write_manifest(args)
        self.log.info("run directory: %s", self.dir)

    # ---------- logging ----------
    def _setup_logging(self):
        logger = logging.getLogger("judgestudy")
        logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.dir / "logs" / "run.log")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
        logger.propagate = False
        return logger

    # ---------- manifest ----------
    def _write_manifest(self, args):
        p = self.dir / "manifest.json"
        entry = {"time": now(), "args": args, "git": git_info(), "env": env_info()}
        if p.exists():                       # resumed run: keep history
            old = json.loads(p.read_text())
            old.setdefault("resumed", []).append(entry)
            p.write_text(json.dumps(old, indent=2, default=_json_default))
            return
        files = {}
        for f in ["configs/study.yaml", "configs/judges.yaml",
                  self.cfg["prompts"]["single"], self.cfg["prompts"]["multi"]]:
            if Path(f).exists():
                files[f] = sha256_file(f)
        entry["file_sha256"] = files
        entry["seed"] = self.cfg.get("seed")
        entry["pip_freeze"] = pip_freeze().splitlines()
        entry["config_snapshot"] = {k: v for k, v in self.cfg.items() if k != "judges_cfg"}
        entry["judges_snapshot"] = self.cfg["judges_cfg"]
        p.write_text(json.dumps(entry, indent=2, default=_json_default))

    # ---------- stages ----------
    @contextmanager
    def stage(self, name):
        t0 = time.time()
        self.log.info("=" * 20 + f" STAGE START: {name} " + "=" * 20)
        try:
            yield
            status = "ok"
        except Exception:
            status = "failed"
            self.log.error("stage %s failed:\n%s", name, traceback.format_exc())
            jsonl_append(self.dir / "stages.jsonl", [{"stage": name, "status": status, "seconds": round(time.time() - t0, 2), "time": now()}])
            raise
        dt = time.time() - t0
        self.log.info("=" * 20 + f" STAGE DONE: {name} ({dt:.1f}s) " + "=" * 20)
        jsonl_append(self.dir / "stages.jsonl", [{"stage": name, "status": status, "seconds": round(dt, 2), "time": now()}])

    # ---------- results ----------
    def record(self, name, **fields):
        row = {"name": name, "time": now(), **fields}
        jsonl_append(self.dir / "results.jsonl", [row])
        short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in fields.items() if not isinstance(v, (list, dict))}
        self.log.info("RESULT %s %s", name, short)

    def table(self, df, name, index=False, show=True):
        df = df.copy()
        df.to_csv(self.dir / "tables" / f"{name}.csv", index=index)
        try:
            md = df.to_markdown(index=index, floatfmt=".4g")
        except Exception:
            md = df.to_string(index=index)
        (self.dir / "tables" / f"{name}.md").write_text(md)
        if show:
            self.log.info("TABLE %s (%d rows)\n%s", name, len(df), df.head(12).to_string(index=index))
        return df

    def figure(self, fig, name):
        p = self.dir / "figures" / f"{name}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        self.log.info("FIGURE %s", p)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def gate(self, name, rows, verdict, fallback=""):
        """rows: list of (criterion, evidence, passed)."""
        lines = [f"## Gate: {name}  ({now()}, run {self.run_id})", "", "| Criterion | Evidence | Pass |", "|---|---|---|"]
        for c, e, ok in rows:
            lines.append(f"| {c} | {e} | {'yes' if ok else 'NO'} |")
        lines += ["", f"**Decision:** {verdict}"]
        if fallback:
            lines.append(f"**Fallback if failed:** {fallback}")
        lines.append("")
        text = "\n".join(lines)
        with open(self.dir / "gate_decisions.md", "a") as f:
            f.write(text + "\n")
        self.log.info("GATE %s -> %s", name, verdict)
