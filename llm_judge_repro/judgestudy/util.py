"""Small shared helpers: config loading, hashing, JSONL I/O, environment capture."""
import hashlib, json, os, platform, subprocess, sys, time, zlib
from pathlib import Path
import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def deep_update(base, over):
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_cfg(path="configs/study.yaml", smoke=False):
    cfg = load_yaml(path)
    cfg["judges_cfg"] = load_yaml(cfg["judge_config"])
    if smoke:
        deep_update(cfg, cfg.get("smoke_overrides", {}))
    return cfg


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(*parts, mod=2**32):
    """Deterministic across processes (unlike built-in hash)."""
    return zlib.crc32("|".join(str(p) for p in parts).encode()) % mod


def jsonl_append(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")


def jsonl_read(path):
    out = []
    if not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:   # a half-written last line after a crash
                    continue
    return out


def _json_default(o):
    import numpy as np
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def run_cmd(cmd, timeout=60):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:  # noqa
        return f"unavailable ({e.__class__.__name__})"


def git_info():
    return {"commit": run_cmd(["git", "rev-parse", "HEAD"]),
            "dirty": bool(run_cmd(["git", "status", "--porcelain"]))}


def gpu_info():
    out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    return out


def env_info():
    return {
        "python": sys.version, "platform": platform.platform(), "host": platform.node(),
        "cpu_count": os.cpu_count(), "gpu": gpu_info(),
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def pip_freeze():
    return run_cmd([sys.executable, "-m", "pip", "freeze"], timeout=120)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")
