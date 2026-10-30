"""Judge backends.  A request is a dict: messages, temperature, seed, meta (ignored by real judges).
A response is a dict: text, finish_reason, n_in, n_out."""
import json, logging, os, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .util import stable_hash

log = logging.getLogger("judgestudy")


def judge_spec(cfg, name):
    spec = dict(cfg["judges_cfg"]["judges"][name])
    spec["name"] = name
    return spec


class VLLMJudge:
    def __init__(self, spec, gen_cfg):
        self.spec, self.gen_cfg = spec, gen_cfg
        self.name = spec["name"]
        self.llm = None
        self.revision = "unknown"

    # -- sampling settings come from the config file (or the model's generation_config.json) --
    def _sampling_settings(self):
        s = self.spec["sampling"]
        if s == "from_generation_config":
            try:
                from transformers import GenerationConfig
                g = GenerationConfig.from_pretrained(self.spec["hf_id"])
                out = {"temperature": g.temperature if g.temperature is not None else 1.0,
                       "top_p": g.top_p if g.top_p is not None else 1.0,
                       "top_k": g.top_k if (g.top_k is not None and g.top_k > 0) else -1, "min_p": 0.0}
                log.info("[%s] sampling read from generation_config.json: %s", self.name, out)
                return out
            except Exception as e:
                log.warning("[%s] could not read generation_config (%s); using sampling_fallback", self.name, e)
                return dict(self.spec["sampling_fallback"])
        return dict(s)

    def load(self):
        from vllm import LLM
        t0 = time.time()
        self.sampling = self._sampling_settings()
        self.llm = LLM(model=self.spec["hf_id"], tensor_parallel_size=self.spec.get("tensor_parallel_size", 1),
                       max_model_len=self.spec["max_model_len"], gpu_memory_utilization=self.spec["gpu_memory_utilization"],
                       dtype=self.spec.get("dtype", "auto"), seed=0, trust_remote_code=False)
        try:
            self.revision = self.llm.llm_engine.model_config.revision or self._revision_from_cache()
        except Exception:
            self.revision = self._revision_from_cache()
        import vllm
        self.vllm_version = vllm.__version__
        log.info("[%s] loaded in %.0fs | vllm %s | revision %s | sampling %s", self.name, time.time() - t0,
                 self.vllm_version, self.revision, self.sampling)

    def _revision_from_cache(self):
        try:
            from huggingface_hub import snapshot_download
            return Path(snapshot_download(self.spec["hf_id"], local_files_only=True)).name
        except Exception:
            return "unknown"

    def header(self):
        import torch
        return {"judge": self.name, "hf_id": self.spec["hf_id"], "revision": self.revision, "vllm": getattr(self, "vllm_version", None),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "n_gpu": torch.cuda.device_count(), "sampling": self.sampling,
                "chat_template_kwargs": self.spec.get("chat_template_kwargs", {})}

    def generate(self, requests):
        from vllm import SamplingParams
        max_tokens = self.spec.get("max_tokens", self.gen_cfg["max_tokens"])
        params = []
        for r in requests:
            if r["temperature"] == 0:
                params.append(SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=r["seed"]))
            else:
                s = self.sampling
                params.append(SamplingParams(temperature=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], min_p=s.get("min_p", 0.0),
                                             max_tokens=max_tokens, seed=r["seed"]))
        outs = self.llm.chat([r["messages"] for r in requests], sampling_params=params, use_tqdm=False,
                             chat_template_kwargs=self.spec.get("chat_template_kwargs", {}))
        res = []
        for o in outs:
            c = o.outputs[0]
            res.append({"text": c.text, "finish_reason": c.finish_reason, "n_in": len(o.prompt_token_ids), "n_out": len(c.token_ids)})
        return res


class OllamaJudge:
    """Judge served by a local Ollama server (REST API, no Python dependency).

    Start the server with parallel slots (see scripts/ollama_setup.sh) and keep spec['workers'] <= OLLAMA_NUM_PARALLEL.
    IMPORTANT: Ollama silently truncates prompts to its default context (4096 tokens); this class always sends
    options.num_ctx = spec['max_model_len'] so that prompts up to 6,000 tokens fit.
    """
    def __init__(self, spec, gen_cfg):
        self.spec, self.gen_cfg, self.name = spec, gen_cfg, spec["name"]
        self.host = os.environ.get("OLLAMA_HOST", spec.get("host", "http://localhost:11434"))
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.model = spec["ollama_model"]
        self.workers = int(spec.get("workers", 8))
        self.sampling = dict(spec["sampling"])
        self.revision = "unknown"; self._header = {}

    # ---- HTTP ----
    def _call(self, path, payload=None, timeout=900):
        req = urllib.request.Request(self.host + path, data=None if payload is None else json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())

    def load(self):
        try:
            tags = self._call("/api/tags", timeout=30)
        except Exception as e:
            raise RuntimeError(f"cannot reach Ollama at {self.host} ({e}). Start it with `ollama serve` (see scripts/ollama_setup.sh).")
        have = {m["name"]: m for m in tags.get("models", [])}
        match = [n for n in have if n == self.model or n == self.model + ":latest"]
        if not match:
            raise RuntimeError(f"model '{self.model}' is not pulled on {self.host}. Run: ollama pull {self.model}  (available: {sorted(have)})")
        m = have[match[0]]; self.revision = m.get("digest", "unknown")
        try:
            show = self._call("/api/show", {"model": self.model}, timeout=60); details = show.get("details", {})
        except Exception:
            details = m.get("details", {})
        try:
            ver = self._call("/api/version", timeout=10).get("version")
        except Exception:
            ver = None
        self._header = {"judge": self.name, "backend": "ollama", "hf_id": self.model, "revision": self.revision, "ollama_version": ver,
                        "quantization": details.get("quantization_level"), "parameter_size": details.get("parameter_size"),
                        "format": details.get("format"), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "ollama-managed"), "n_gpu": None,
                        "sampling": self.sampling, "think": self.spec.get("think"), "num_ctx": self.spec["max_model_len"],
                        "num_predict": self.spec.get("max_tokens", self.gen_cfg["max_tokens"]), "workers": self.workers,
                        "note": "top_k omitted when <= 0, so Ollama's default (40) applies"}
        log.info("[%s] Ollama %s | model %s | digest %s | quantization %s | workers %d", self.name, ver, self.model, self.revision[:12],
                 details.get("quantization_level"), self.workers)

    def header(self):
        return dict(self._header)

    def _options(self, r):
        o = {"seed": int(r["seed"]), "num_ctx": int(self.spec["max_model_len"]), "num_predict": int(self.spec.get("max_tokens", self.gen_cfg["max_tokens"]))}
        if r["temperature"] == 0:
            o["temperature"] = 0.0
        else:
            s = self.sampling
            o.update(temperature=s["temperature"], top_p=s["top_p"], min_p=s.get("min_p", 0.0))
            if s.get("top_k", -1) and s["top_k"] > 0:
                o["top_k"] = int(s["top_k"])
        return o

    def _one(self, r):
        payload = {"model": self.model, "messages": r["messages"], "stream": False, "options": self._options(r),
                   "keep_alive": self.spec.get("keep_alive", "30m")}
        if self.spec.get("think") is not None:
            payload["think"] = self.spec["think"]                    # false / true / "low" / "medium" / "high"
        last = None
        for attempt in range(4):
            try:
                d = self._call("/api/chat", payload)
                msg = d.get("message", {})
                return {"text": msg.get("content", "") or "", "finish_reason": "length" if d.get("done_reason") == "length" else "stop",
                        "n_in": int(d.get("prompt_eval_count", 0)), "n_out": int(d.get("eval_count", 0))}
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last = e; time.sleep(2 ** attempt)
        log.warning("[%s] request failed after retries (%s); recorded as invalid and retried by the pipeline", self.name, last)
        return {"text": "", "finish_reason": "error", "n_in": 0, "n_out": 0}

    def generate(self, requests):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, requests))


class MockJudge:
    """Latent-model judge for smoke tests and for exercising the whole pipeline on a laptop.
    Verdicts depend on a hidden pair margin (meta['m']), a position bias, a persistent per-item
    difficulty term, and per-call noise. It never sees real data."""
    PARAMS = {"mock-a": dict(beta=5.0, gamma=0.35, unit_sd=0.7, noise=0.9, p_invalid=0.01),
              "mock-b": dict(beta=4.0, gamma=0.10, unit_sd=0.9, noise=0.7, p_invalid=0.02),
              "mock-c": dict(beta=6.0, gamma=0.20, unit_sd=0.5, noise=0.5, p_invalid=0.005)}

    def __init__(self, spec, gen_cfg):
        self.spec, self.gen_cfg, self.name = spec, gen_cfg, spec["name"]
        self.p = self.PARAMS.get(self.name, self.PARAMS["mock-a"])
        self.revision = "mock"
        self.sampling = {"temperature": 1.0}

    def load(self):
        log.info("[%s] mock judge ready", self.name)

    def header(self):
        return {"judge": self.name, "hf_id": "mock", "revision": "mock", "vllm": None, "gpu": "none", "n_gpu": 0,
                "sampling": self.sampling, "chat_template_kwargs": {}}

    def generate(self, requests):
        res = []
        for r in requests:
            meta = r["meta"]; p = self.p
            sgn = 1.0 if meta["order"] == "AB" else -1.0
            u = np.random.default_rng(stable_hash(self.name, meta["unit_id"], "unit")).normal(0, p["unit_sd"])
            sd = 0.05 if r["temperature"] == 0 else p["noise"]
            eps = np.random.default_rng(stable_hash(self.name, meta["unit_id"], meta["order"], r["seed"], r["temperature"])).normal(0, sd)
            mu = p["beta"] * meta["m"] * sgn + p["gamma"] + sgn * u + eps          # evidence that the FIRST-shown answer is better
            shown = "A" if mu > 0.3 else ("B" if mu < -0.3 else "C")
            ok = np.random.default_rng(stable_hash(self.name, meta["unit_id"], meta["order"], r["seed"], "inv")).random() > p["p_invalid"]
            text = f"Both answers are reasonable; comparing them carefully.\nFinal verdict: [[{shown}]]" if ok else "I cannot decide."
            res.append({"text": text, "finish_reason": "stop", "n_in": 400, "n_out": 40})
        return res


def make_judge(cfg, name, mock=False):
    spec = judge_spec(cfg, name)
    if mock or spec.get("backend") == "mock":
        return MockJudge(spec, cfg["generation"])
    if spec.get("backend") == "ollama":
        return OllamaJudge(spec, cfg["generation"])
    return VLLMJudge(spec, cfg["generation"])
