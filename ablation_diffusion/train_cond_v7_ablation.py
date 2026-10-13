"""
Stage-2 ablation driver.

Freeze a stage-1 *tune-side* ablation aligner (tune_side=text or tune_side=ts)
and train the conditional diffusion model EXACTLY like the main V7 pipeline,
additionally teeing the generation-eval metrics into a plain-text JSONL file so
they can be read back offline without touching W&B.

Design notes
------------
* This file does NOT modify t2ts_jit. It reuses `t2ts_jit/train_cond_v7.py`
  (which itself shims `train_cond_v6.py`) verbatim and only wraps `wandb.log`.
* Every eval metric in train_cond_v6 is logged through `wandb.log`, so wrapping
  it lets us capture the generation-eval payloads non-invasively. Only payloads
  whose keys start with `gen_eval/` are written to TXT; `train/*` and `val/*`
  payloads are passed through untouched and NOT written (per the ablation spec:
  only gen-eval results go to the TXT file).
* The TXT path is derived from `--save` (i.e. <save_dir>/test_metrics.txt), so no
  new CLI flag is added to the upstream argparse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_T2TS_DIR = _REPO_ROOT / "t2ts_jit"
# train_cond_v6/v7 live in t2ts_jit and perform their own feature_alignment/
# sys.path wiring on import, so we only need t2ts_jit/ on the path here.
sys.path.insert(0, str(_T2TS_DIR))

import train_cond_v6 as _v6  # noqa: E402

# IMPORTANT: wandb.init() REBINDS wandb.log (the run's real logger replaces the
# module-level function), so any tee installed before init is silently wiped out
# -- that bug left every test_metrics.txt with only its `meta` header. Wrap the
# real wandb.init *before* importing train_cond_v7 (which snapshots the current
# _v6.wandb.init into its module-level _ORIG_WANDB_INIT) so the gen-eval tee is
# re-installed on wandb.log AFTER each init, once the real logger is in place.
_REAL_WANDB_INIT = _v6.wandb.init


def _wandb_init_then_tee(*args, **kwargs):
    run = _REAL_WANDB_INIT(*args, **kwargs)
    _install_gen_eval_tee()  # defined below; bound at call time (post-import)
    return run


_v6.wandb.init = _wandb_init_then_tee

import train_cond_v7 as _v7  # noqa: E402


def _argv_value(flag: str) -> str | None:
    """Read a value passed as `--flag value` or `--flag=value` from sys.argv."""
    argv = sys.argv
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _resolve_txt_path() -> Path:
    save = _argv_value("--save")
    if save:
        return Path(save).resolve().parent / "test_metrics.txt"
    return _HERE / "test_metrics.txt"


_TXT_PATH = _resolve_txt_path()


def _append_record(rec: dict) -> None:
    _TXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_TXT_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _install_gen_eval_tee() -> None:
    """Wrap wandb.log so every `gen_eval/*` payload is also appended to TXT."""
    orig_log = _v6.wandb.log

    def tee_log(data=None, *args, **kwargs):
        try:
            if isinstance(data, dict) and any(k.startswith("gen_eval/") for k in data):
                step = kwargs.get("step")
                if step is None and args:
                    step = args[0]
                rec: dict = {"kind": "gen_eval", "step": step}
                for k, v in data.items():
                    if k == "gen_eval/epoch":
                        rec["epoch"] = v
                    elif k.startswith("gen_eval/"):
                        # e.g. "gen_eval/ema2/fidelity/fid" -> "ema2/fidelity/fid"
                        rec[k[len("gen_eval/"):]] = v
                _append_record(rec)
        except Exception as e:  # never let TXT logging crash training
            print(f"[ablation-txt] WARN failed to write gen-eval metrics: {e}")
        return orig_log(data, *args, **kwargs)

    _v6.wandb.log = tee_log


def main() -> None:
    # One-line run header (valid JSON) so each run is self-describing on parse.
    _append_record({
        "kind": "meta",
        "source": _argv_value("--source"),
        "length": _argv_value("--length"),
        "aligner_ckpt": _argv_value("--aligner-ckpt"),
        "metrics_aligner_ckpt": _argv_value("--metrics-aligner-ckpt"),
        "save": _argv_value("--save"),
    })
    print(f"[ablation-txt] gen-eval metrics -> {_TXT_PATH}")
    # The gen-eval tee is installed by _wandb_init_then_tee right AFTER
    # wandb.init() (see the import-time wrapper above). Installing it here would
    # be wiped out by wandb.init() rebinding wandb.log.
    # Hand off to the unmodified V7 conditional-diffusion training entrypoint.
    _v7.main()


if __name__ == "__main__":
    main()
