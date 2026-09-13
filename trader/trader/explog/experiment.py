"""Experiment bookkeeping ([C]): a run directory with everything needed to reproduce it."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn

from ..connectome.mb_topology import TOPOLOGY_VERSION
from ..features.raw import FEATURE_VERSION
from ..features.sensory import SENSORY_VERSION
from ..labels.excursion import LABEL_VERSION


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


class RunDir:
    def __init__(self, cfg: dict, tag: str = ""):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}_{cfg['experiment']}{('_' + tag) if tag else ''}"
        self.path = os.path.join(cfg.get("output_dir", "runs"), name)
        os.makedirs(self.path, exist_ok=True)
        self.t0 = time.time()
        self.manifest = {
            "experiment": cfg["experiment"], "created": stamp, "config": cfg, "git_commit": git_commit(),
            "versions": {"features": FEATURE_VERSION, "sensory": SENSORY_VERSION, "labels": LABEL_VERSION,
                         "topology": TOPOLOGY_VERSION, "topology_category": "B (statistics only; no measured edges)",
                         "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                         "sklearn": sklearn.__version__},
        }
        self.write_json("manifest.json", self.manifest)

    def file(self, name: str) -> str:
        return os.path.join(self.path, name)

    def write_json(self, name: str, obj) -> None:
        with open(self.file(name), "w") as f:
            json.dump(obj, f, indent=2, default=_default)

    def append_jsonl(self, name: str, line: str) -> None:
        with open(self.file(name), "a") as f:
            f.write(line + "\n")

    def finish(self, extra: dict) -> None:
        self.manifest.update(extra)
        self.manifest["elapsed_s"] = round(time.time() - self.t0, 1)
        self.write_json("manifest.json", self.manifest)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (o != o):
        return None
    return str(o)
