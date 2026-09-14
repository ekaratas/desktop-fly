"""Replay a run's recorded decisions into the DesktopFly bridge file ([C]).

    python -m trader.ui.replay runs/<run_dir> [--split test] [--rate 2] [--loop]

Each recorded decision becomes the current market state at `--rate` decisions per
second (2 = a day of 1h bars every 12 s), with `updated` set to now so the fly
treats it as live. The fly sleeps/explores when the organism is CALM, raises its
wings and gets nervous on ALERT, and gets one looming step on ESCAPE. Nothing
here touches the model; it only re-emits what the organism already decided.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .state import creature_behavior, creature_state_path, write_state_atomic


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--split", default="test")
    p.add_argument("--rate", type=float, default=2.0, help="decisions per second")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--path", default=None, help="bridge file (default: DesktopFly's)")
    a = p.parse_args(argv)
    path = a.path or creature_state_path()
    src = os.path.join(a.run_dir, f"decisions_{a.split}.jsonl")
    rows = [json.loads(l) for l in open(src)]
    print(f"replaying {len(rows)} {a.split} decisions → {path} at {a.rate}/s (Ctrl-C to stop)")
    counts: dict[str, int] = {}
    try:
        while True:
            for r in rows:
                behavior = creature_behavior(r["action"], r["arousal"], r["gated"], r.get("reward"))
                counts[behavior] = counts.get(behavior, 0) + 1
                write_state_atomic(path, {
                    "updated": time.time(), "t": r["t"], "split": r["split"], "decision": r["action"], "label": r["label"],
                    "arousal": r["arousal"], "gated": r["gated"], "behavior": behavior, "channels": r.get("channels", {}),
                    "pop_share": r.get("pop_share", {}), "reward": r.get("reward"), "replay": True,
                })
                print(f"\r{r['t']}  {r['action']:<8} arousal {r['arousal']:.2f} → {behavior:<9}   ", end="", flush=True)
                time.sleep(1.0 / max(0.01, a.rate))
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    print("\nbehaviors:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
