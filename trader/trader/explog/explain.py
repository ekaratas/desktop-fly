"""Per-decision explainability record ([C])."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class DecisionRecord:
    t: str
    split: str
    arousal: float
    channels: dict
    gated: bool
    kc_active_frac: float
    pop_drive: dict
    pop_share: dict
    mbon_spikes: list
    action: str
    label: str
    reward: float | None = None
    dan: dict = field(default_factory=dict)
    fwd_return_atr: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=float)

    def pretty(self) -> str:
        lines = [f"[{self.t}] AROUSAL: {self.arousal:.2f}" + ("  (gated: resting)" if self.gated else "")]
        for k, v in self.channels.items():
            lines.append(f"  {k.upper():<11} {v:+.2f}")
        for k, v in self.pop_share.items():
            lines.append(f"  {k.upper()+' POP':<11} {v:.2f}")
        lines.append(f"  DECISION: {self.action}   (label {self.label}, reward {self.reward if self.reward is None else round(self.reward, 3)})")
        return "\n".join(lines)
