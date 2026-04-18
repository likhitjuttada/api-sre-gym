from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from gym.registry import TIERS, all_bug_types, tier_for

_STATE_FILE = Path(__file__).resolve().parent.parent / "curriculum_state.json"


@dataclass
class BugStats:
    attempts: int = 0
    successes: int = 0
    recent_rewards: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def mastery(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def avg_recent(self) -> float:
        if not self.recent_rewards:
            return 0.0
        return sum(self.recent_rewards) / len(self.recent_rewards)

    def to_dict(self) -> dict:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "recent_rewards": list(self.recent_rewards),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BugStats":
        s = cls(attempts=int(data.get("attempts", 0)), successes=int(data.get("successes", 0)))
        for r in data.get("recent_rewards", []):
            s.recent_rewards.append(float(r))
        return s


class Curriculum:
    def __init__(self, state_path: Path = _STATE_FILE) -> None:
        self.state_path = state_path
        self.current_tier: int = 1
        self.stats: dict[str, BugStats] = {b: BugStats() for b in all_bug_types()}
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self.current_tier = int(data.get("current_tier", 1))
        for bug, d in data.get("stats", {}).items():
            if bug in self.stats:
                self.stats[bug] = BugStats.from_dict(d)

    def save(self) -> None:
        payload = {
            "current_tier": self.current_tier,
            "stats": {b: s.to_dict() for b, s in self.stats.items()},
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    def record_episode(self, bug_type: str, total_reward: float, resolved: bool) -> None:
        if bug_type not in self.stats:
            self.stats[bug_type] = BugStats()
        s = self.stats[bug_type]
        s.attempts += 1
        if resolved:
            s.successes += 1
        s.recent_rewards.append(total_reward)
        self._maybe_escalate()

    def _maybe_escalate(self) -> None:
        if self.current_tier >= max(TIERS.keys()):
            return
        current_bugs = [b for b in TIERS[self.current_tier] if b in self.stats]
        if not current_bugs:
            return
        masteries = [self.stats[b].mastery for b in current_bugs]
        min_attempts = min(self.stats[b].attempts for b in current_bugs)
        if min_attempts >= 10 and all(m >= 0.7 for m in masteries):
            self.current_tier += 1

    def weak_spots(self, threshold: float = 0.5) -> list[str]:
        return [
            b for b, s in self.stats.items()
            if s.attempts >= 3 and s.mastery < threshold
        ]

    def get_next_episode(self) -> tuple[str, float]:
        available: list[str] = []
        for tier in range(1, self.current_tier + 1):
            available.extend(b for b in TIERS.get(tier, []) if b in self.stats)
        if not available:
            available = list(self.stats.keys())

        weights: list[float] = []
        for b in available:
            s = self.stats[b]
            w = 1.0 - s.mastery + 0.1
            weights.append(max(0.05, w))
        bug = random.choices(available, weights=weights, k=1)[0]

        base = (tier_for(bug) - 1) / max(1, max(TIERS.keys()) - 1)
        mastery_bump = self.stats[bug].mastery * 0.3
        difficulty = min(1.0, base + mastery_bump)
        return bug, difficulty
