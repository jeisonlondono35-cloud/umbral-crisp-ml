"""Exact Python replicas of the three ONLINE models that run inside the game (scripts/ml_brain.gd).

They are the "deployed baseline" of this project: every new candidate model is compared against
what the game already does. They learn while the player plays, so they are evaluated
PREQUENTIALLY (predict first, then learn), never with future information.
"""
from __future__ import annotations

import math

from .config import ACTIONS, DECAY, GAP_MS, HIT_LR, MIN_SUPPORT


class MarkovDecay:
    """Order-2 Markov chain with forgetting (falls back to order 1). Same rules as ml_brain.gd."""

    def __init__(self, decay: float = DECAY, order: int = 2, min_support: float = MIN_SUPPORT):
        self.decay, self.order, self.min_support = decay, order, min_support
        self.counts: dict[str, dict[str, float]] = {}
        self.prev1 = self.prev2 = "start"
        self.last_ms = -10**9
        self.made = self.right = 0
        self.acc = 0.0

    def _bump(self, ctx: str, a: str) -> None:
        d = self.counts.setdefault(ctx, {})
        for k in d:
            d[k] *= self.decay
        d[a] = d.get(a, 0.0) + 1.0

    def contexts(self, now: int) -> tuple[str, str]:
        if now - self.last_ms > GAP_MS:
            return "start>start", "start"
        return self.prev2 + ">" + self.prev1, self.prev1

    def proba(self, now: int) -> dict[str, float] | None:
        """Probability of each next action, or None if there is not enough evidence."""
        c2, c1 = self.contexts(now)
        ctxs = [c2, c1] if self.order == 2 else [c1]
        for ctx in ctxs:
            d = self.counts.get(ctx, {})
            tot = sum(d.values())
            if tot < self.min_support:
                continue
            return {a: d.get(a, 0.0) / tot for a in ACTIONS}
        return None

    def predict(self, now: int) -> tuple[str, float]:
        p = self.proba(now)
        if not p:
            return "", 0.0
        a = max(p, key=p.get)
        return a, p[a]

    def reliability(self) -> float:
        return max(0.0, min(1.0, self.acc * min(1.0, self.made / 20.0) * 1.4))

    def expects_evade(self, now: int) -> float:
        a, p = self.predict(now)
        return p * self.reliability() if a in ("dodge", "jump") else 0.0

    def observe(self, a: str, now: int) -> None:
        if now - self.last_ms > GAP_MS:
            self.prev1 = self.prev2 = "start"
        else:
            pa, _ = self.predict(now)
            if pa:
                self.made += 1
                ok = pa == a
                self.right += ok
                self.acc += (float(ok) - self.acc) * max(0.08, 1.0 / self.made)
        self._bump(self.prev2 + ">" + self.prev1, a)
        self._bump(self.prev1, a)
        self.prev2, self.prev1 = self.prev1, a
        self.last_ms = now


class OnlineLogit:
    """One logistic regression per boss attack, trained by SGD (the in-game contextual bandit)."""

    def __init__(self, lr: float = HIT_LR, n: int = 8):
        self.lr, self.n = lr, n
        self.w: dict[str, list[float]] = {}

    def _weights(self, m: str) -> list[float]:
        if m not in self.w:
            self.w[m] = [-1.0] + [0.0] * (self.n - 1)   # in-game prior: ~27% hit chance
        return self.w[m]

    def proba(self, m: str, x: list[float]) -> float:
        z = sum(wi * xi for wi, xi in zip(self._weights(m), x))
        z = max(-12.0, min(12.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def learn(self, m: str, x: list[float], y: float) -> None:
        w = self._weights(m)
        err = y - self.proba(m, x)
        for i in range(self.n):
            w[i] += self.lr * err * x[i]


class DodgeMean:
    """Running mean of the dodge offset, weight 1/min(n, 8) (in-game dodge model)."""

    def __init__(self):
        self.x = self.y = 0.0
        self.n = 0

    def predict(self) -> tuple[float, float]:
        k = self.n / (self.n + 4.0) if self.n else 0.0      # dodge_offset shrinkage
        return self.x * k, self.y * k

    def learn(self, ox: float, oy: float) -> None:
        self.n += 1
        a = 1.0 / min(self.n, 8)
        mag = math.hypot(ox, oy)
        if mag > 120.0:
            ox, oy = ox * 120.0 / mag, oy * 120.0 / mag
        self.x += (ox - self.x) * a
        self.y += (oy - self.y) * a
