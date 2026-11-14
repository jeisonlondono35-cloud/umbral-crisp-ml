"""PHASE 5 - the saved models + their preprocessing, used by the Flask demo and the monitoring.

`predict_next(history, context)` rebuilds EXACTLY the features of prepare.action_features from the
player's recent actions (so training and serving share one definition), and `predict_hit(...)`
does the same for the boss-attack task. The preprocessing (imputation, one-hot, scaling) is inside
each saved sklearn Pipeline.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .config import ACTIONS, HIT_X, MODELS
from threadpoolctl import threadpool_limits

from .online import MarkovDecay


@lru_cache(maxsize=None)
def load(name: str):
    return joblib.load(MODELS / f"{name}.joblib")


@lru_cache(maxsize=None)
def card() -> dict:
    return json.loads((MODELS / "model_card.json").read_text(encoding="utf-8"))


def _markov_from(history: list[str], decay: float, order: int) -> MarkovDecay:
    mk = MarkovDecay(decay=decay, order=order)
    t = 0
    for a in history:
        t += 500                       # recent actions inside one fight (no 4 s gap)
        mk.observe(a, t)
    mk._now = t + 500                  # moment of the prediction
    return mk


def action_row(history: list[str], ctx: dict) -> dict:
    """Features of the NEXT action given the past actions (oldest first) and the context."""
    start = bool(ctx.get("is_start", False)) or not history
    prev1 = "start" if start else history[-1]
    prev2 = "start" if start or len(history) < 2 else history[-2]
    hist = Counter(history)
    n = len(history)
    mk = _markov_from(history, 0.97, 2)
    pr = (mk.proba(mk._now) or {}) if not start else (mk.proba(mk.last_ms + 10**6) or {})
    row = {"prev1": prev1, "prev2": prev2, "is_start": int(start), "w": int(ctx.get("w", 1)),
           "mode": ctx.get("mode", "story"), "diff": int(ctx.get("diff", 1)), "hp": float(ctx.get("hp", 5)),
           "ne": float(ctx["ne"]) if ctx.get("ne") not in (None, "") else np.nan,
           "boss": int(bool(ctx.get("boss", False))), "log_hist": math.log1p(n)}
    for a in ACTIONS:
        row["hist_" + a] = (hist[a] + 1) / (n + 6)
        row["mk_" + a] = pr.get(a, np.nan)
    return row


def _sk_proba(bundle, row: dict) -> dict:
    X = pd.DataFrame([row])[bundle["features"]]
    with threadpool_limits(1):          # one row: thread start-up costs more than it saves
        P = bundle["model"].predict_proba(X)[0]
    return {c: float(p) for c, p in zip(bundle["classes"], P)}


def predict_next(history: list[str], ctx: dict) -> dict:
    history = [a for a in history if a in ACTIONS]
    b = load("next_action")
    row = action_row(history, ctx)
    if b["kind"] == "sklearn":
        p = _sk_proba(b, row)
    else:
        mk = _markov_from(history, b["decay"], b["order"])
        p = mk.proba(mk._now) if not row["is_start"] else None
        p = p or b["prior"]
    p = {a: p.get(a, 0.0) for a in ACTIONS}
    best = max(p, key=p.get)
    # what the game's current model would say, for comparison
    g = _markov_from(history, 0.97, 2)
    gp = g.proba(g._now) if not row["is_start"] else None
    return {"model": b["name"], "prediction": best, "probability": p[best], "proba": p,
            "in_game": ({"prediction": max(gp, key=gp.get), "probability": max(gp.values())}
                        if gp else {"prediction": None, "probability": None}),
            "features": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}}


def hit_row(move: str, x: dict, ctx: dict) -> dict:
    row = {"mv": move, "family": move.split("_")[0] if "_" in move else "guardian",
           "mode": ctx.get("mode", "story"), "w": int(ctx.get("w", 1)), "diff": int(ctx.get("diff", 1))}
    for name in HIT_X[1:]:
        row["x_" + name] = float(x.get(name, 0.0))
    return row


def predict_hit(move: str, x: dict, ctx: dict) -> dict:
    b = load("hit")
    row = hit_row(move, x, ctx)
    p = float(b["model"].predict_proba(pd.DataFrame([row])[b["features"]])[0, 1])
    return {"model": card()["hit"]["model"], "p_hit": p, "features": row, "known_move": move in b["moves"]}


def predict_dodge(history: list[tuple[float, float]]) -> dict:
    b = load("dodge")
    gx, gy = b["global"]
    k = b["k"]
    n = len(history)
    sx = sum(h[0] for h in history)
    sy = sum(h[1] for h in history)
    return {"model": card()["dodge"]["model"], "ox": float((sx + k * gx) / (n + k)), "oy": float((sy + k * gy) / (n + k))}
