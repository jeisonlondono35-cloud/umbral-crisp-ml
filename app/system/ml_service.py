"""Activity 2 - the Machine Learning step INSIDE the business process.

The models are the ones chosen in Activity 1 (models/*.joblib + models/model_card.json). Nothing is
retrained here and the features are built by the SAME functions used in training/serving
(src/umbral_ml/predict.py: action_row, hit_row), so the preprocessing and the input structure are preserved.

Process step "Analyze session" (analysts):
  1. Next-action model, step by step over the recorded actions of the session (prequential: predict
     action i from actions 0..i-1, then compare with the REAL action i). Result: how predictable this
     player is for the bosses (accuracy) against a simple baseline (always the player's usual action).
  2. Hit model: P(a boss attack hits) in the player's usual situation, averaged over the four guardian
     attacks. Result: how exposed the player is.
  3. Decision rule (DECISION_RULE below) -> a difficulty Recommendation that an analyst approves or rejects;
     an approved one is applied to the player's account. The next analyzed session of the player is
     compared with the previous one to say if the change worked (outcome tracking).
Every call is stored in the Inference table (date, user, model, version, input, result, latency).
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.umbral_ml import predict as P
from src.umbral_ml.config import ACTIONS, PROC

from .models import DIFFS, Inference, PlaySession, Recommendation, audit, db, now

GUARDIAN_MOVES = ["dive", "feint", "volley", "special"]
MAX_STEPS = 400                    # longest action list analyzed (a long boss fight is ~150 actions)
PAST_CAP = 1000                    # earlier actions of the player fed as history (keeps one analysis < ~2 s)

# Thresholds of the decision rule (documented in the report and shown on the page)
# Risk thresholds = quartiles of the hit model's own predictions for the four guardian attacks on the Activity 1
# TEST players (p25 0.221, median 0.276, p75 0.362; real hit rate 0.30), so "high" means the top quarter.
RISK_HIGH = 0.36
RISK_LOW = 0.22
DEATHS_HIGH = 3.0                  # deaths per 10 minutes
DEATHS_VERY_HIGH = 5.0
DEATHS_LOW = 1.5

DECISION_RULE = [
    f"LOWER one level when the player lost or died {DEATHS_HIGH:g}+ times per 10 min AND the hit model says boss "
    f"attacks hit them {RISK_HIGH:.0%}+ of the time, or when they die {DEATHS_VERY_HIGH:g}+ times per 10 min.",
    f"RAISE one level when the player won, died at most {DEATHS_LOW:g} time per 10 min AND the hit risk is "
    f"{RISK_LOW:.0%} or less.",
    "KEEP the difficulty in any other case (no recommendation is opened).",
    "The predictability of the player (next-action accuracy) is shown to explain WHY the bosses read them; "
    "it does not change the difficulty by itself.",
]


def model_version() -> str:
    c = P.card()
    return f"{c.get('source', '?')} {str(c.get('trained_at', '?'))[:19]}"


def model_names() -> dict:
    c = P.card()
    return {"next": c.get("next_action", {}).get("model", "?"), "hit": c.get("hit", {}).get("model", "?")}


def _log(kind: str, user_id, player_id, session_id, model: str, inp: dict, out: dict, result: str, ms: float,
         checked=None, correct=None) -> Inference:
    inf = Inference(kind=kind, user_id=user_id, player_id=player_id, session_id=session_id, model_name=model,
                    model_version=model_version(), input_json=json.dumps(inp, default=str)[:20000],
                    output_json=json.dumps(out, default=str)[:20000], result=result[:120], latency_ms=round(ms, 2),
                    n_checked=checked, n_correct=correct)
    db.session.add(inf)
    return inf


# ----------------------------------------------------------------------------------------- validation
def parse_actions(text: str) -> list[str]:
    """Business rule: a session needs 5..400 actions, all of them one of the six game actions."""
    acts = [a.strip().lower() for a in (text or "").replace(";", ",").replace("\n", ",").split(",") if a.strip()]
    bad = sorted({a for a in acts if a not in ACTIONS})
    if bad:
        raise ValueError(f"Unknown actions: {', '.join(bad)}. Use only: {', '.join(ACTIONS)}.")
    if len(acts) < 5:
        raise ValueError("A session needs at least 5 combat actions to be analyzed.")
    if len(acts) > MAX_STEPS:
        raise ValueError(f"A session can have at most {MAX_STEPS} actions.")
    return acts


def _ctx(s: PlaySession) -> dict:
    return {"w": min(int(s.world), 5), "mode": s.mode, "diff": int(s.difficulty), "hp": float(max(1, s.hp_end)),
            "ne": s.enemy_distance if s.enemy_distance is not None else None, "boss": bool(s.boss_fight)}


# ------------------------------------------------------------------------------------- next action
def player_past(s: PlaySession) -> list[str]:
    """Actions of the SAME player in earlier sessions (oldest first). In training the features hist_* and mk_*
    describe the player's whole past, not only the current session (prepare.action_features); without it the
    accuracy fell from ~54% to ~50% on the Activity 1 test players. Capped at PAST_CAP actions."""
    earlier = (PlaySession.query.filter(PlaySession.player_id == s.player_id, PlaySession.id != s.id,
                                        db.or_(PlaySession.played_at < s.played_at,
                                               db.and_(PlaySession.played_at == s.played_at, PlaySession.id < s.id)))
               .order_by(PlaySession.played_at, PlaySession.id).all())
    past = [a for e in earlier for a in e.action_list]
    return past[-PAST_CAP:]


def _next_probas(acts: list[str], ctx: dict, past: list[str] | None = None) -> list[dict]:
    """P(next action) after every prefix acts[:i], i = 1..len(acts), with the player's past before it.
    Batched: one predict_proba call."""
    b = P.load("next_action")
    past = past or []
    rows = []
    for i in range(1, len(acts) + 1):
        r = P.action_row((past + acts[:i])[-PAST_CAP:], ctx)
        if i == 1:
            r["prev2"] = "start"           # a session starts a new "phrase" (as in training)
        rows.append(r)
    if b["kind"] == "sklearn":
        X = pd.DataFrame(rows)[b["features"]]
        with threadpool_limits(1):
            M = b["model"].predict_proba(X)
        out = []
        for r in M:
            d = {c: float(p) for c, p in zip(b["classes"], r)}        # by NAME (Activity 1 bug: column order)
            out.append({a: d.get(a, 0.0) for a in ACTIONS})
        return out
    return [P.predict_next((past + acts[:i])[-PAST_CAP:], ctx)["proba"] for i in range(1, len(acts) + 1)]


def analyze_session(s: PlaySession, user_id: int) -> dict:
    """Run both models on a session, store the inferences and return the analysis."""
    acts = s.action_list
    ctx = _ctx(s)
    t0 = time.perf_counter()
    past = player_past(s)
    probas = _next_probas(acts, ctx, past)
    # prequential comparison with the REAL next action
    checked = correct = base_ok = 0
    seen: Counter = Counter()
    confusion: Counter = Counter()
    for i in range(1, len(acts)):
        seen[acts[i - 1]] += 1
        p = probas[i - 1]
        guess = max(p, key=p.get)
        checked += 1
        correct += guess == acts[i]
        base_ok += seen.most_common(1)[0][0] == acts[i]
        confusion[(acts[i], guess)] += 1
    last = probas[-1]
    nxt = max(last, key=last.get)
    acc = correct / checked if checked else None
    base = base_ok / checked if checked else None
    ms_next = 1000 * (time.perf_counter() - t0)
    names = model_names()
    inf_next = _log("session_profile", user_id, s.player_id, s.id, names["next"],
                    {"actions": len(acts), "past_actions": len(past), **ctx}, {"accuracy": acc, "baseline": base, "next": nxt, "next_proba": last,
                                                    "confusion": {f"{a}>{g}": n for (a, g), n in confusion.items()}},
                    f"{correct}/{checked} right · next {nxt} {last[nxt]:.0%}", ms_next, checked, correct)

    # ---- hit risk in the player's usual situation
    share = Counter(acts)
    n = len(acts)
    sh = {a: share[a] / n for a in ACTIONS}
    ne = s.enemy_distance
    x = {"close": 1.0 if (ne is not None and ne < 70) or (ne is None and sh["melee"] + sh["block"] > 0.45) else 0.0,
         "far": 1.0 if (ne is not None and ne > 150) or (ne is None and sh["shoot"] > 0.45) else 0.0,
         "in_air": 1.0 if sh["jump"] > 0.2 else 0.0,
         "blocking": 1.0 if sh["block"] > 0.25 else 0.0,
         "moving": 1.0 if sh["dodge"] + sh["jump"] > 0.3 else 0.0,
         "low_hp": 1.0 if s.hp_end <= 2 else 0.0,
         "expects_evade": round(max(last["dodge"], last["jump"]), 3)}
    hctx = {"w": ctx["w"], "diff": ctx["diff"], "mode": ctx["mode"]}
    t1 = time.perf_counter()
    per_move = {m: P.predict_hit(m, x, hctx)["p_hit"] for m in GUARDIAN_MOVES}
    risk = float(np.mean(list(per_move.values())))
    worst = max(per_move, key=per_move.get)
    ms_hit = 1000 * (time.perf_counter() - t1)
    _log("hit", user_id, s.player_id, s.id, names["hit"], {"situation": x, **hctx},
         {"p_hit": per_move, "risk": risk}, f"risk {risk:.0%} · worst {worst} {per_move[worst]:.0%}", ms_hit)

    s.predictability, s.baseline, s.risk = acc, base, risk
    s.predicted_next, s.predicted_p = nxt, last[nxt]
    s.status = "analyzed"
    rec = decide(s, inf_next)
    track_outcome(s)
    return {"accuracy": acc, "baseline": base, "checked": checked, "correct": correct, "next": nxt,
            "next_proba": last, "share": sh, "risk": risk, "per_move": per_move, "situation": x,
            "recommendation": rec, "latency_ms": round(ms_next + ms_hit, 1)}


# ----------------------------------------------------------------------------------------- decision
def rule(diff: int, outcome: str, d10: float, risk: float) -> tuple[str, str]:
    struggling = outcome == "lost" or d10 >= DEATHS_HIGH
    if diff > 0 and ((struggling and risk >= RISK_HIGH) or d10 >= DEATHS_VERY_HIGH):
        return "lower", (f"Died {d10:.1f} times per 10 min ({outcome}) and boss attacks hit this player "
                         f"{risk:.0%} of the time: the fights are probably too hard or feel unfair.")
    if diff < 3 and outcome == "won" and d10 <= DEATHS_LOW and risk <= RISK_LOW:
        return "raise", (f"Won with {d10:.1f} deaths per 10 min and only {risk:.0%} hit risk: "
                         "the player can take a harder challenge.")
    return "keep", "The session is inside the expected range for this difficulty."


def decide(s: PlaySession, inf: Inference) -> Recommendation | None:
    action, reason = rule(int(s.player.difficulty), s.outcome, s.deaths_per_10, float(s.risk or 0))
    if action == "keep":
        return None
    cur = int(s.player.difficulty)
    new = max(0, cur - 1) if action == "lower" else min(3, cur + 1)
    # business rule: only ONE pending recommendation per player (an older one is superseded)
    for old in Recommendation.query.filter_by(player_id=s.player_id, status="pending").all():
        old.status = "rejected"
        old.decision_note = "Superseded by a newer analysis."
        old.decided_at = now()
    db.session.flush()
    rec = Recommendation(player_id=s.player_id, session_id=s.id, current_difficulty=cur, suggested_difficulty=new,
                         action=action, reason=reason, before_deaths10=s.deaths_per_10)
    db.session.add(rec)
    db.session.flush()
    rec.inference_id = inf.id if inf.id else None
    return rec


def apply_decision(rec: Recommendation, approve: bool, user_id: int, note: str) -> None:
    """Analyst decision. Approving changes the player's account difficulty (the business action)."""
    if rec.status != "pending":
        raise ValueError("This recommendation was already decided.")
    rec.decided_by, rec.decided_at, rec.decision_note = user_id, now(), (note or "")[:300]
    if not approve:
        rec.status = "rejected"
        audit("recommendation_rejected", f"rec {rec.id} player {rec.player_id}", user_id)
        return
    p = rec.player
    if p.difficulty != rec.current_difficulty:
        raise ValueError("The player's difficulty changed since the analysis; analyze a new session first.")
    p.difficulty = rec.suggested_difficulty
    rec.status = "applied"
    if rec.session:
        rec.session.status = "reviewed"
    audit("difficulty_changed", f"player {p.gamer_tag}: {DIFFS[rec.current_difficulty]} -> "
                                f"{DIFFS[rec.suggested_difficulty]} (rec {rec.id})", user_id)


def track_outcome(s: PlaySession) -> None:
    """Compare the real result after an applied change with the result before it."""
    for rec in Recommendation.query.filter_by(player_id=s.player_id, status="applied", after_session_id=None).all():
        if rec.session_id == s.id or not rec.decided_at or s.played_at < rec.decided_at:
            continue
        rec.after_session_id, rec.after_deaths10 = s.id, s.deaths_per_10
        if rec.action == "lower":
            rec.effective = s.deaths_per_10 < (rec.before_deaths10 or 0) or s.outcome == "won"
        else:
            rec.effective = s.deaths_per_10 <= DEATHS_HIGH and s.outcome != "quit"


# ----------------------------------------------------------------------------------------- monitoring
@lru_cache(maxsize=1)
def reference_shares() -> dict:
    """Share of each action in the TRAINING data of the model (the drift reference)."""
    try:
        df = pd.read_csv(PROC / "actions.csv", usecols=["target", "split"])
        t = df[df["split"] == "train"]["target"].value_counts(normalize=True)
        return {a: float(t.get(a, 0.0)) for a in ACTIONS}
    except (FileNotFoundError, ValueError, KeyError):
        return {a: 1 / len(ACTIONS) for a in ACTIONS}


def psi(ref: dict, cur: dict, eps: float = 1e-4) -> float:
    tot = 0.0
    for a in ACTIONS:
        r, c = max(ref.get(a, 0.0), eps), max(cur.get(a, 0.0), eps)
        tot += (c - r) * math.log(c / r)
    return tot


def psi_state(v: float | None) -> str:
    if v is None:
        return "no data"
    return "ALARM" if v > 0.2 else "WARNING" if v > 0.1 else "OK"
