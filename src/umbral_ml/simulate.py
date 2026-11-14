"""SIMULATED players (clearly marked as simulated everywhere they are used).

Why it exists: the real gameplay log (scripts/ml_log.gd in the game -> Supabase table
umbral_ml_log) only started recording in November 2026. The simulator writes events in EXACTLY
the same format as the real export, so the whole pipeline can be built, tested and demonstrated
before enough real data arrives, and so monitoring can be tested with simulated operational data.
When the real export has enough data, `pipeline.py --source real` re-runs everything on it.

The simulated behaviour is grounded in how the game works (same actions, same 8 situation
features, same boss attacks, same in-game online models filling the `p`/`pp` fields), with
per-player habits, context effects (low life -> defensive, far enemy -> bow), slow habit drift
across sessions and a few realistic data-quality problems (re-uploaded batches, missing distance).
"""
from __future__ import annotations

import json
import math
import random
from datetime import datetime, timedelta, timezone

import numpy as np

from .config import ACTIONS, SEED, SIM
from .online import DodgeMean, MarkovDecay, OnlineLogit

BOSS_MOVES = ["dive", "volley", "special", "summon", "feint", "repel",
              "lobo_charge", "lobo_zarpas", "lobo2_breath", "espejo_shot"]
# true effect of each situation feature on "the attack hits you", per attack family (logit scale)
BASE_LOGIT = {"dive": -0.6, "volley": -0.9, "special": -0.4, "summon": -1.6, "feint": -0.7,
              "repel": -0.9, "lobo_charge": -0.3, "lobo_zarpas": -0.5, "lobo2_breath": -0.2,
              "espejo_shot": -0.8}
SCENES = {1: "Platformer", 2: "BosquePlat", 3: "VolcanPlat", 4: "CiudadPlat", 5: "MarPlat"}


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _true_hit_logit(m: str, x: list[float], skill: float) -> float:
    close, far, air, block, moving, low, exp_ev = x[1:]
    z = BASE_LOGIT[m] - skill
    z += -1.6 * block if m != "feint" else 1.3 * block          # a feint punishes blocking
    z += {"dive": -0.9, "lobo_charge": -0.7}.get(m, -0.2) * moving
    z += (-1.4 if m in ("lobo_zarpas", "repel") else 0.3) * air  # jumping avoids ground waves
    z += (0.9 if m in ("repel", "lobo_charge", "lobo_zarpas") else -0.2) * close
    z += (0.8 if m in ("volley", "espejo_shot", "lobo2_breath") else -0.3) * far
    z += 0.5 * low + 0.9 * exp_ev * (m in ("dive", "lobo_charge"))
    z += 0.6 * close * block * (m == "repel")                    # interaction trees can find
    return z


def simulate(n_players: int = 48, seed: int = SEED, drift: float = 0.0,
             start: datetime | None = None, days: float = 10.0) -> list[dict]:
    """`drift` > 0 simulates a change in how people play (e.g. after a game update): more dodging
    and fights from farther away. Used only by the monitoring test (monitor.py)."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    batches: list[dict] = []
    t0 = start or datetime(2026, 11, 1, tzinfo=timezone.utc)
    bid = 0
    for _ in range(n_players):
        pid = _hex(rng, 16)
        archetype = rng.choice(["ranged", "ranged", "melee", "block", "mixed"])
        base = nrng.dirichlet([2.0] * 6) * 0.5
        base[{"ranged": 0, "melee": 1, "block": 2, "mixed": 3}[archetype]] += 0.5
        habit_next = {a: rng.choice(ACTIONS) for a in ACTIONS + ["start"]}
        habit_str = {a: rng.uniform(0.6, 2.2) for a in habit_next}
        combos = {(rng.choice(ACTIONS), rng.choice(ACTIONS)): rng.choice(ACTIONS) for _ in range(3)}
        skill = nrng.normal(0, 0.6)
        diff = rng.choice([0, 1, 1, 1, 2, 2, 3])
        dodge_style = (nrng.normal(25, 30), nrng.normal(0, 35))
        brain, bandit, dmodel = MarkovDecay(), OnlineLogit(), DodgeMean()
        n_sessions = rng.randint(2, 8) if days > 3 else rng.randint(1, 2)
        day = t0 + timedelta(hours=rng.uniform(0, 24 * days))
        world = 1
        for s in range(n_sessions):
            sid = _hex(rng, 12)
            mode = rng.choices(["story", "ole", "pan"], [0.7, 0.25, 0.05])[0]
            if mode == "story" and s and rng.random() < 0.6:
                world = min(5, world + 1)
            w = world if mode == "story" else rng.randint(1, 5)
            # habits drift a little between sessions (non-stationary players)
            base = 0.9 * base + 0.1 * nrng.dirichlet([2.0] * 6)
            events: list[dict] = []
            t = rng.randint(2000, 20000)
            for _f in range(rng.randint(4, 18)):
                boss = rng.random() < 0.35
                scene = ("Arena" if boss else SCENES[w]) if mode == "story" else (
                    "Oleadas" if mode == "ole" else "UmbralArena")
                hp = 5 + w // 2
                n_act = max(3, int(nrng.geometric(1 / (22 if boss else 10))))
                ne = rng.uniform(30, 200)
                for i in range(n_act):
                    gap = int(nrng.exponential(650) + 120) if i else rng.randint(5000, 40000)
                    t += gap
                    ne = float(np.clip(ne + nrng.normal(0, 35) + 6 * drift, 12, 219))
                    p1, p2 = brain.prev1, brain.prev2
                    if gap > 4000:
                        p1 = p2 = "start"
                    z = np.log(base + 1e-3)
                    z[ACTIONS.index(habit_next[p1])] += habit_str[p1] * (1 - 0.35 * drift)
                    if (p2, p1) in combos:
                        z[ACTIONS.index(combos[(p2, p1)])] += 1.6
                    if hp <= 2:
                        z[[2, 3, 5]] += [0.8, 0.6, 0.5]
                    if ne < 60:
                        z[[1, 2]] += [0.8, 0.3]
                    elif ne > 150:
                        z[0] += 0.9
                    if boss:
                        z[[3, 4]] += [0.5, 0.4]
                    z[2] += 0.15 * diff
                    z[3] += 1.2 * drift
                    a = ACTIONS[int(nrng.choice(6, p=_softmax(z)))]
                    pa, pp = brain.predict(t)
                    events.append({"k": "a", "t": t, "w": w, "sc": scene, "m": mode, "d": diff, "hp": hp,
                                   "a": a, "p": pa, "pp": round(pp, 3), "gap": min(gap, 600000),
                                   "ne": int(ne) if rng.random() > 0.03 else 9999})
                    brain.observe(a, t)
                    if boss and rng.random() < 0.45:
                        m = rng.choice(BOSS_MOVES[:6] if w < 5 else BOSS_MOVES)
                        x = [1.0, float(ne < 70), float(ne > 150), float(a == "jump" or rng.random() < 0.1),
                             float(a == "block" and rng.random() < 0.85), float(a in ("dodge", "jump") or rng.random() < 0.35),
                             float(hp <= 2), round(brain.expects_evade(t), 3)]
                        y = int(rng.random() < 1 / (1 + math.exp(-_true_hit_logit(m, x, skill))))
                        events.append({"k": "h", "t": t + 300, "w": w, "sc": scene, "m": mode, "d": diff, "hp": hp,
                                       "mv": m, "x": x, "y": y, "p": round(bandit.proba(m, x), 3)})
                        bandit.learn(m, x, y)
                        if y:
                            hp = max(1, hp - 1)
                        if m in ("dive", "lobo_charge") and rng.random() < 0.7:
                            ox = float(np.clip(dodge_style[0] + nrng.normal(0, 18), -120, 120))
                            oy = float(np.clip(dodge_style[1] * rng.choice([1, -1, 1]) + nrng.normal(0, 18), -120, 120))
                            events.append({"k": "v", "t": t + 200, "w": w, "sc": scene, "m": mode, "d": diff,
                                           "hp": hp, "ox": round(ox, 1), "oy": round(oy, 1)})
                            dmodel.learn(ox, oy)
            # upload in batches of up to 150 events, as the game does
            for j in range(0, len(events), 150):
                chunk = events[j:j + 150]
                bid += 1
                created = day + timedelta(milliseconds=chunk[-1]["t"])
                b = {"id": bid, "pid": pid, "sid": sid, "ver": "sim", "n": len(chunk), "events": chunk,
                     "created_at": created.isoformat(), "source": "SIMULATED"}
                batches.append(b)
                if rng.random() < 0.02:          # a re-upload after a lost answer: duplicated batch
                    bid += 1
                    batches.append(dict(b, id=bid))
            day += timedelta(hours=rng.uniform(5, 60))
    return batches


def main() -> None:
    batches = simulate()
    out = SIM / "sim_export.json"
    out.write_text(json.dumps({"source": "SIMULATED", "note": "Simulated players, NOT real data.",
                               "batches": batches}), encoding="utf-8")
    n = sum(b["n"] for b in batches)
    print(f"SIMULATED: {len(batches)} batches, {n} events -> {out}")


if __name__ == "__main__":
    main()
