"""PHASE 2 - DATA ENGINEERING.

Raw export (real or simulated, same format) -> quality profile (BEFORE) -> documented cleaning ->
feature engineering -> grouped train / validation / test split -> processed CSVs + data_report.json.

No-leakage rules:
  * every sequential feature of an event uses ONLY earlier events of the same player;
  * the split is by PLAYER (or by session when there are too few players): a player's events never
    appear in two partitions;
  * everything that is FITTED (imputation medians, one-hot categories, scaling) lives inside the
    sklearn Pipeline of each model, so it is fitted on the training partition only (train.py).
"""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
import pandas as pd

from .config import ACTIONS, GAP_MS, HIT_X, MIN_REAL_ACTIONS, MIN_REAL_PLAYERS, PROC, RAW, SEED, SIM
from .online import DodgeMean, MarkovDecay

BOSS_SCENES = ("Arena", "Umbral", "Espejo", "Panteon")
REAL_AVAILABLE: dict = {}


def load_export(source: str) -> tuple[list[dict], str]:
    """source = real | simulated | auto (real if there is enough of it)."""
    real_p, sim_p = RAW / "real_export.json", SIM / "sim_export.json"
    real = json.loads(real_p.read_text(encoding="utf-8"))["batches"] if real_p.exists() else []
    n_a = sum(1 for b in real for e in b["events"] if e.get("k") == "a")
    n_p = len({b["pid"] for b in real})
    REAL_AVAILABLE.update(actions=n_a, players=n_p, batches=len(real), min_actions=MIN_REAL_ACTIONS,
                          min_players=MIN_REAL_PLAYERS)
    if source == "auto":
        source = "real" if n_a >= MIN_REAL_ACTIONS and n_p >= MIN_REAL_PLAYERS else "simulated"
    if source == "real":
        return real, "REAL"
    return json.loads(sim_p.read_text(encoding="utf-8"))["batches"], "SIMULATED"


def flatten(batches: list[dict]) -> pd.DataFrame:
    rows = []
    for b in batches:
        for i, e in enumerate(b["events"]):
            r = {k: v for k, v in e.items() if k != "x"}
            if "x" in e:
                for name, v in zip(HIT_X, e["x"]):
                    r["x_" + name] = v
            r.update(batch_id=b["id"], pid=b["pid"], sid=b["sid"], ver=b.get("ver", ""),
                     created_at=b["created_at"], pos=i)
            rows.append(r)
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    return df


def profile(df: pd.DataFrame) -> dict:
    """Quality numbers shown as the BEFORE / AFTER evidence."""
    a = df[df.k == "a"]
    h = df[df.k == "h"]
    key = ["pid", "sid", "k", "t"]
    out = {
        "events": int(len(df)), "players": int(df.pid.nunique()), "sessions": int(df.sid.nunique()),
        "by_kind": {k: int(v) for k, v in df.k.value_counts().items()},
        "duplicate_events": int(df.duplicated(key).sum()),
        "missing": {c: int(df.loc[df.k == k, c].isna().sum()) for k, cols in
                    (("a", ["a", "hp", "ne", "gap"]), ("h", ["mv", "y"] + ["x_" + x for x in HIT_X]))
                    for c in cols if c in df},
        "ne_sentinel_9999": int((a.get("ne", pd.Series(dtype=float)) >= 9999).sum()),
        "invalid_action": int((~a.a.isin(ACTIONS)).sum()) if len(a) else 0,
        "gap_over_10min": int((a.gap >= 600000).sum()) if len(a) else 0,
        "action_classes": {k: int(v) for k, v in a.a.value_counts().items()} if len(a) else {},
        "hit_rate": float(h.y.mean()) if len(h) else None,
        "hit_by_move": {k: int(v) for k, v in h.mv.value_counts().items()} if len(h) else {},
    }
    return out


def clean(df: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    def step(name, why, before, after):
        log.append({"step": name, "why": why, "rows_before": int(before), "rows_after": int(after)})

    n = len(df)
    df = df.drop_duplicates(["pid", "sid", "k", "t", "a", "mv"] if "mv" in df else ["pid", "sid", "k", "t"])
    step("Drop duplicated events", "A batch re-uploaded after a lost server answer repeats the same "
         "(player, session, kind, time) events.", n, len(df))
    n = len(df)
    df = df[(df.k != "a") | df.a.isin(ACTIONS)]
    df = df[df.w.between(1, 5)]
    step("Drop inconsistent rows", "Unknown action names or a world outside 1..5 cannot come from the game.", n, len(df))
    n = len(df)
    if "ne" in df:
        df.loc[df["ne"] >= 9999, "ne"] = np.nan
    step("Distance sentinel 9999 -> missing", "9999 means 'no enemy found' (not a distance). It is set to "
         "missing and imputed with the TRAIN median inside each model pipeline.", n, len(df))
    df["hp"] = df.hp.clip(0, 12)
    if "gap" in df:
        df["gap"] = df.gap.clip(0, 600000)
    step("Clip outliers", "Life is clipped to 0..12 and the time gap to 10 min (the game already caps it); "
         "long pauses are kept: they mark a new 'phrase' (gap > 4 s).", len(df), len(df))
    # chronological order inside each player: session start, then time inside the session
    first = df.groupby("sid").created_at.transform("min")
    df = df.assign(_s=first).sort_values(["pid", "_s", "sid", "t", "pos"]).drop(columns="_s")
    df["seq"] = df.groupby("pid").cumcount()
    step("Sort chronologically per player", "Sequential features and the online models must only see the past.",
         len(df), len(df))
    return df.reset_index(drop=True)


def action_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player action. Target = the action. Features use only the player's past."""
    rows = []
    for pid, g in df[df.k == "a"].groupby("pid", sort=False):
        mk = MarkovDecay()
        hist = Counter()
        prev1 = prev2 = "start"
        last_t, last_sid = None, None
        clock = 0
        for r in g.itertuples(index=False):
            if r.sid != last_sid:            # a new session never continues the previous phrase
                clock += 10**7
                last_t = None
            now = clock + int(r.t)
            start = last_t is None or now - last_t > GAP_MS
            if start:
                prev1 = prev2 = "start"
            n_hist = sum(hist.values())
            pr = mk.proba(now) or {}
            row = {"pid": pid, "sid": r.sid, "seq": r.seq, "target": r.a,
                   "prev1": prev1, "prev2": prev2, "is_start": int(start),
                   "w": int(r.w), "mode": r.m, "diff": int(r.d), "hp": float(r.hp),
                   "ne": r.ne, "boss": int(any(s in str(r.sc) for s in BOSS_SCENES)),
                   "log_hist": math.log1p(n_hist),
                   "game_pred": r.p if isinstance(r.p, str) else "", "game_pp": float(r.pp or 0)}
            for a in ACTIONS:
                row["hist_" + a] = (hist[a] + 1) / (n_hist + 6)      # Laplace-smoothed past frequency
                row["mk_" + a] = pr.get(a, np.nan)                  # in-game Markov replica, before learning
            rows.append(row)
            mk.observe(r.a, now)
            hist[r.a] += 1
            prev2, prev1 = prev1, r.a
            last_t, last_sid = now, r.sid
    return pd.DataFrame(rows)


def hit_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df[df.k == "h"].copy()
    cols = ["pid", "sid", "seq", "mv", "w", "m", "d", "hp", "y", "p"] + ["x_" + x for x in HIT_X]
    h = h[cols].rename(columns={"m": "mode", "d": "diff", "p": "game_p"})
    h["family"] = h.mv.str.split("_").str[0].where(h.mv.str.contains("_"), "guardian")
    return h.reset_index(drop=True)


def dodge_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df[df.k == "v"].groupby("pid", sort=False):
        dm = DodgeMean()
        for r in g.itertuples(index=False):
            gx, gy = dm.predict()
            rows.append({"pid": pid, "sid": r.sid, "seq": r.seq, "w": int(r.w), "diff": int(r.d),
                         "ox": float(r.ox), "oy": float(r.oy), "game_ox": gx, "game_oy": gy, "n_prev": dm.n})
            dm.learn(float(r.ox), float(r.oy))
    return pd.DataFrame(rows)


def split_groups(acts: pd.DataFrame, seed: int = SEED) -> tuple[dict, dict]:
    """Assign every player (or session) to train / validation / test: 60 / 20 / 20 of groups."""
    n_players = acts.pid.nunique()
    by = "pid" if n_players >= 10 else "sid"
    groups = sorted(acts[by].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test, n_val = max(1, round(0.2 * n)), max(1, round(0.2 * n))
    part = {g: "test" for g in groups[:n_test]}
    part.update({g: "val" for g in groups[n_test:n_test + n_val]})
    part.update({g: "train" for g in groups[n_test + n_val:]})
    info = {"group_by": by, "groups": n, "train": n - n_test - n_val, "val": n_val, "test": n_test,
            "why": ("Players are split as whole units: a model must work for a NEW player, and the same "
                    "player's sequence in two partitions would leak his habits." if by == "pid" else
                    "Fewer than 10 players: whole SESSIONS are split instead (documented limitation).")}
    return part, info


def main(source: str = "auto") -> dict:
    batches, label = load_export(source)
    raw = flatten(batches)
    before = profile(raw)
    log: list[dict] = []
    df = clean(raw, log)
    after = profile(df)
    acts, hits, dodges = action_features(df), hit_features(df), dodge_features(df)
    part, split_info = split_groups(acts)
    by = split_info["group_by"]
    for t in (acts, hits, dodges):
        t["split"] = t[by].map(part).fillna("train")
    log.append({"step": "Feature engineering", "why": "prev1/prev2 (last two actions, 'start' after 4 s), "
                "is_start, Laplace-smoothed past frequency of each action (hist_*), log of history length, "
                "context (world, mode, difficulty, life, enemy distance, boss fight) for the next-action task; "
                "8 in-game situation features + attack id/family for the hit task.",
                "rows_before": int(len(df)), "rows_after": int(len(acts) + len(hits) + len(dodges))})
    raw.head(25).to_csv(PROC / "sample_raw.csv", index=False)
    acts.to_csv(PROC / "actions.csv", index=False)
    hits.to_csv(PROC / "hits.csv", index=False)
    dodges.to_csv(PROC / "dodges.csv", index=False)
    acts.head(25).to_csv(PROC / "sample_actions.csv", index=False)
    report = {"source": label, "real_available": REAL_AVAILABLE, "before": before, "after": after, "transformations": log, "split": split_info,
              "rows": {"actions": len(acts), "hits": len(hits), "dodges": len(dodges)},
              "split_rows": {t: {s: int(v) for s, v in d.split.value_counts().items()}
                             for t, d in (("actions", acts), ("hits", hits), ("dodges", dodges))}}
    (PROC / "data_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"[{label}] actions {len(acts)}  hits {len(hits)}  dodges {len(dodges)}  split {split_info}")
    return report


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "auto")
