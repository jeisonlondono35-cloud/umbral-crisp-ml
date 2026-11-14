"""PHASE 6 - MONITORING AND MAINTENANCE: initial monitoring test.

1) SIMULATED operations (marked SIMULATED): 8 weeks of new players; from week 4 the way people play
   changes on purpose (more dodging, fights from farther away), as after a game update. Each week runs
   the SAME pipeline as production (prepare.flatten/clean/action_features -> saved model) and records:
     - data drift: Population Stability Index (PSI) of the inputs and of the target vs. the training data
     - performance: accuracy of the chosen model and of the in-game model on that week
     - alerts from the thresholds documented in the app (PSI > 0.2, accuracy drop > 5 points)
2) REAL operational indicators (when available): in-game accuracy over all accounts
   (umbral_ml_public) and events per day of the anonymous log (umbral_ml_log_stats).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from . import plots
from .config import ACTIONS, MON, PROC, RAW, REPORTS
from .predict import load
from .prepare import action_features, clean, flatten
from .simulate import simulate

PSI_WARN, PSI_ALERT, ACC_DROP, BASE_WEEKS = 0.10, 0.20, 0.05, 3
DRIFT_BY_WEEK = [0, 0, 0, 0.5, 0.9, 1.3, 1.3, 1.3]


def psi(ref: pd.Series, cur: pd.Series, bins=None) -> float:
    """Population Stability Index; categorical if bins is None, else numeric bins from the REFERENCE."""
    if bins is None:
        cats = sorted(set(ref.dropna()) | set(cur.dropna()))
        r = ref.value_counts(normalize=True).reindex(cats).fillna(0).values
        c = cur.value_counts(normalize=True).reindex(cats).fillna(0).values
    else:
        r = np.histogram(ref.dropna(), bins)[0] / max(1, ref.notna().sum())
        c = np.histogram(cur.dropna(), bins)[0] / max(1, cur.notna().sum())
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(((c - r) * np.log(c / r)).sum())


def run() -> dict:
    res = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    test_acc = res["A"]["test"][res["A"]["chosen"]]["accuracy"]
    b = load("next_action")
    weeks, frames = [], []
    t0 = datetime(2026, 12, 1, tzinfo=timezone.utc)
    for w, drift in enumerate(DRIFT_BY_WEEK, start=1):
        batches = simulate(n_players=25, seed=1000 + w, drift=drift, start=t0 + timedelta(days=7 * (w - 1)), days=6)
        df = clean(flatten(batches), [])
        cur = action_features(df)
        cur["prev1"] = cur.prev1.astype(str)
        cur["prev2"] = cur.prev2.astype(str)
        if b["kind"] == "sklearn":
            P = b["model"].predict_proba(cur[b["features"]])
            pred = np.array(b["classes"])[P.argmax(1)]
        else:
            from .train import markov_prequential
            prior = np.array([b["prior"][a] for a in ACTIONS])
            pred = np.array(ACTIONS)[markov_prequential(cur, b["decay"], b["order"], prior).argmax(1)]
        m = cur.game_pred != ""
        row = {"week": w, "start": (t0 + timedelta(days=7 * (w - 1))).date().isoformat(), "simulated": True,
               "drift_injected": drift, "actions": int(len(cur)), "players": int(cur.pid.nunique()),
               "accuracy": float((pred == cur.target).mean()),
               "in_game_accuracy": float((cur.game_pred[m] == cur.target[m]).mean()),
               "share_dodge": float((cur.target == "dodge").mean())}
        frames.append(cur)
        roll = float(np.mean([r["accuracy"] for r in weeks[-1:]] + [row["accuracy"]]))
        row["accuracy_2w"] = roll        # one week of 25 players is noisy: players differ a lot
        weeks.append(row)
    # ALERTS. The operational baseline is the mean accuracy of the first BASE_WEEKS weeks after deployment
    # (the test accuracy is not used: production players have shorter histories than the test players).
    # REFERENCE WINDOW = the first BASE_WEEKS weeks after deployment (same for any training source)
    ref = pd.concat(frames[:BASE_WEEKS])
    ne_bins = np.quantile(ref["ne"].dropna(), np.linspace(0, 1, 11))
    ne_bins[0], ne_bins[-1] = -1, 10**4
    base = float(np.mean([r["accuracy"] for r in weeks[:BASE_WEEKS]]))
    for row, cur in zip(weeks, frames):
        row.update(psi_target=psi(ref.target, cur.target), psi_ne=psi(ref["ne"], cur["ne"], ne_bins),
                   psi_mode=psi(ref["mode"], cur["mode"]), psi_diff=psi(ref["diff"], cur["diff"]))
        worst = max(row["psi_target"], row["psi_ne"], row["psi_mode"], row["psi_diff"])
        alerts = []
        if worst > PSI_ALERT:
            alerts.append(f"DATA DRIFT: PSI {worst:.2f} > {PSI_ALERT}")
        elif worst > PSI_WARN:
            alerts.append(f"drift warning: PSI {worst:.2f} > {PSI_WARN}")
        if row["week"] > BASE_WEEKS and row["accuracy_2w"] < base - ACC_DROP:
            alerts.append(f"PERFORMANCE DROP: 2-week accuracy {row['accuracy_2w']:.3f} < baseline {base:.3f} - {ACC_DROP}")
        row["alerts"] = alerts
        row["status"] = ("RETRAIN" if any(a.startswith(("DATA", "PERF")) for a in alerts)
                         else ("WATCH" if alerts else "OK"))
        print(f"week {row['week']}: acc {row['accuracy']:.3f} (2w {row['accuracy_2w']:.3f})  "
              f"psi_target {row['psi_target']:.3f}  psi_ne {row['psi_ne']:.3f}  {row['status']}")
    # real indicators
    real = {}
    for f in ("ml_public.json", "log_stats.json"):
        p = RAW / f
        if p.exists():
            real[f[:-5]] = json.loads(p.read_text(encoding="utf-8"))
    first = lambda pre: next((r["week"] for r in weeks if any(a.startswith(pre) for a in r["alerts"])), None)  # noqa: E731
    out = {"first_drift_alert": first("DATA"), "first_perf_alert": first("PERF"), "note": "Operational weeks are SIMULATED (generated players with injected drift). "
                   "Real indicators come from the game's Supabase project.",
           "thresholds": {"psi_warn": PSI_WARN, "psi_alert": PSI_ALERT, "acc_drop": ACC_DROP,
                          "test_accuracy": test_acc, "baseline_weeks": BASE_WEEKS, "baseline_accuracy": base}, "weeks": weeks, "real": real}
    (MON / "monitoring_simulated.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    with open(MON / "monitoring_log_simulated.jsonl", "w", encoding="utf-8") as f:
        for r in weeks:
            f.write(json.dumps({"ts": r["start"], "source": "SIMULATED", **r}) + "\n")
    _figures(out)
    return out


def _figures(out: dict) -> None:
    import matplotlib.pyplot as plt
    W = out["weeks"]
    x = [r["week"] for r in W]
    th = out["thresholds"]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(x, [r["accuracy_2w"] for r in W], color=plots.CHOSEN, lw=2, marker="o", ms=4,
            label="chosen model (2-week mean)")
    ax.plot(x, [r["in_game_accuracy"] for r in W], color=plots.SERIES[1], lw=2, marker="o", ms=4, label="in-game model")
    ax.axhline(th["baseline_accuracy"] - th["acc_drop"], color=plots.INK2, lw=1, ls="--")
    ax.text(x[0], th["baseline_accuracy"] - th["acc_drop"] + 0.004, "retrain line (weeks 1-3 baseline - 5 points)",
            fontsize=7.5, color=plots.INK2)
    ax.axvspan(3.5, x[-1] + 0.5, color=plots.GRID, alpha=0.5, lw=0)
    ax.text(3.6, ax.get_ylim()[0] + 0.005, "behaviour change injected", fontsize=7.5, color=plots.INK2)
    ax.set_xlabel("week (SIMULATED operations)")
    ax.set_ylabel("accuracy")
    ax.set_title("Monitoring test: weekly accuracy - SIMULATED data")
    ax.legend(loc="upper right", fontsize=7.5)
    plots.save(fig, "fig_mon_accuracy")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(x, [r["psi_target"] for r in W], color=plots.CHOSEN, lw=2, marker="o", ms=4, label="PSI of the action mix")
    ax.plot(x, [r["psi_ne"] for r in W], color=plots.SERIES[2], lw=2, marker="o", ms=4, label="PSI of enemy distance")
    ax.axhline(th["psi_alert"], color=plots.INK2, lw=1, ls="--")
    ax.text(x[0], th["psi_alert"] + 0.01, "alert 0.20", fontsize=7.5, color=plots.INK2)
    ax.axhline(th["psi_warn"], color=plots.INK2, lw=1, ls=":")
    ax.text(x[0], th["psi_warn"] + 0.01, "warning 0.10", fontsize=7.5, color=plots.INK2)
    ax.set_xlabel("week (SIMULATED operations)")
    ax.set_ylabel("PSI")
    ax.set_title("Monitoring test: data drift - SIMULATED data")
    ax.legend(loc="upper left", fontsize=7.5)
    plots.save(fig, "fig_mon_psi")


if __name__ == "__main__":
    run()
