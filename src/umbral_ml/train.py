"""PHASES 3 and 4 - MODEL ENGINEERING and MODEL EVALUATION.

Three learning tasks, the same three the game already solves online (scripts/ml_brain.gd):
  A. NEXT ACTION  - multiclass (6 actions): which action will the player do next?
  B. HIT          - binary: will this boss attack hit the player in this situation?
  C. DODGE        - regression (2D): where will the player move while a charge is announced?

Protocol (identical for every candidate):
  * hyper-parameters: GridSearchCV with GroupKFold(5) by player on the TRAIN partition only;
  * model selection: score on the VALIDATION players with the criterion fixed in Phase 1;
  * final: the chosen model is refitted on train+validation and evaluated ONCE on the TEST players,
    which were never used to choose or tune anything.
The in-game online models (replicas in online.py) are the deployed baseline; they learn while the
player plays, so they are evaluated prequentially (predict, then learn) on the same players.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import plots
from .features import H_X, interactions as _interactions
from .config import ACTIONS, ACTION_LABEL, DIFFS, HIT_X, MODELS, MODES, MOVES, PROC, REPORTS, SEED

EPS = 1e-6
GROUP = "pid"            # grouping column of the split (data_report.json): player, or session with < 10 players


def cv_for(groups) -> GroupKFold:
    """5 grouped folds, fewer if there are fewer groups (first weeks of real data)."""
    return GroupKFold(n_splits=max(2, min(5, len(set(groups)))))
EXPERIMENTS: list[dict] = []

# ---------------------------------------------------------------------------- feature sets
A_CAT_SEQ = ["prev1", "prev2"]
A_NUM_SEQ = ["is_start"]
A_CAT_CTX = ["mode"]
A_NUM_CTX = ["w", "diff", "hp", "ne", "boss"]
A_NUM_HIST = ["hist_" + a for a in ACTIONS] + ["log_hist"]
A_NUM_MK = ["mk_" + a for a in ACTIONS]     # output of the player's own online Markov (stacking)
A_FEATURE_SETS = {
    "seq": (A_CAT_SEQ, A_NUM_SEQ),
    "seq+ctx": (A_CAT_SEQ + A_CAT_CTX, A_NUM_SEQ + A_NUM_CTX),
    "seq+ctx+hist": (A_CAT_SEQ + A_CAT_CTX, A_NUM_SEQ + A_NUM_CTX + A_NUM_HIST),
    "seq+ctx+hist+online": (A_CAT_SEQ + A_CAT_CTX, A_NUM_SEQ + A_NUM_CTX + A_NUM_HIST + A_NUM_MK),
}
H_CAT = ["mv", "family", "mode"]
H_NUM = H_X + ["w", "diff"]


def _pre(cat, num):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
    ])


def log_exp(task, model, fset, params, cv_mean, cv_std, val, train_score, secs, metric, note=""):
    EXPERIMENTS.append({"id": f"{task}{len([e for e in EXPERIMENTS if e['task'] == task]) + 1:02d}",
                        "task": task, "model": model, "features": fset, "params": json.dumps(params),
                        "metric": metric, "cv_mean": None if cv_mean is None else round(cv_mean, 4),
                        "cv_std": None if cv_std is None else round(cv_std, 4),
                        "val": round(val, 4), "train": None if train_score is None else round(train_score, 4),
                        "gap_train_val": None if train_score is None else round(train_score - val, 4),
                        "fit_s": round(secs, 2), "note": note})


def _grid(est, grid, X, y, groups, scoring):
    t = time.time()
    gs = GridSearchCV(est, grid, cv=cv_for(groups), scoring=scoring, n_jobs=-1, return_train_score=True, refit=True)
    gs.fit(X, y, groups=groups)
    i = gs.best_index_
    return gs, gs.cv_results_["mean_test_score"][i], gs.cv_results_["std_test_score"][i], \
        gs.cv_results_["mean_train_score"][i], time.time() - t


# ---------------------------------------------------------------------------- task A helpers
def markov_prequential(df: pd.DataFrame, decay: float, order: int, fallback: np.ndarray) -> np.ndarray:
    """In-game predictor, re-run with another decay/order (domain-guided tuning of the deployed model).
    Rows must be sorted by player and time; returns P(next action) before learning each one."""
    P = np.zeros((len(df), 6))
    idx = {a: i for i, a in enumerate(ACTIONS)}
    for _pid, g in df.groupby("pid", sort=False):
        counts: dict[str, np.ndarray] = {}
        for j, r in zip(g.index, g[["prev1", "prev2", "target"]].itertuples(index=False)):
            ctxs = [r.prev2 + ">" + r.prev1, r.prev1] if order == 2 else [r.prev1]
            p = None
            for c in ctxs:
                v = counts.get(c)
                if v is not None and v.sum() >= 3.0:
                    p = v / v.sum()
                    break
            P[df.index.get_loc(j)] = fallback if p is None else 0.97 * p + 0.03 * fallback
            for c in (r.prev2 + ">" + r.prev1, r.prev1):
                v = counts.setdefault(c, np.zeros(6))
                v *= decay
                v[idx[r.target]] += 1.0
    return P


SORTED = np.argsort(ACTIONS)     # sklearn orders labels alphabetically; P columns follow ACTIONS


def a_scores(y, P):
    """P columns are in ACTIONS order."""
    pred = np.array(ACTIONS)[P.argmax(1)]
    Ps = np.clip(P[:, SORTED], EPS, 1)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "log_loss": log_loss(y, Ps / Ps.sum(1, keepdims=True), labels=sorted(ACTIONS))}


def proba_a(est, X) -> np.ndarray:
    """predict_proba of a fitted sklearn model, columns reordered to ACTIONS."""
    P = est.predict_proba(X)
    cls = list(est.classes_)
    return np.column_stack([P[:, cls.index(a)] if a in cls else np.zeros(len(P)) for a in ACTIONS])


def task_a(acts: pd.DataFrame) -> dict:
    tr, va, te = (acts[acts.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    y_tr, y_va = tr.target.values, va.target.values
    prior = tr.target.value_counts(normalize=True).reindex(ACTIONS).fillna(0).values
    res = {"candidates": {}}

    def add(name, P_va, P_tr=None, cv=(None, None), secs=0.0, params=None, fset="-", note=""):
        s = a_scores(y_va, P_va)
        s_tr = a_scores(y_tr, P_tr) if P_tr is not None else None
        res["candidates"][name] = {"val": s, "train": s_tr, "cv_acc_mean": cv[0], "cv_acc_std": cv[1],
                                   "params": params or {}, "features": fset}
        log_exp("A", name, fset, params or {}, cv[0], cv[1], s["accuracy"],
                s_tr["accuracy"] if s_tr else None, secs, "accuracy", note)

    # A0 majority
    add("Majority class", np.tile(prior, (len(va), 1)), np.tile(prior, (len(tr), 1)), note="lower bound")
    # A1 global order-1 Markov (frequency table fitted on train)
    tab = pd.crosstab(tr.prev1, tr.target).reindex(columns=ACTIONS, fill_value=0) + 1
    tab = tab.div(tab.sum(1), axis=0)
    P1 = lambda d: np.vstack([tab.loc[p].values if p in tab.index else prior for p in d.prev1])  # noqa: E731
    add("Global Markov-1 (table)", P1(va), P1(tr), fset="prev1", note="same for every player")
    # A2 in-game online Markov: tune decay and order on TRAIN players (prequential), then validation
    best, grid_log = None, []
    for order in (1, 2):
        for decay in (0.85, 0.9, 0.95, 0.97, 0.99, 1.0):
            acc = a_scores(y_tr, markov_prequential(tr, decay, order, prior))["accuracy"]
            grid_log.append({"order": order, "decay": decay, "train_acc": round(acc, 4)})
            if best is None or acc > best[0]:
                best = (acc, order, decay)
    res["markov_grid"] = grid_log
    t = time.time()
    add("In-game Markov-2 (decay 0.97)", markov_prequential(va, 0.97, 2, prior),
        markov_prequential(tr, 0.97, 2, prior), secs=time.time() - t, params={"order": 2, "decay": 0.97},
        fset="prev1, prev2 (own history)", note="deployed baseline, prequential")
    _, bo, bd = best
    add(f"Tuned online Markov-{bo} (decay {bd})", markov_prequential(va, bd, bo, prior),
        markov_prequential(tr, bd, bo, prior), params={"order": bo, "decay": bd},
        fset="prev1, prev2 (own history)", note="decay/order tuned prequentially on train players")
    res["markov_best"] = {"order": bo, "decay": bd}
    # logged in-game prediction (what the game really predicted), when present
    if "game_pred" in va and (va.game_pred.fillna("") != "").any():
        m = va.game_pred.fillna("") != ""
        res["logged_game_acc_val"] = float((va.game_pred[m] == va.target[m]).mean())

    # A3 multinomial logistic regression on three feature sets (experiment log of the feature design)
    fitted = {}
    for fset, (cat, num) in A_FEATURE_SETS.items():
        cols = cat + num
        pipe = Pipeline([("pre", _pre(cat, num)), ("clf", LogisticRegression(max_iter=2000))])
        gs, m_, s_, trs, secs = _grid(pipe, {"clf__C": [0.1, 1.0, 10.0]}, tr[cols], y_tr, tr[GROUP], "accuracy")
        add(f"Logistic regression [{fset}]", proba_a(gs, va[cols]), proba_a(gs, tr[cols]),
            (m_, s_), secs, gs.best_params_, fset)
        fitted[f"Logistic regression [{fset}]"] = (gs.best_estimator_, cols)
    cat, num = A_FEATURE_SETS["seq+ctx+hist+online"]
    cols = cat + num
    for name, est, grid in [
        ("Random forest", RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=-1),
         {"clf__max_depth": [8, 16], "clf__min_samples_leaf": [5, 20]}),
        ("Gradient boosting (HistGB)", HistGradientBoostingClassifier(random_state=SEED),
         {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 6], "clf__max_iter": [150]}),
    ]:
        pipe = Pipeline([("pre", _pre(cat, num)), ("clf", est)])
        gs, m_, s_, trs, secs = _grid(pipe, grid, tr[cols], y_tr, tr[GROUP], "accuracy")
        add(name, proba_a(gs, va[cols]), proba_a(gs, tr[cols]), (m_, s_), secs, gs.best_params_,
            "seq+ctx+hist+online")
        fitted[name] = (gs.best_estimator_, cols)

    # SELECTION (criterion fixed in Phase 1): highest VALIDATION accuracy among ALL candidates (global
    # models AND online ones), provided its log loss is not worse than the majority class.
    maj_ll = res["candidates"]["Majority class"]["val"]["log_loss"]
    ok = {k: v for k, v in res["candidates"].items() if v["val"]["log_loss"] <= maj_ll and k != "Majority class"}
    chosen = max(ok, key=lambda k: ok[k]["val"]["accuracy"])
    res["chosen"] = chosen
    best_global = max(fitted, key=lambda k: res["candidates"][k]["val"]["accuracy"])
    res["best_global"] = best_global
    est, cols = fitted[best_global]
    trva = pd.concat([tr, va])
    prior_tv = trva.target.value_counts(normalize=True).reindex(ACTIONS).fillna(0).values
    tuned = f"Tuned online Markov-{bo} (decay {bd})"
    # FINAL: refit on train+val (global models) and evaluate ONCE on the test players
    final = clone(est).fit(trva[cols], trva.target)
    P_test = {best_global: proba_a(final, te[cols]),
              "In-game Markov-2 (decay 0.97)": markov_prequential(te, 0.97, 2, prior_tv),
              tuned: markov_prequential(te, bd, bo, prior_tv),
              "Majority class": np.tile(prior_tv, (len(te), 1))}
    if chosen not in P_test:
        cest, ccols = fitted[chosen]
        P_test[chosen] = proba_a(clone(cest).fit(trva[ccols], trva.target), te[ccols])
    res["test"] = {k: a_scores(te.target, P) for k, P in P_test.items()}
    Pt = P_test[chosen]
    pred_t = np.array(ACTIONS)[Pt.argmax(1)]
    res["test_confusion"] = confusion_matrix(te.target, pred_t, labels=ACTIONS).tolist()
    res["test_recall"] = {a: float((pred_t[te.target == a] == a).mean()) for a in ACTIONS}
    # segments (bias check): accuracy per difficulty and per mode on TEST
    seg = {}
    for col, names in (("diff", DIFFS), ("mode", MODES)):
        for v, g in te.assign(ok=(pred_t == te.target)).groupby(col):
            if len(g) >= 50:
                seg[f"{col}={names.get(v, v)}"] = {"n": int(len(g)), "accuracy": float(g.ok.mean())}
    res["test_segments"] = seg
    # how fast it learns a NEW player: accuracy by how many actions of that player came before
    mk = P_test["In-game Markov-2 (decay 0.97)"].argmax(1)
    pos = te.groupby("pid").cumcount()
    bins = [0, 10, 25, 50, 100, 200, 400, 10**6]
    lab = ["1-10", "11-25", "26-50", "51-100", "101-200", "201-400", "400+"]
    b = pd.cut(pos + 1, bins, labels=lab)
    res["by_experience"] = {"bins": lab,
                            "chosen": [float((pred_t[b == l] == te.target[b == l]).mean()) for l in lab],
                            "in_game": [float((np.array(ACTIONS)[mk][b == l] == te.target[b == l]).mean()) for l in lab],
                            "n": [int((b == l).sum()) for l in lab]}
    # permutation importance on validation (with the model fitted on train)
    pi = permutation_importance(est, va[cols], y_va, n_repeats=3, random_state=SEED, scoring="accuracy",
                                n_jobs=-1)
    res["importance"] = dict(sorted(zip(cols, map(float, pi.importances_mean)), key=lambda t: -t[1]))
    # learning curve (over/under-fitting): grouped CV on train+val
    sizes, trs, vas = learning_curve(clone(est), trva[cols], trva.target, groups=trva[GROUP], cv=cv_for(trva[GROUP]),
                                     train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], scoring="accuracy", n_jobs=-1)
    res["learning_curve"] = {"sizes": sizes.tolist(), "train": trs.mean(1).tolist(), "val": vas.mean(1).tolist()}
    if chosen in fitted:
        cest, ccols = fitted[chosen]
        cfinal = final if chosen == best_global else clone(cest).fit(trva[ccols], trva.target)
        joblib.dump({"kind": "sklearn", "name": chosen, "model": cfinal, "features": ccols,
                     "classes": list(cfinal.classes_)}, MODELS / "next_action.joblib")
        res["features"] = ccols
    else:
        o, d = (2, 0.97) if chosen.startswith("In-game") else (bo, bd)
        joblib.dump({"kind": "online_markov", "name": chosen, "order": o, "decay": d,
                     "prior": dict(zip(ACTIONS, map(float, prior_tv)))}, MODELS / "next_action.joblib")
        res["features"] = ["prev1", "prev2", "is_start", "the player's own action history"]
    # the best GLOBAL model is also saved (diagnostics and the demo comparison)
    joblib.dump({"kind": "sklearn", "name": best_global, "model": final, "features": cols,
                 "classes": list(final.classes_)}, MODELS / "next_action_global.joblib")
    return res


# ---------------------------------------------------------------------------- task B
def b_scores(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return {"brier": brier_score_loss(y, p), "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
            "log_loss": log_loss(y, p, labels=[0, 1])}


def online_logit(df, lr):
    from .online import OnlineLogit
    out = np.zeros(len(df))
    for _pid, g in df.groupby("pid", sort=False):
        m = OnlineLogit(lr=lr)
        for j, r in zip(g.index, g.itertuples(index=False)):
            x = [1.0] + [float(getattr(r, c)) for c in H_X]
            out[df.index.get_loc(j)] = m.proba(r.mv, x)
            m.learn(r.mv, x, float(r.y))
    return out


def task_b(hits: pd.DataFrame) -> dict:
    tr, va, te = (hits[hits.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    res = {"candidates": {}}

    def add(name, p_va, p_tr=None, cv=(None, None), secs=0.0, params=None, fset="-", note=""):
        s = b_scores(va.y, p_va)
        s_tr = b_scores(tr.y, p_tr) if p_tr is not None else None
        res["candidates"][name] = {"val": s, "train": s_tr, "cv_brier_mean": cv[0], "cv_brier_std": cv[1],
                                   "params": params or {}, "features": fset}
        log_exp("B", name, fset, params or {}, cv[0], cv[1], s["brier"], s_tr["brier"] if s_tr else None,
                secs, "brier (lower=better)", note)

    rate = tr.y.mean()
    add("Constant hit rate", np.full(len(va), rate), np.full(len(tr), rate), note="lower bound")
    add("In-game bandit (SGD, lr 0.3)", online_logit(va, 0.3), online_logit(tr, 0.3), params={"lr": 0.3},
        fset="8 situation features, own history", note="deployed baseline, prequential")
    best = min((b_scores(tr.y, online_logit(tr, lr))["brier"], lr) for lr in (0.03, 0.1, 0.2, 0.3, 0.6))
    res["bandit_best_lr"] = best[1]
    add(f"Tuned online bandit (lr {best[1]})", online_logit(va, best[1]), online_logit(tr, best[1]),
        params={"lr": best[1]}, fset="8 situation features, own history", note="lr tuned on train players")
    fitted = {}
    specs = [
        ("Logistic regression [situation]", H_X, [],
         Pipeline([("pre", _pre([], H_X)), ("clf", LogisticRegression(max_iter=2000))]), {"clf__C": [0.1, 1, 10]}),
        ("Logistic regression [situation x attack]", H_CAT + H_NUM, [],
         Pipeline([("int", FunctionTransformer(_interactions)),
                   ("pre", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), H_CAT)],
                                             remainder=StandardScaler())),
                   ("clf", LogisticRegression(max_iter=3000))]), {"clf__C": [0.1, 1, 10]}),
        ("Decision tree", H_CAT + H_NUM, [], Pipeline([("pre", _pre(H_CAT, H_NUM)),
                                                       ("clf", DecisionTreeClassifier(random_state=SEED))]),
         {"clf__max_depth": [4, 8], "clf__min_samples_leaf": [20, 50]}),
        ("Random forest", H_CAT + H_NUM, [], Pipeline([("pre", _pre(H_CAT, H_NUM)), (
            "clf", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1))]),
         {"clf__max_depth": [6, 12], "clf__min_samples_leaf": [10, 30]}),
        ("Gradient boosting (HistGB)", H_CAT + H_NUM, [], Pipeline([("pre", _pre(H_CAT, H_NUM)), (
            "clf", HistGradientBoostingClassifier(random_state=SEED))]),
         {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5], "clf__max_iter": [150]}),
    ]
    for name, cols, _, pipe, grid in specs:
        gs, m_, s_, trs, secs = _grid(pipe, grid, tr[cols], tr.y, tr[GROUP], "neg_brier_score")
        add(name, gs.predict_proba(va[cols])[:, 1], gs.predict_proba(tr[cols])[:, 1], (-m_, s_), secs,
            gs.best_params_, "+".join(["situation"] + (["attack"] if "mv" in cols else [])))
        fitted[name] = (gs.best_estimator_, cols)
    chosen = min(fitted, key=lambda k: res["candidates"][k]["val"]["brier"])
    res["chosen"] = chosen
    est, cols = fitted[chosen]
    trva = pd.concat([tr, va])
    final = clone(est).fit(trva[cols], trva.y)
    pt = final.predict_proba(te[cols])[:, 1]
    p_game = online_logit(te, 0.3)
    res["test"] = {chosen: b_scores(te.y, pt), "In-game bandit (SGD, lr 0.3)": b_scores(te.y, p_game),
                   "Constant hit rate": b_scores(te.y, np.full(len(te), trva.y.mean()))}
    fx, fy = calibration_curve(te.y, pt, n_bins=8, strategy="quantile")
    gx, gy = calibration_curve(te.y, p_game, n_bins=8, strategy="quantile")
    res["calibration"] = {"chosen": [fx.tolist(), fy.tolist()], "in_game": [gx.tolist(), gy.tolist()]}
    seg = {}
    for v, g in te.assign(p=pt).groupby("diff"):
        if len(g) >= 50 and g.y.nunique() > 1:
            seg[f"diff={DIFFS.get(v, v)}"] = {"n": int(len(g)), "brier": float(brier_score_loss(g.y, g.p)),
                                             "hit_rate": float(g.y.mean()), "mean_pred": float(g.p.mean())}
    res["test_segments"] = seg
    joblib.dump({"model": final, "features": cols, "moves": sorted(trva.mv.unique())}, MODELS / "hit.joblib")
    res["features"] = cols
    return res


# ---------------------------------------------------------------------------- task C
def task_c(dod: pd.DataFrame) -> dict:
    tr, va, te = (dod[dod.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    err = lambda d, px, py: float(np.hypot(d.ox - px, d.oy - py).mean())  # noqa: E731
    mx, my = tr.ox.mean(), tr.oy.mean()

    def shrunk(d, k, gx, gy):
        out = []
        for _pid, g in d.groupby("pid", sort=False):
            sx = sy = 0.0
            for i, r in enumerate(g.itertuples(index=False)):
                out.append(((sx + k * gx) / (i + k), (sy + k * gy) / (i + k)))
                sx += r.ox
                sy += r.oy
        return np.array(out)

    res = {"candidates": {}}

    def add(name, e_va, e_tr=None, params=None, note=""):
        res["candidates"][name] = {"val": {"mean_error_px": e_va}, "train": None if e_tr is None else
                                   {"mean_error_px": e_tr}, "params": params or {}}
        log_exp("C", name, "own dodge history", params or {}, None, None, e_va, e_tr, 0, "mean error px (lower=better)", note)

    add("No movement (0, 0)", err(va, 0, 0), err(tr, 0, 0), note="lower bound")
    add("Global mean (train)", err(va, mx, my), err(tr, mx, my))
    add("In-game running mean", err(va, va.game_ox, va.game_oy), err(tr, tr.game_ox, tr.game_oy),
        note="deployed baseline, prequential")
    best = min((err(tr, *shrunk(tr, k, mx, my).T), k) for k in (0.5, 1, 2, 4, 8, 16))
    S = shrunk(va, best[1], mx, my)
    add(f"Per-player mean shrunk to global (k={best[1]})", err(va, S[:, 0], S[:, 1]), best[0], {"k": best[1]},
        "k tuned on train players")
    chosen = min(res["candidates"], key=lambda k: res["candidates"][k]["val"]["mean_error_px"])
    res["chosen"] = chosen
    trva = pd.concat([tr, va])
    gmx, gmy = trva.ox.mean(), trva.oy.mean()
    St = shrunk(te, best[1], gmx, gmy)
    res["test"] = {"Per-player mean shrunk to global (k=%s)" % best[1]: {"mean_error_px": err(te, St[:, 0], St[:, 1])},
                   "In-game running mean": {"mean_error_px": err(te, te.game_ox, te.game_oy)},
                   "Global mean (train)": {"mean_error_px": err(te, gmx, gmy)},
                   "No movement (0, 0)": {"mean_error_px": err(te, 0, 0)}}
    joblib.dump({"global": [gmx, gmy], "k": best[1]}, MODELS / "dodge.joblib")
    return res


# ---------------------------------------------------------------------------- figures
def figures(rep: dict, A: dict, B: dict, C: dict, acts: pd.DataFrame, hits: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt
    figs = {}
    vc = acts.target.value_counts().reindex(ACTIONS)
    figs["classes"] = plots.hbar([ACTION_LABEL[a] for a in ACTIONS], (vc / vc.sum()).tolist(),
                                 "Next-action target: share of each class", "share of actions",
                                 fmt="{:.1%}", name="fig_classes")
    hm = hits.groupby("mv").y.agg(["mean", "size"]).sort_values("mean", ascending=False)
    figs["hit_by_move"] = plots.hbar([f"{m} (n={n})" for m, n in zip(hm.index, hm["size"])], hm["mean"].tolist(),
                                     "Hit target: how often each boss attack hits", "hit rate", fmt="{:.0%}",
                                     name="fig_hit_by_move")
    # before / after
    b, a = rep["before"], rep["after"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    items = [("Events", b["events"], a["events"]), ("Duplicated events", b["duplicate_events"], a["duplicate_events"]),
             ("Distance = 9999 (no enemy)", b["ne_sentinel_9999"], a["ne_sentinel_9999"])]
    y = np.arange(len(items))
    ax.barh(y - 0.18, [i[1] for i in items], 0.34, color=plots.OTHER, label="before cleaning")
    ax.barh(y + 0.18, [i[2] for i in items], 0.34, color=plots.CHOSEN, label="after cleaning")
    for k, (_, bv, av) in enumerate(items):
        ax.text(bv, k - 0.18, f"  {bv:,}", va="center", fontsize=8)
        ax.text(av, k + 0.18, f"  {av:,}", va="center", fontsize=8)
    ax.set_yticks(y, [i[0] for i in items])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.set_title("Data quality before and after cleaning (log scale)")
    figs["before_after"] = plots.save(fig, "fig_before_after")
    # model comparisons
    ca = A["candidates"]
    names = list(ca)
    figs["cmp_a"] = plots.hbar(names, [ca[n]["val"]["accuracy"] for n in names],
                               "Next action: validation accuracy", "accuracy", chosen=A["chosen"],
                               baseline=["Majority class", "In-game Markov-2 (decay 0.97)"], name="fig_cmp_a")
    cb = B["candidates"]
    names = list(cb)
    figs["cmp_b"] = plots.hbar(names, [cb[n]["val"]["brier"] for n in names],
                               "Hit probability: validation Brier score", "Brier score", chosen=B["chosen"],
                               baseline=["Constant hit rate", "In-game bandit (SGD, lr 0.3)"], name="fig_cmp_b",
                               lower_better=True)
    cc = C["candidates"]
    names = list(cc)
    figs["cmp_c"] = plots.hbar(names, [cc[n]["val"]["mean_error_px"] for n in names],
                               "Dodge: validation mean error", "pixels", chosen=C["chosen"],
                               baseline=["No movement (0, 0)", "In-game running mean"], fmt="{:.1f}",
                               name="fig_cmp_c", lower_better=True)
    # confusion matrix (sequential blue)
    cm = np.array(A["test_confusion"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("b", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if cmn[i, j] > 0.55 else plots.INK)
    ax.set_xticks(range(6), ACTIONS, rotation=30)
    ax.set_yticks(range(6), ACTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("real action")
    ax.set_title(f"Confusion matrix on TEST - {A['chosen']}", fontsize=9)
    figs["confusion"] = plots.save(fig, "fig_confusion")
    # learning curve
    lc = A["learning_curve"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(lc["sizes"], lc["train"], color=plots.SERIES[1], lw=2, marker="o", ms=4, label="training score")
    ax.plot(lc["sizes"], lc["val"], color=plots.CHOSEN, lw=2, marker="o", ms=4, label="cross-validation score")
    ax.set_xlabel("training rows")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Learning curve - {A['best_global']} (best global model)", fontsize=9.5)
    ax.legend()
    figs["learning_curve"] = plots.save(fig, "fig_learning_curve")
    # experience curve
    ex = A["by_experience"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(ex["bins"], ex["chosen"], color=plots.CHOSEN, lw=2, marker="o", ms=4, label=A["chosen"])
    ax.plot(ex["bins"], ex["in_game"], color=plots.SERIES[1], lw=2, marker="o", ms=4, label="in-game Markov-2")
    ax.set_xlabel("actions of that player already seen")
    ax.set_ylabel("accuracy on TEST")
    ax.set_title("How fast each model learns a NEW player")
    ax.legend()
    figs["experience"] = plots.save(fig, "fig_experience")
    # calibration
    cal = B["calibration"]
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], color=plots.GRID, lw=1.5, ls="--", label="perfect")
    ax.plot(*cal["chosen"], color=plots.CHOSEN, lw=2, marker="o", ms=4, label=B["chosen"])
    ax.plot(*cal["in_game"], color=plots.SERIES[1], lw=2, marker="o", ms=4, label="in-game bandit")
    ax.set_xlabel("predicted hit probability")
    ax.set_ylabel("observed hit rate")
    ax.set_title("Calibration on TEST (hit task)", fontsize=9.5)
    ax.legend(fontsize=7.5)
    figs["calibration"] = plots.save(fig, "fig_calibration")
    imp = list(A["importance"].items())[:10]
    figs["importance"] = plots.hbar([k for k, _ in imp], [max(v, 0) for _, v in imp],
                                    f"Permutation importance - {A['best_global']}", "accuracy drop",
                                    fmt="{:.3f}", name="fig_importance")
    return figs


def benchmark(n: int = 200) -> dict:
    """Latency of one prediction as the Flask demo serves it (S5)."""
    from . import predict as P
    P.load.cache_clear()
    P.card.cache_clear() if (MODELS / "model_card.json").exists() else None
    hist = ["shoot", "shoot", "dodge", "shoot", "block", "shoot"]
    P.predict_next(hist, {"w": 2})
    t = time.perf_counter()
    for _ in range(n):
        P.predict_next(hist, {"w": 2, "hp": 4, "ne": 120, "boss": True})
    nx = 1000 * (time.perf_counter() - t) / n
    b = joblib.load(MODELS / "hit.joblib")
    from .predict import hit_row
    X = pd.DataFrame([hit_row("dive", {"close": 1}, {})])[b["features"]]
    b["model"].predict_proba(X)
    t = time.perf_counter()
    for _ in range(n):
        b["model"].predict_proba(X)
    ht = 1000 * (time.perf_counter() - t) / n
    sizes = {p.name: round(p.stat().st_size / 1024, 1) for p in MODELS.glob("*.joblib")}
    return {"next_ms": nx, "hit_ms": ht, "model_kb": sizes}


def main() -> dict:
    acts = pd.read_csv(PROC / "actions.csv", keep_default_na=False, na_values=[""])
    acts["prev1"] = acts.prev1.astype(str)
    acts["prev2"] = acts.prev2.astype(str)
    acts["game_pred"] = acts.game_pred.fillna("")
    hits = pd.read_csv(PROC / "hits.csv")
    dod = pd.read_csv(PROC / "dodges.csv")
    rep = json.loads((PROC / "data_report.json").read_text(encoding="utf-8"))
    EXPERIMENTS.clear()
    global GROUP
    GROUP = rep["split"]["group_by"]
    t = time.time()
    A = task_a(acts)
    print(f"A done {time.time() - t:.0f}s  chosen={A['chosen']}  test={A['test'][A['chosen']]}")
    B = task_b(hits)
    print(f"B done {time.time() - t:.0f}s  chosen={B['chosen']}  test={B['test'][B['chosen']]}")
    C = task_c(dod)
    print(f"C done  chosen={C['chosen']}")
    figs = figures(rep, A, B, C, acts, hits)
    pd.DataFrame(EXPERIMENTS).to_csv(REPORTS / "experiments.csv", index=False)
    import sklearn
    serving = benchmark()
    results = {"serving": serving, "source": rep["source"], "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "sklearn": sklearn.__version__, "data": rep, "A": A, "B": B, "C": C, "figures": figs,
               "experiments": EXPERIMENTS}
    (REPORTS / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    card = {"source": rep["source"], "trained_at": results["trained_at"], "sklearn": sklearn.__version__,
            "next_action": {"model": A["chosen"], "features": A["features"], "test": A["test"][A["chosen"]]},
            "hit": {"model": B["chosen"], "features": B["features"], "test": B["test"][B["chosen"]]},
            "dodge": {"model": C["chosen"], "test": C["test"]}}
    (MODELS / "model_card.json").write_text(json.dumps(card, indent=1, default=float), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
