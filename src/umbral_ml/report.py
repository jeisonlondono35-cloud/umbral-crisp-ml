"""Technical report (PDF), generated from the SAME files the Flask app shows
(results.json, data_report.json, experiments, monitoring, expert review): numbers cannot disagree."""
from __future__ import annotations

import json
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .config import FIG, MON, PROC, REPORTS, RAW, ROOT

APP_URL = "https://umbral-crisp-ml.onrender.com"
REPO_URL = "https://github.com/jeisonlondono35-cloud/umbral-crisp-ml"
GAME_URL = "https://juego-pixelart.onrender.com"
IG, IB = "In-game Markov-2 (decay 0.97)", "In-game bandit (SGD, lr 0.3)"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0d366b"))
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=9, spaceAfter=4)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=9.5, leading=13, spaceAfter=4)
SM = ParagraphStyle("sm", parent=P, fontSize=8, leading=10.5, textColor=colors.HexColor("#52514e"))
CELL = ParagraphStyle("cell", parent=P, fontSize=7.8, leading=9.6, spaceAfter=0)
WARN = ParagraphStyle("warn", parent=P, backColor=colors.HexColor("#fff3dc"), borderColor=colors.HexColor("#a35f00"),
                      borderWidth=0.6, borderPadding=5, spaceBefore=6, spaceAfter=8)


def pct(v, d=1):
    return "-" if v is None else f"{100 * float(v):.{d}f}%"


def f3(v, d=3):
    return "-" if v is None else f"{float(v):.{d}f}"


def table(rows, widths=None, chosen_rows=()):
    data = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efeee9")),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d6d5cf")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    for i in chosen_rows:
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#e3eefb")))
    t.setStyle(TableStyle(st))
    return t


def fig(name, width=15.5, caption=None):
    from PIL import Image as PI
    p = FIG / name
    w, h = PI.open(p).size
    out = [Image(str(p), width=width * cm, height=width * cm * h / w)]
    if caption:
        out.append(Paragraph(caption, SM))
    return KeepTogether(out)


def main() -> None:
    R = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    D = R["data"]
    M = json.loads((MON / "monitoring_simulated.json").read_text(encoding="utf-8"))
    E = json.loads((ROOT / "docs" / "expert_review.json").read_text(encoding="utf-8"))
    PUB = json.loads((RAW / "ml_public.json").read_text(encoding="utf-8"))
    A, B, C = R["A"], R["B"], R["C"]
    b = D["before"]
    sim = R["source"] == "SIMULATED"
    s = []

    # ------------------------------------------------------------------ cover
    s += [Spacer(1, 3 * cm),
          Paragraph("UMBRAL · CRISP-ML", ParagraphStyle("t", parent=ss["Title"], fontSize=26, textColor=colors.HexColor("#0d366b"))),
          Paragraph("Teaching a game's bosses to read the player", ParagraphStyle("st", parent=ss["Title"], fontSize=15)),
          Spacer(1, 0.6 * cm),
          Paragraph("Technical report of the Machine Learning project, developed with the six phases of CRISP-ML", ParagraphStyle("c", parent=P, alignment=TA_CENTER)),
          Spacer(1, 0.4 * cm),
          Paragraph("Team: <b>J.A · J.P · J.Q</b>", ParagraphStyle("c2", parent=P, alignment=TA_CENTER)),
          Paragraph(f"{date.today().isoformat()} · data source of this build: <b>{R['source']}</b> · trained {R['trained_at']}", ParagraphStyle("c3", parent=SM, alignment=TA_CENTER)),
          Spacer(1, 1.2 * cm),
          table([["Deliverable", "Where"], ["Flask application (six phases + demo)", APP_URL],
                 ["Repository (code, scripts, dependencies, models, README)", REPO_URL], ["The game", GAME_URL]],
                [7 * cm, 9 * cm])]
    if sim:
        s.append(Paragraph(f"<b>SIMULATED DATA.</b> The anonymous gameplay log of the real game started recording in November 2026 and so far "
                           f"has {D['real_available']['actions']:,} real actions from {D['real_available']['players']} players, below the minimum fixed in Phase 1 "
                           f"({D['real_available']['min_actions']:,} actions, {D['real_available']['min_players']} players). All model results in this build come from "
                           f"<b>simulated players</b> generated from the game's rules; real production evidence is labelled REAL. "
                           f"Running <font face='Courier'>python pipeline.py --source real</font> regenerates this report, the app and the models on the real data.", WARN))
    s.append(PageBreak())

    # ------------------------------------------------------------------ context
    s += [Paragraph("1. Context and objectives", H1),
          Paragraph("UMBRAL is a pixel-art action game made in Godot 4 and published on the web, Android and PC. Its bosses adapt to the person "
                    "playing: three small models in <font face='Courier'>scripts/ml_brain.gd</font> learn online while the player fights — an order-2 Markov "
                    "chain with forgetting that predicts the next action, a running mean of the dodge direction, and a contextual bandit (one logistic "
                    "regression per boss attack) that estimates whether an attack will hit. The bosses use them to feint the shield, wait for the dodge, "
                    "aim where the player goes and choose the attack most likely to hit.", P),
          Paragraph("<b>Approved project:</b> study the whole ML of the game with CRISP-ML: measure how well those models really work, compare them with "
                    "other algorithms on recorded play data, choose the best model with a criterion fixed in advance, deploy it in a Flask application and "
                    "plan its monitoring.", P)]

    # ------------------------------------------------------------------ phase 1
    s += [Paragraph("2. Phase 1 — Business and Data Understanding", H1),
          Paragraph("Stakeholders and beneficiaries", H2),
          table([["Who", "Role", "Need"],
                 ["Players", "Main beneficiaries; produce the data", "Fights that adapt and stay fair; transparency"],
                 ["J.A", "Creator, domain expert, owner of model and data", "Know if the in-game ML works; which model to ship"],
                 ["J.P", "Game developer (Godot integration)", "A model that runs on phones in real time"],
                 ["J.Q", "Developer (web, reports)", "Publishable indicators and reports"],
                 ["Instructor", "Evaluator", "A traceable CRISP-ML process"]], [2.6 * cm, 6 * cm, 7.4 * cm]),
          Paragraph("Business objectives translated into ML objectives", H2),
          table([["Business objective", "ML objective", "Task"],
                 ["Bosses anticipate the player", "Predict the next action better than the in-game model and the majority class", "Multiclass (6), sequential"],
                 ["Attacks that challenge but are fair", "Calibrated P(attack hits | situation)", "Binary, probabilistic"],
                 ["Bosses aim where you go", "Predict the dodge offset", "2D regression"],
                 ["Runs on phones during a fight", "Milliseconds per prediction, small files, fast learning of a new player", "Constraint"]],
                [5 * cm, 7.5 * cm, 3.5 * cm]),
          Paragraph("Questions the model answers: (1) which action comes next and how sure we are; (2) the probability that attack <i>m</i> hits now; "
                    "(3) where the player will move while a charge is announced.", P),
          Paragraph("Data sources, variables, size and target", H2),
          table([["Source", "Content", "Status"],
                 ["Anonymous gameplay log (Supabase umbral_ml_log)", "Added to the game for this project (scripts/ml_log.gd): every action in combat with the in-game "
                  "prediction, every boss attack with its 8 situation features and whether it hit, every dodge. Random player and session ids; opt-out switch.",
                  f"REAL, since Nov 2026: {D['real_available']['actions']:,} actions, {D['real_available']['players']} players"],
                 ["In-game accuracy per account (umbral_ml_public)", "Predictions made and correct per account",
                  f"REAL: {PUB['players']} players, {PUB['made']:,} predictions, {pct(PUB['right'] / PUB['made'])} correct"],
                 ["Simulated players (simulate.py)", "Generated from the game's rules, same format", f"SIMULATED: {b['events']:,} events, {b['players']} players, {b['sessions']} sessions"]],
                [4.2 * cm, 8 * cm, 3.8 * cm]),
          Paragraph(f"Targets: the action (6 classes: shoot, melee, block, dodge, jump, hide), the hit outcome y ∈ {{0,1}} (overall rate {pct(b['hit_rate'])}) "
                    f"and the dodge offset (ox, oy) in pixels. Size used in this build ({D['source']}): {D['rows']['actions']:,} actions, {D['rows']['hits']:,} attacks, "
                    f"{D['rows']['dodges']:,} dodges.", P),
          fig("fig_classes.png", 12, "Figure 1. Class balance of the next-action target (unbalanced: macro-F1 and log loss are also reported)."),
          Paragraph("Quality, relevance, sufficiency, feasibility and risks", H2),
          Paragraph("The log is written by the game itself (typed, bounded values) and records exactly what the in-game models see when they decide, so a "
                    "better model can be swapped in without new sensors. Known problems are measured in Phase 2. Sufficiency: at least "
                    f"{D['real_available']['min_actions']:,} real actions from {D['real_available']['min_players']} players to train on real data, and 10 players to split "
                    "by player. Risks: few and non-representative real players (friends of the team), players and game changing over time, privacy, a boss "
                    "that feels unfair or too hard, and the real-time constraint on phones; each has a mitigation (simulation and re-run, drift monitoring, "
                    "anonymous ids and opt-out, fairness caps in the game, latency measured).", P),
          Paragraph("Measurable success criteria (fixed before training)", H2),
          table([["#", "Criterion"], ["S1", "Next action: test accuracy ≥ majority class + 10 points"], ["S2", "Next action: test accuracy ≥ current in-game model"],
                 ["S3", "Hit: test Brier below the in-game bandit and the constant rate"], ["S4", "Dodge: test mean error below the in-game running mean"],
                 ["S5", "Serving latency below 50 ms per prediction"],
                 ["S6", "Selection by the best VALIDATION score (log loss not worse than the majority class); the test set is used once"]], [1.2 * cm, 14.8 * cm])]

    # ------------------------------------------------------------------ phase 2
    a = D["after"]
    s += [PageBreak(), Paragraph("3. Phase 2 — Data Engineering", H1),
          Paragraph("Selected variables: action, time gap, enemy distance, world, mode, difficulty, life, boss scene, attack id, the 8 situation features, the hit "
                    "outcome and the dodge offset. The in-game predictions (p, pp) are kept only to measure the deployed model; ids and times are used for "
                    "ordering and grouping, never as inputs.", P),
          table([["Check", "Before", "After", "Decision"],
                 ["Events", f"{b['events']:,}", f"{a['events']:,}", ""],
                 ["Duplicated events", b["duplicate_events"], a["duplicate_events"], "removed (batch re-uploaded after a lost answer)"],
                 ["Distance = 9999 (no enemy)", b["ne_sentinel_9999"], a["ne_sentinel_9999"], "set to missing; median of TRAIN inside each pipeline"],
                 ["Unknown actions", b["invalid_action"], a["invalid_action"], "removed if present"],
                 ["Gaps ≥ 10 min", b["gap_over_10min"], a["gap_over_10min"], "kept, clipped; > 4 s starts a new phrase"]],
                [4.5 * cm, 2 * cm, 2 * cm, 7.5 * cm]),
          fig("fig_before_after.png", 13, "Figure 2. Evidence before and after cleaning."),
          Paragraph("Transformations, in order", H2),
          table([["#", "Step", "Why", "Rows before → after"]] +
                [[i + 1, t["step"], t["why"], f"{t['rows_before']:,} → {t['rows_after']:,}"] for i, t in enumerate(D["transformations"])] +
                [[len(D["transformations"]) + 1, "Encoding and scaling", "Inside each sklearn Pipeline, fitted on training only: one-hot (unknown ignored), "
                  "median imputation, standard scaling", "-"]], [0.8 * cm, 3.6 * cm, 8.8 * cm, 2.8 * cm]),
          Paragraph("Engineered features from domain knowledge: prev1/prev2 with the game's 4-second phrase rule; the player's past frequency of each action "
                    "(hist_*, Laplace-smoothed) and log of history length; the probabilities of the player's own online Markov chain (mk_*, stacking); "
                    "boss-fight flag; attack family; attack × situation products. Every sequential feature uses only earlier events of the same player.", P),
          Paragraph("Train / validation / test split without leakage", H2),
          table([["Partition", "Groups", "Actions", "Attacks", "Dodges", "Use"],
                 ["train", D["split"]["train"], f"{D['split_rows']['actions']['train']:,}", f"{D['split_rows']['hits']['train']:,}", D['split_rows']['dodges']['train'], "fit + GroupKFold(5) tuning"],
                 ["validation", D["split"]["val"], f"{D['split_rows']['actions']['val']:,}", f"{D['split_rows']['hits']['val']:,}", D['split_rows']['dodges']['val'], "model choice"],
                 ["test", D["split"]["test"], f"{D['split_rows']['actions']['test']:,}", f"{D['split_rows']['hits']['test']:,}", D['split_rows']['dodges']['test'], "one final measurement"]],
                [2.4 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 2 * cm, 5.6 * cm]),
          Paragraph(f"Grouping: by <b>{'player' if D['split']['group_by'] == 'pid' else 'session'}</b>. {D['split']['why']}", P)]

    # ------------------------------------------------------------------ phase 3
    s += [PageBreak(), Paragraph("4. Phase 3 — Model Engineering", H1),
          Paragraph("Candidates per task, chosen because they fit the problem: majority class (lower bound); global Markov-1 table; the in-game Markov-2 with "
                    "decay 0.97 (exact replica) and a tuned version (order and decay chosen by data); multinomial logistic regression (calibrated, tiny, portable "
                    "to GDScript) on four feature sets; random forest (non-linear interactions); histogram gradient boosting (strong on tabular data, missing "
                    "values). Hit task: constant rate, in-game SGD bandit and a tuned one, logistic regression with and without attack × situation interactions, "
                    "decision tree, random forest, gradient boosting. Dodge: no movement, global mean, in-game running mean, per-player mean shrunk to the global mean.", P),
          Paragraph("Configuration: GridSearchCV with GroupKFold(5) by player on the training partition (scoring accuracy for task 1, Brier for task 2); online "
                    "models tuned prequentially on training players; random seed 42; scikit-learn " + R["sklearn"] + ".", P),
          Paragraph("Experiment log (all runs, reports/experiments.csv)", H2)]
    rows = [["id", "model", "features", "best params", "cv", "val", "train"]]
    for e in R["experiments"]:
        rows.append([e["id"], e["model"], e["features"], e["params"].replace('"', "")[:60],
                     "-" if e["cv_mean"] is None else f"{e['cv_mean']:.3f}±{e['cv_std']:.3f}", f3(e["val"]), f3(e["train"]) if e["train"] is not None else "-"])
    s += [table(rows, [1 * cm, 4.2 * cm, 3 * cm, 3.8 * cm, 1.8 * cm, 1.1 * cm, 1.1 * cm]),
          Paragraph("Metric of the val/train columns: accuracy (A), Brier (B, lower is better), mean error in px (C).", SM),
          Paragraph("How domain knowledge guided the decisions: split by player (the model must serve a new player); the 4-second phrase rule and order-2 "
                    "contexts (combat comes in bursts); personal history features (players have styles); one slope per attack (a feint beats a shield, ground "
                    "waves are jumped); Brier for the hit task (the boss compares probabilities between attacks); prequential evaluation of online models "
                    "(in the game they predict first and learn afterwards).", P)]

    # ------------------------------------------------------------------ phase 4
    s += [PageBreak(), Paragraph("5. Phase 4 — Model Evaluation", H1),
          Paragraph("Metrics: accuracy for the next action (the boss acts on the most likely action; it is the number the game shows), with macro-F1 (unbalanced "
                    "classes) and log loss (the boss also weighs the probability); Brier score for the hit task (calibrated probabilities), with ROC-AUC and log "
                    "loss; mean Euclidean error in pixels for the dodge. Validation adapted to the data: sequences grouped by player → GroupKFold by player, separate "
                    "validation players, prequential evaluation for the online models.", P),
          fig("fig_cmp_a.png", 14, "Figure 3. Next action, validation accuracy (blue = chosen, dark gray = baselines)."),
          fig("fig_cmp_b.png", 14, "Figure 4. Hit probability, validation Brier score (lower is better).")]
    rows = [["Model", "CV acc", "val acc", "val macro-F1", "val log loss", "train acc"]]
    ch = []
    for i, (k, v) in enumerate(A["candidates"].items(), start=1):
        rows.append([k, "-" if not v["cv_acc_mean"] else f"{pct(v['cv_acc_mean'])}±{pct(v['cv_acc_std'])}", pct(v["val"]["accuracy"]),
                     f3(v["val"]["macro_f1"]), f3(v["val"]["log_loss"]), pct(v["train"]["accuracy"]) if v["train"] else "-"])
        if k == A["chosen"]:
            ch.append(i)
    s += [Paragraph("Next action — validation players", H2), table(rows, [5.6 * cm, 2.4 * cm, 1.8 * cm, 2.1 * cm, 2.1 * cm, 2 * cm], ch)]
    rows, ch = [["Model", "CV Brier", "val Brier", "val AUC", "val log loss", "train Brier"]], []
    for i, (k, v) in enumerate(B["candidates"].items(), start=1):
        rows.append([k, "-" if not v["cv_brier_mean"] else f"{v['cv_brier_mean']:.3f}±{v['cv_brier_std']:.3f}", f3(v["val"]["brier"]),
                     f3(v["val"]["roc_auc"]), f3(v["val"]["log_loss"]), f3(v["train"]["brier"]) if v["train"] else "-"])
        if k == B["chosen"]:
            ch.append(i)
    s += [Paragraph("Hit — validation players", H2), table(rows, [5.6 * cm, 2.4 * cm, 1.8 * cm, 2.1 * cm, 2.1 * cm, 2 * cm], ch)]
    rows, ch = [["Model", "val error (px)", "train error (px)"]], []
    for i, (k, v) in enumerate(C["candidates"].items(), start=1):
        rows.append([k, f3(v["val"]["mean_error_px"], 1), f3(v["train"]["mean_error_px"], 1) if v["train"] else "-"])
        if k == C["chosen"]:
            ch.append(i)
    s += [Paragraph("Dodge — validation players", H2), table(rows, [8 * cm, 3 * cm, 3 * cm], ch)]
    lc = A["learning_curve"]
    ga = [e for e in R["experiments"] if e["task"] == "A" and e["gap_train_val"] is not None]
    worst = max(ga, key=lambda e: e["gap_train_val"])
    seg = "; ".join(f"{k} {pct(v['accuracy'])} (n={v['n']})" for k, v in A["test_segments"].items())
    segb = "; ".join(f"{k}: real {pct(v['hit_rate'])} vs predicted {pct(v['mean_pred'])}" for k, v in B["test_segments"].items())
    s += [Paragraph("Over-fitting, under-fitting, bias and risks", H2),
          fig("fig_learning_curve.png", 12), fig("fig_experience.png", 12),
          Paragraph(f"Learning curve of the best global model: cross-validation accuracy {pct(lc['val'][0])} with {lc['sizes'][0]:,} rows and {pct(lc['val'][-1])} "
                    f"with {lc['sizes'][-1]:,} rows (training {pct(lc['train'][-1])}). The train − validation gap of every run is in the experiment log; the "
                    f"largest is {worst['model']} ({100 * worst['gap_train_val']:.1f} points). A global model without the player's history under-fits (logistic regression on prev1/prev2: "
                    f"{pct(A['candidates']['Logistic regression [seq]']['val']['accuracy'])} vs majority {pct(A['candidates']['Majority class']['val']['accuracy'])}); "
                    f"personal features fix it. Accuracy per segment on test: {seg}. Hit calibration per difficulty: {segb}. Test recall per action: "
                    + ", ".join(f"{k} {pct(v, 0)}" for k, v in A["test_recall"].items()) + " — rare actions are predicted worse.", P),
          fig("fig_confusion.png", 10), fig("fig_calibration.png", 9), fig("fig_importance.png", 12),
          Paragraph("Review with the domain expert", H2),
          Paragraph(f"{E['expert']} — {E['date']}. {E['method']} ({E['data_shown']})", P),
          table([["Question", "Answer (original)", "Answer (English)"]] + [[q["question"], q["answer_es"], q["answer_en"]] for q in E["answers"]],
                [6 * cm, 3.2 * cm, 6.8 * cm]),
          Paragraph("Consequences: " + " ".join(E["consequences"]), P),
          Paragraph("Selection and final result on the independent test players", H2),
          Paragraph(f"Rule fixed in Phase 1: best validation score, log loss not worse than the majority class. Chosen: <b>{A['chosen']}</b> (next action), "
                    f"<b>{B['chosen']}</b> (hit), <b>{C['chosen']}</b> (dodge); refitted on train + validation and measured once on the "
                    f"{D['split']['test']} test groups.", P)]
    rows = [["Task", "Model", "Test result"]]
    ch = []
    for k, v in A["test"].items():
        rows.append(["Next action", k, f"accuracy {pct(v['accuracy'])} · macro-F1 {f3(v['macro_f1'])} · log loss {f3(v['log_loss'])}"])
        if k == A["chosen"]:
            ch.append(len(rows) - 1)
    for k, v in B["test"].items():
        rows.append(["Hit", k, f"Brier {f3(v['brier'])} · AUC {f3(v['roc_auc'])} · log loss {f3(v['log_loss'])}"])
        if k == B["chosen"]:
            ch.append(len(rows) - 1)
    for i, (k, v) in enumerate(C["test"].items()):
        rows.append(["Dodge", k, f"{f3(v['mean_error_px'], 1)} px"])
        if i == 0:
            ch.append(len(rows) - 1)
    s.append(table(rows, [2.4 * cm, 6 * cm, 7.6 * cm], ch))
    at, bt = A["test"][A["chosen"]], B["test"][B["chosen"]]
    ck = list(C["test"])[0]
    sv = R["serving"]
    crit = [("S1", at["accuracy"] - A["test"]["Majority class"]["accuracy"] >= 0.10, f"{pct(at['accuracy'])} vs {pct(A['test']['Majority class']['accuracy'])}"),
            ("S2", at["accuracy"] >= A["test"][IG]["accuracy"], f"{pct(at['accuracy'])} vs {pct(A['test'][IG]['accuracy'])}"),
            ("S3", bt["brier"] < B["test"][IB]["brier"] and bt["brier"] < B["test"]["Constant hit rate"]["brier"], f"{f3(bt['brier'])} vs {f3(B['test'][IB]['brier'])} / {f3(B['test']['Constant hit rate']['brier'])}"),
            ("S4", C["test"][ck]["mean_error_px"] < C["test"]["In-game running mean"]["mean_error_px"], f"{f3(C['test'][ck]['mean_error_px'], 1)} vs {f3(C['test']['In-game running mean']['mean_error_px'], 1)} px"),
            ("S5", sv["next_ms"] < 50 and sv["hit_ms"] < 50, f"{f3(sv['next_ms'], 2)} ms / {f3(sv['hit_ms'], 2)} ms")]
    s += [Spacer(1, 6), table([["Criterion", "Result", "Met"]] + [[c, r, "yes" if ok else "no"] for c, ok, r in crit], [2 * cm, 9 * cm, 2 * cm])]

    # ------------------------------------------------------------------ phase 5
    s += [PageBreak(), Paragraph("6. Phase 5 — Model Deployment", H1),
          Paragraph("Saved artefacts (models/): next_action.joblib (chosen model as a full sklearn Pipeline: imputer + one-hot + scaler + model), "
                    "next_action_global.joblib, hit.joblib (Pipeline with the attack × situation step), dodge.joblib, model_card.json (source, date, features, "
                    "test metrics, library version). The serving code (src/umbral_ml/predict.py) rebuilds the features with the same functions as training. "
                    f"Sizes: " + ", ".join(f"{k} {v} KB" for k, v in sv["model_kb"].items()) + ".", P),
          Paragraph(f"Flask application ({APP_URL}): cover page, the six phases, a live demo (/demo) and a JSON API: POST /api/predict/next "
                    "(history of actions + world, mode, difficulty, life, enemy distance, boss flag → predicted action, probabilities, what the in-game model would "
                    "say, latency), POST /api/predict/hit (attack + situation → probability of hit), POST /api/predict/dodge, GET /api/health, GET /monitoring/live. "
                    "Dependencies: Flask, scikit-learn, pandas, numpy, matplotlib, joblib, reportlab, requests, gunicorn (requirements.txt). Run: "
                    "<font face='Courier'>pip install -r requirements.txt; python pipeline.py; python -m flask --app app.app run</font>; production "
                    "<font face='Courier'>gunicorn app.app:app</font> on Render (render.yaml).", P),
          Paragraph("First-use instructions: open the cover, read the phases in order, open Demo, type recent actions separated by commas, set the situation and "
                    "press Predict; read the predicted action, its probability and the in-game comparison; choose an attack and read the hit probability; use the "
                    "form on Phase 5 to send an opinion or report an incident.", P),
          Paragraph("Integration plan with the information system of Activity 2 (game + website + Supabase)", H2),
          table([["Step", "What", "Owner", "Status"],
                 ["1 Data in", "The game writes anonymous batches to Supabase (umbral_ml_log_put), next to accounts, saves, records and ML accuracy", "J.P / J.A", "done"],
                 ["2 Training", "pipeline.py reads umbral_ml_log_export, retrains and regenerates report and app; weekly GitHub Action", "J.A", "done"],
                 ["3 Serving", "Flask app on Render exposes /api/predict/*; the game's website links to it", "J.Q", "done"],
                 ["4 In the game", "In-game model stays as offline fallback; when online the boss asks the API at fight start and every 2 s and uses answers "
                  "fresher than 2 s; a linear/Markov winner is exported as JSON shipped with the game", "J.P", "planned"],
                 ["5 Feedback", "umbral_ml_public accuracy, in-game report form and this app's form, reviewed weekly", "J.Q / J.A", "running"]],
                [2.2 * cm, 9.6 * cm, 2.2 * cm, 2 * cm]),
          Paragraph("Monitoring, incidents and opinions: every prediction is logged (time, model, inputs, output, latency) to monitoring/live_predictions.jsonl "
                    "and summarised at /monitoring/live; incidents via the app form, the game's Report error form or a GitHub issue, with severities (high: API "
                    "down or wrong for everyone → fix or roll back the same day; medium: a segment worse than expected → weekly review; low → next release); "
                    "rollback by redeploying the previous commit (models are versioned with their model card); opinions with a 1–5 rating, reviewed weekly.", P)]

    # ------------------------------------------------------------------ phase 6
    th = M["thresholds"]
    s += [Paragraph("7. Phase 6 — Monitoring and Maintenance", H1),
          table([["Indicator", "Source", "Threshold"],
                 ["API up, requests, latency", "/api/health, /monitoring/live", "< 50 ms; any 5xx = incident"],
                 ["Real events and players per day", "umbral_ml_log_stats (REAL)", "0 events while people play = logging broken"],
                 ["Next-action accuracy (2-week mean)", "logged predictions vs real actions", f"baseline (weeks 1–{th['baseline_weeks']}) − {int(th['acc_drop'] * 100)} points"],
                 ["In-game accuracy, all accounts", f"umbral_ml_public (REAL: {pct(PUB['right'] / PUB['made'])} on {PUB['made']:,})", "same rule"],
                 ["Hit: Brier, real vs predicted rate per difficulty", "attacks in the log", "Brier above constant baseline"],
                 ["PSI of actions, distance, mode, difficulty", "weekly vs reference window", f"warning > {th['psi_warn']}, alert > {th['psi_alert']}"],
                 ["Opinions and incidents", "forms", "any high-severity incident"]], [5.6 * cm, 5.6 * cm, 4.8 * cm]),
          Paragraph("Drift detection: Population Stability Index PSI = Σ (cur − ref)·ln(cur/ref) over categories or the 10 reference deciles; performance loss: "
                    "weekly score of the chosen and in-game models on the new real actions (the log contains the true next action), 2-week mean. Frequency and "
                    "owners: automatic weekly run (GitHub Action) reviewed by J.A (model, data); J.Q (website, reports); J.P checks the game integration at each "
                    "release; full CRISP-ML review each semester or after a big update. Retrain when any PSI > 0.20, the 2-week accuracy falls below the baseline "
                    "− 5 points, combat changes in a game update, or the real data doubles; deploy the new model only if it passes the Phase 4 rule and is not worse "
                    "on test. Quality controls: server-side validation and rate limits, Phase 2 rules, leakage checks, the in-game model as fallback, fairness by "
                    "difficulty, anonymous ids with opt-out, model + data report + model card committed together (traceable and reversible).", P),
          Paragraph("Initial monitoring test — SIMULATED operations", H2),
          Paragraph(f"<b>SIMULATED:</b> {M['note']} Weekly batches of 25 new simulated players; a change in how people play is injected from week 4.", WARN),
          fig("fig_mon_psi.png", 13), fig("fig_mon_accuracy.png", 13),
          table([["Week", "Drift", "Actions", "Accuracy", "2-week", "In-game", "PSI actions", "PSI distance", "Status"]] +
                [[f"{w['week']} (SIM)", w["drift_injected"], f"{w['actions']:,}", pct(w["accuracy"]), pct(w["accuracy_2w"]), pct(w["in_game_accuracy"]),
                  f3(w["psi_target"]), f3(w["psi_ne"]), w["status"]] for w in M["weeks"]],
                [1.6 * cm, 1.2 * cm, 1.6 * cm, 1.7 * cm, 1.6 * cm, 1.7 * cm, 1.8 * cm, 1.9 * cm, 2 * cm]),
          Paragraph(f"Reading: the first drift alert is raised in week {M['first_drift_alert']}"
                    + (f" and accuracy crosses the retraining line in week {M['first_perf_alert']}." if M["first_perf_alert"] else
                       ", while the 2-week accuracy never crosses the retraining line: the input drift is detected before any visible loss of accuracy.")
                    + f" Real indicator now: in-game accuracy {pct(PUB['right'] / PUB['made'])} over {PUB['made']:,} real predictions from {PUB['players']} players.", P)]

    # ------------------------------------------------------------------ conclusions
    s += [Paragraph("8. Conclusions", H1),
          Paragraph(f"(1) The game's own online models are already a strong baseline: on new players the in-game Markov reaches {pct(A['test'][IG]['accuracy'])} "
                    f"against {pct(A['test']['Majority class']['accuracy'])} for always guessing the most common action, and in production it is right "
                    f"{pct(PUB['right'] / PUB['made'])} of the time for real accounts. (2) Global models only compete when they receive the player's own history; "
                    f"the chosen {A['chosen']} reaches {pct(at['accuracy'])} with clearly better probabilities (log loss {f3(at['log_loss'])} vs "
                    f"{f3(A['test'][IG]['log_loss'])}). (3) For the hit probability, one pooled logistic regression with attack × situation interactions beats the "
                    f"per-attack SGD bandit (Brier {f3(bt['brier'])} vs {f3(B['test'][IB]['brier'])}): sharing information across players helps where each player "
                    f"has few attacks. (4) For the dodge, shrinking each player's mean towards the global mean beats the in-game running mean. (5) The pipeline, "
                    f"report and app are generated from the same files, so the {'simulated' if sim else 'real'} results shown here can be regenerated on real data "
                    "with one command; the main limitation is the amount and diversity of real players, which the new anonymous log addresses.", P),
          Paragraph("9. References", H1)]
    for ref in ["Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., & Müller, K.-R. (2021). Towards CRISP-ML(Q): A machine learning "
                "process model with quality assurance methodology. Machine Learning and Knowledge Extraction, 3(2), 392–413.",
                "Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.",
                "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1), 1–3.",
                "Dawid, A. P. (1984). Present position and potential developments: The prequential approach. Journal of the Royal Statistical Society A, 147(2), 278–292.",
                "Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS 30 (basis of HistGradientBoosting).",
                "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
                "Siddiqi, N. (2006). Credit Risk Scorecards (Population Stability Index). Wiley.",
                "Yannakakis, G. N., & Togelius, J. (2018). Artificial Intelligence and Games. Springer (player modelling).",
                "Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. WWW 2010.",
                "UMBRAL source code: scripts/ml_brain.gd, scripts/ml_log.gd, tools/supabase_ml_log.sql (game repository of the team)."]:
        s.append(Paragraph("• " + ref, SM))

    out = REPORTS / "UMBRAL_CRISP-ML_report.pdf"

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#8a8984"))
        canvas.drawString(2 * cm, 1.2 * cm, f"UMBRAL · CRISP-ML · data source: {R['source']}")
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                      title="UMBRAL CRISP-ML technical report", author="J.A · J.P · J.Q").build(s, onFirstPage=footer, onLaterPages=footer)
    print("PDF ->", out)


if __name__ == "__main__":
    main()
