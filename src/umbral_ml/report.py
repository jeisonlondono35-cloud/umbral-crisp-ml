"""Technical report "R3A2 Traces in data" (PDF).

Same layout as the team's previous report (R2A2): university cover, table of contents, numbered sections
with simple bordered tables, screenshots of the app, a debugging note, the Git workflow, discussion,
conclusion and references. Written in simple English (CEFR A2-B1).

Every number comes from the files that the Flask app also shows (results.json, data_report.json,
monitoring, expert review), so the report and the app cannot disagree.
"""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus.doctemplate import BaseDocTemplate, PageTemplate
from reportlab.platypus.frames import Frame
from reportlab.platypus.tableofcontents import TableOfContents

from .config import FIG, MON, RAW, REPORTS, ROOT

TITLE = "Traces in Data: How UMBRAL's Bosses Learn from the Player"
FILE_NAME = "R3A2 Traces in data.pdf"
APP_URL = "https://umbral-crisp-ml.onrender.com"
REPO_URL = "https://github.com/jeisonlondono35-cloud/umbral-crisp-ml"
GAME_URL = "https://juego-pixelart.onrender.com"
AUTHORS = ["Jeison Arboleda Londoño", "Juan Pablo Lugo Cárdenas", "Johan Sebastián Quintero Ramírez"]
INSTITUTION = "Universidad de Cundinamarca"
TEACHER = "Edison Gustavo Cañón Varela"
IMG = ROOT / "docs" / "img"
IG, IB = "In-game Markov-2 (decay 0.97)", "In-game bandit (SGD, lr 0.3)"
TMP = REPORTS / "_crops"


# ------------------------------------------------------------------ fonts: Calibri/Cambria like the Word model
def _fonts() -> tuple[str, str, str]:
    win = Path("C:/Windows/Fonts")
    try:
        pdfmetrics.registerFont(TTFont("Calibri", str(win / "calibri.ttf")))
        pdfmetrics.registerFont(TTFont("Calibri-Bold", str(win / "calibrib.ttf")))
        pdfmetrics.registerFont(TTFont("Calibri-Italic", str(win / "calibrii.ttf")))
        pdfmetrics.registerFont(TTFont("Cambria-Bold", str(win / "cambriab.ttf")))
        from reportlab.lib.fonts import addMapping
        addMapping("Calibri", 0, 0, "Calibri")
        addMapping("Calibri", 1, 0, "Calibri-Bold")
        addMapping("Calibri", 0, 1, "Calibri-Italic")
        addMapping("Calibri", 1, 1, "Calibri-Bold")
        return "Calibri", "Calibri-Bold", "Cambria-Bold"
    except Exception:          # Linux (GitHub Actions): same layout with Helvetica
        return "Helvetica", "Helvetica-Bold", "Times-Bold"


BODY, BOLD, SERIF = _fonts()
P = ParagraphStyle("p", fontName=BODY, fontSize=11, leading=14.5, spaceAfter=7)
H1 = ParagraphStyle("h1", fontName=BOLD, fontSize=15, leading=19, spaceBefore=16, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName=BOLD, fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4, keepWithNext=1)
BUL = ParagraphStyle("bul", parent=P, leftIndent=18, bulletIndent=6, spaceAfter=3)
CELL = ParagraphStyle("cell", fontName=BODY, fontSize=10, leading=12.2)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName=BOLD)
CAP = ParagraphStyle("cap", parent=P, fontSize=9.5, leading=12, textColor=colors.HexColor("#444444"), alignment=TA_CENTER)
CEN = ParagraphStyle("cen", parent=P, alignment=TA_CENTER, fontSize=12, leading=17, spaceAfter=4)
LINK = ParagraphStyle("link", parent=CEN, fontName=BOLD, textColor=colors.HexColor("#0000ee"))


def pct(v, d=1):
    return "-" if v is None else f"{100 * float(v):.{d}f}%"


def f3(v, d=3):
    return "-" if v is None else f"{float(v):.{d}f}"


def para(t):
    return Paragraph(t, P)


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(rows, widths, bold_rows=()):
    data = [[Paragraph(str(c), CELLB if i == 0 or i in bold_rows else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.6, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4)]))
    t.spaceBefore, t.spaceAfter = 2, 10
    return t


def picture(path: Path, width_in: float, crop=None, caption=None):
    from PIL import Image as PI
    if not path.exists():
        return Spacer(1, 1)
    im = PI.open(path)
    if crop:
        TMP.mkdir(exist_ok=True)
        crop = (crop[0], crop[1], min(crop[2], im.width), min(crop[3], im.height))
        im = im.crop(crop)
        path = TMP / f"{path.stem}_{crop[1]}_{crop[3]}.png"
        im.save(path)
    w, h = im.size
    height = width_in * h / w
    if height > 8.2:                      # never taller than the page
        width_in, height = width_in * 8.2 / height, 8.2
    out = [Image(str(path), width=width_in * inch, height=height * inch)]
    if caption:
        out.append(Paragraph(caption, CAP))
    out.append(Spacer(1, 6))
    return KeepTogether(out)


class Doc(BaseDocTemplate):
    """Page number top-left (like the Word model) and headings collected for the table of contents."""

    def __init__(self, path, **kw):
        super().__init__(path, pagesize=LETTER, leftMargin=1.15 * inch, rightMargin=1.15 * inch,
                         topMargin=1.1 * inch, bottomMargin=1 * inch, **kw)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._number)])

    @staticmethod
    def _number(canvas, doc):
        canvas.saveState()
        canvas.setFont(BODY, 11)
        canvas.drawString(1.15 * inch, LETTER[1] - 0.65 * inch, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and getattr(f, "toc_level", None) is not None:
            key = f"h{f.toc_key}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (f.toc_level, f.getPlainText(), self.page, key))


_HN = [0]


def h(text, level=0):
    p = Paragraph(text, H1 if level == 0 else H2)
    p.toc_level = level
    _HN[0] += 1
    p.toc_key = _HN[0]
    return p


def main() -> None:
    R = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    D = R["data"]
    M = json.loads((MON / "monitoring_simulated.json").read_text(encoding="utf-8"))
    E = json.loads((ROOT / "docs" / "expert_review.json").read_text(encoding="utf-8"))
    PUB = json.loads((RAW / "ml_public.json").read_text(encoding="utf-8"))
    A, B, C = R["A"], R["B"], R["C"]
    b, af = D["before"], D["after"]
    sim = R["source"] == "SIMULATED"
    real_acc = PUB["right"] / PUB["made"] if PUB.get("made") else None
    at, bt = A["test"][A["chosen"]], B["test"][B["chosen"]]
    ck = list(C["test"])[0]
    s = []

    # ------------------------------------------------------------------ cover
    if (IMG / "logo_udec.png").exists():
        s += [Spacer(1, 0.3 * inch), Image(str(IMG / "logo_udec.png"), width=1.25 * inch, height=1.25 * inch * 150 / 99)]
    s += [Spacer(1, 0.55 * inch),
          Paragraph(TITLE, ParagraphStyle("t", parent=CEN, fontName=BOLD, fontSize=16, leading=21, spaceAfter=12)),
          Paragraph("Autor: " + AUTHORS[0], CEN)] + [Paragraph(a, CEN) for a in AUTHORS[1:]] + [
          Spacer(1, 0.3 * inch), Paragraph("Institución: " + INSTITUTION, CEN), Paragraph("Profesor: " + TEACHER, CEN),
          Spacer(1, 0.3 * inch), Paragraph("<b>Selected Topic:</b>", CEN),
          Paragraph("Technical report: how a video game learns from the way people play (CRISP-ML)", CEN),
          Spacer(1, 0.7 * inch),
          Paragraph(f'<link href="{REPO_URL}"><u>GitHub Repository</u></link>', LINK), Spacer(1, 0.15 * inch),
          Paragraph(f'<link href="{APP_URL}"><u>Render Deployment URL</u></link>', LINK),
          Spacer(1, 0.1 * inch),
          Paragraph(f"{REPO_URL}<br/>{APP_URL}", ParagraphStyle("u", parent=CAP, fontSize=9)),
          PageBreak()]

    # ------------------------------------------------------------------ contents
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle("t0", fontName=BODY, fontSize=11, leading=15, leftIndent=8),
                       ParagraphStyle("t1", fontName=BODY, fontSize=11, leading=15, leftIndent=24)]
    toc.dotsMinLevel = 0
    s += [Paragraph("Contenido", ParagraphStyle("c", fontName=SERIF, fontSize=17, leading=22, spaceAfter=8,
                                                 textColor=colors.HexColor("#1f3864"))), toc, PageBreak()]

    # ------------------------------------------------------------------ 1. introduction
    s += [h("1. Introduction"),
          para("This report explains a Machine Learning (ML) project about our own video game, <b>UMBRAL</b>. UMBRAL is a "
               "pixel-art action game made with Godot. People can play it on the web, on Android and on PC. In the game, "
               "the bosses (the big enemies) learn how you play while you fight. For example, if you always use your "
               "shield, the boss learns it and tricks you."),
          para("The game already has three small ML models inside it. In this project we take these models out of the game, "
               "we study them with data, we compare them with other algorithms, we choose the best one, and we put it in a "
               "Flask web application. We follow the six phases of <b>CRISP-ML</b>, a standard process for ML projects."),
          para("The main goals of this project are:")]
    s += bullets(["Record real play data from the game, without names or accounts.",
                  "Clean the data and prepare it, without data leakage.",
                  "Compare several algorithms for three questions about the player.",
                  "Choose the best model with a rule that we fixed before training.",
                  "Put the model in a Flask app with a demo, and plan how to watch it over time.",
                  "Write all the work in a web page with the six phases, and in this report."])
    if sim:
        s.append(para(f"<b>Important note about the data.</b> The real recording started in November 2026. Right now it has "
                      f"{D['real_available']['actions']:,} real actions from {D['real_available']['players']} players. This is "
                      f"less than our minimum ({D['real_available']['min_actions']:,} actions from {D['real_available']['min_players']} "
                      f"players). So the model results in this report use <b>simulated players</b>: computer players that follow "
                      f"the rules of the game. We always write SIMULATED or REAL next to the numbers. Every Monday the system "
                      f"trains again, and it changes to the real data by itself when there is enough."))

    # ------------------------------------------------------------------ 2. concepts
    s += [h("2. Machine Learning Concepts"),
          para("In this project the computer learns from <b>traces</b>: the small records that each player leaves while fighting. "
               "From these traces it learns patterns, and it uses the patterns to predict what the player will do. The table "
               "below explains the main words we use in this report."),
          h("2.1 Core Concepts", 1),
          table([["Concept", "What it means in this project"],
                 ["Player action", "One thing the player does in a fight: shoot, melee hit, block, dodge, jump or hide."],
                 ["Trace / event", "One line of data that the game records, for example \"the player dodged at 120 px from the boss\"."],
                 ["Feature", "An input for the model, for example the last two actions or the player's life."],
                 ["Target", "The answer that the model must predict, for example the next action."],
                 ["Model", "The program that learns the pattern from the data and makes predictions."],
                 ["Training", "The moment when the model learns from data."],
                 ["Validation / Test", "Data that the model did not see. We use it to check if the model is really good."],
                 ["Online learning", "The model learns during the game, one action at a time. The game models work like this."],
                 ["Data leakage", "A mistake: information from the test data enters the training. The results look better than they are."],
                 ["Drift", "The data changes over time (for example after a game update), so the model can get worse."]],
                [1.6, 4.4]),
          h("2.2 The Six Phases of CRISP-ML", 1),
          table([["Phase", "Main question", "Where in this report"],
                 ["1. Business and Data Understanding", "What is the problem, and what data do we have?", "Section 3"],
                 ["2. Data Engineering", "How do we clean and prepare the data?", "Section 4"],
                 ["3. Model Engineering", "Which algorithms do we try, and how do we train them?", "Section 5"],
                 ["4. Model Evaluation", "Which model is the best, and is it good enough?", "Section 6"],
                 ["5. Model Deployment", "How can people use the model?", "Section 7"],
                 ["6. Monitoring and Maintenance", "How do we watch the model and update it?", "Section 8"]],
                [2.3, 2.6, 1.1])]

    # ------------------------------------------------------------------ 3. phase 1
    s += [h("3. Phase 1: Business and Data Understanding"),
          h("3.1 The Problem and the People", 1),
          para("UMBRAL wants bosses that \"learn you\". If a player repeats the same pattern, the boss must notice it. But the "
               "boss must also be fair. The approved project studies all the ML of the game: three models that already learn "
               "online inside the game (file <i>scripts/ml_brain.gd</i>)."),
          table([["Who", "Role", "What they need"],
                 ["Players", "They use the game and create the data.", "Fun fights that change, and fair bosses."],
                 ["J.A (Jeison)", "Creator of the game, domain expert, owner of the model and the data.", "To know if the game ML works, and which model to use."],
                 ["J.P (Juan Pablo)", "Game developer.", "A fast model that works on phones."],
                 ["J.Q (Johan)", "Web developer.", "Clear numbers and reports for the web."],
                 ["Teacher", "Evaluator.", "A clear CRISP-ML process."]], [1.3, 2.4, 2.3]),
          h("3.2 Business Goals and ML Goals", 1),
          table([["Business goal", "ML goal", "Type of task"],
                 ["The boss knows what you will do next.", "Predict the next action better than the game model and better than \"always the most common action\".", "Classification (6 classes)"],
                 ["The boss chooses good attacks, but fair ones.", "Give a good probability that an attack hits the player.", "Classification (yes / no) with probability"],
                 ["The boss aims where you go.", "Predict where the player moves when the boss charges.", "Regression (x, y)"],
                 ["It works on a phone during a fight.", "Fast predictions (milliseconds) and small files.", "Limit for every model"]],
                [1.9, 2.6, 1.5]),
          h("3.3 Data Sources", 1),
          table([["Source", "What it has", "Status"],
                 ["Anonymous play log (Supabase, table umbral_ml_log)", "We added it to the game for this project. It saves every action in a fight, every boss attack and every dodge. It uses random ids, not names. Players can turn it off.",
                  f"REAL: {D['real_available']['actions']:,} actions from {D['real_available']['players']} players (started November 2026)"],
                 ["Accuracy per account (umbral_ml_public)", "How many times the game model was right, for each account.",
                  f"REAL: {PUB.get('players', 0)} players, {PUB.get('made', 0):,} predictions, {pct(real_acc)} correct"],
                 ["Simulated players (simulate.py)", "Computer players that follow the game rules. Same format as the real log.",
                  f"SIMULATED: {b['events']:,} events, {b['players']} players" if sim else "SIMULATED: only for the monitoring test"]],
                [1.7, 2.8, 1.5]),
          h("3.4 Variables and Target", 1),
          table([["Variable", "Meaning"],
                 ["a (target 1)", "The action: shoot, melee, block, dodge, jump or hide."],
                 ["gap", "Time since the last action (more than 4 seconds starts a new group of actions)."],
                 ["ne", "Distance to the nearest enemy, in pixels."],
                 ["w, m, d, hp", "World (1 to 5), game mode, difficulty and life."],
                 ["mv and x[8]", "The boss attack and 8 numbers about the situation (close, far, in the air, blocking, moving, low life...)."],
                 ["y (target 2)", f"1 if the attack hit the player, 0 if not (hit rate: {pct(b['hit_rate'])})."],
                 ["ox, oy (target 3)", "Where the player moved during a boss charge, in pixels."]], [1.5, 4.5]),
          para(f"Size of the data in this build ({R['source']}): <b>{D['rows']['actions']:,}</b> actions, "
               f"<b>{D['rows']['hits']:,}</b> boss attacks and <b>{D['rows']['dodges']:,}</b> dodges."),
          picture(FIG / "fig_classes.png", 5.2, caption="Figure 1. How often each action appears. The classes are not balanced."),
          h("3.5 Data Quality, Limits and Risks", 1)]
    s += bullets(["<b>Quality:</b> the game writes the data itself, so the values have the right type. We found some problems and we fix them in Phase 2.",
                  "<b>Relevance:</b> the data is exactly what the game models see when they decide.",
                  f"<b>Enough data:</b> our minimum for real data is {D['real_available']['min_actions']:,} actions from {D['real_available']['min_players']} players.",
                  "<b>Risks:</b> few real players at the start (friends of the team), players change over time, privacy, and a boss that feels unfair or too difficult."])
    s += [h("3.6 Success Criteria", 1),
          para("We wrote these rules <b>before</b> training, so we cannot change them after we see the results."),
          table([["#", "Criterion"],
                 ["S1", "Next action: test accuracy is at least 10 points higher than \"always the most common action\"."],
                 ["S2", "Next action: test accuracy is not lower than the game model."],
                 ["S3", "Hit: the Brier score is lower than the game model and lower than a constant guess."],
                 ["S4", "Dodge: the average error is lower than the game model."],
                 ["S5", "Speed: one prediction takes less than 50 ms."],
                 ["S6", "We choose the model with the validation data. We use the test data only one time, at the end."]],
                [0.5, 5.5]),
          picture(IMG / "app_phase1.png", 6.0, crop=(0, 60, 1280, 1250), caption="Figure 2. Phase 1 page in the Flask app.")]

    # ------------------------------------------------------------------ 4. phase 2
    s += [h("4. Phase 2: Data Engineering"),
          para("In this phase we clean the raw data and we create the inputs for the models. The code is in "
               "<i>src/umbral_ml/prepare.py</i>. The table shows the problems that we found, before and after cleaning."),
          h("4.1 Problems in the Data (Before and After)", 1),
          table([["Check", "Before", "After", "What we did"],
                 ["Events", f"{b['events']:,}", f"{af['events']:,}", ""],
                 ["Duplicated events", b["duplicate_events"], af["duplicate_events"], "We removed them. They come from a batch that was sent two times."],
                 ["Distance = 9999 (no enemy)", b["ne_sentinel_9999"], af["ne_sentinel_9999"], "9999 is not a real distance. We changed it to \"missing\" and filled it with the median of the training data."],
                 ["Unknown actions", b["invalid_action"], af["invalid_action"], "We remove them if they appear."],
                 ["Very long pauses (10 min or more)", b["gap_over_10min"], af["gap_over_10min"], "We keep them, but with a limit of 10 minutes."]],
                [1.7, 0.7, 0.7, 2.9]),
          picture(FIG / "fig_before_after.png", 5.6, caption="Figure 3. The data before and after cleaning."),
          h("4.2 Each Transformation", 1)]
    rows = [["#", "Step", "Why"]] + [[i + 1, t["step"], t["why"]] for i, t in enumerate(D["transformations"])]
    rows.append([len(rows), "Encoding and scaling", "Inside each model pipeline: one-hot encoding for categories, median for missing numbers, and scaling. It is fitted only with the training data."])
    s += [table(rows, [0.35, 1.75, 3.9]),
          h("4.3 New Features (Feature Engineering)", 1)]
    s += bullets(["<b>prev1, prev2:</b> the last two actions of the player. After 4 seconds without actions they become \"start\", like in the game.",
                  "<b>hist_*:</b> how often this player did each action before. This helps a general model to know each player.",
                  "<b>mk_*:</b> the probabilities of the player's own online model at that moment.",
                  "<b>boss:</b> 1 if the fight is against a boss.",
                  "<b>attack x situation:</b> one slope for each boss attack, like the game model."])
    s += [para("All these features use only the <b>past</b> of the same player. They never use the future."),
          h("4.4 Train, Validation and Test Split", 1),
          table([["Part", "Groups", "Actions", "Attacks", "What we use it for"],
                 ["Train", D["split"]["train"], f"{D['split_rows']['actions']['train']:,}", f"{D['split_rows']['hits']['train']:,}", "To train, and to tune the models with cross-validation"],
                 ["Validation", D["split"]["val"], f"{D['split_rows']['actions']['val']:,}", f"{D['split_rows']['hits']['val']:,}", "To choose the best model"],
                 ["Test", D["split"]["test"], f"{D['split_rows']['actions']['test']:,}", f"{D['split_rows']['hits']['test']:,}", "One final check, at the end"]],
                [1.0, 0.7, 0.9, 0.9, 2.5]),
          para(f"We split the data by <b>{'player' if D['split']['group_by'] == 'pid' else 'game session'}</b>: all the data of one "
               f"{'player' if D['split']['group_by'] == 'pid' else 'session'} goes to only one part. If the same player was in train and "
               "in test, the model could remember the player, and the result would be too good. This is how we avoid data leakage."),
          picture(IMG / "app_phase2.png", 6.0, crop=(0, 60, 1280, 1320), caption="Figure 4. Phase 2 page in the Flask app.")]

    # ------------------------------------------------------------------ 5. phase 3
    s += [h("5. Phase 3: Model Engineering"),
          para("We tried several algorithms for each question. All of them use the same data and the same rules, so the "
               "comparison is fair. The code is in <i>src/umbral_ml/train.py</i>."),
          h("5.1 The Algorithms We Tried", 1),
          table([["Model", "Why we tried it"],
                 ["Most common action (baseline)", "The simplest guess. A useful model must be better than this."],
                 ["Global Markov-1 table", "A simple model of sequences: the same for all players."],
                 ["In-game Markov-2 (decay 0.97)", "The model that the game uses now. It learns each player online and forgets old data slowly."],
                 ["Tuned online Markov", "The same model, but with other numbers (order and decay) chosen with data."],
                 ["Logistic regression", "Simple and fast, gives good probabilities, easy to copy into the game code."],
                 ["Random forest", "Many decision trees together. It finds non-linear patterns."],
                 ["Gradient boosting (HistGB)", "Strong for tabular data, and it works with missing values."]],
                [2.0, 4.0]),
          para("For the <b>hit</b> question we also tried: a constant hit rate, the game bandit (one logistic regression for each "
               "attack, trained online), logistic regression with and without attack x situation, a decision tree, a random forest "
               "and gradient boosting. For the <b>dodge</b> we tried: no movement, the global average, the game average and an "
               "average for each player that moves towards the global average."),
          h("5.2 Training Configuration", 1),
          table([["Parameter", "Value", "Reason"],
                 ["Cross-validation", "GroupKFold, 5 folds", "Groups by player, so the same player is never in two folds."],
                 ["Hyperparameter search", "GridSearchCV", "It tries all the combinations in a small grid."],
                 ["Logistic regression C", "0.1, 1, 10", "How strong the regularization is."],
                 ["Random forest", "depth 8/16, leaf 5/20, 150 trees", "Depth controls how complex the trees are."],
                 ["Gradient boosting", "learning rate 0.05/0.1, depth 3/6", "Small steps and short trees avoid overfitting."],
                 ["Markov order and decay", "order 1/2, decay 0.85 to 1.0", "How much history and how much memory the game model uses."],
                 ["Bandit learning rate", "0.03 to 0.6", "How fast the game bandit learns."],
                 ["Random seed", "42", "The results are always the same."],
                 ["Library", f"scikit-learn {R['sklearn']}", "Standard ML library for Python."]],
                [1.7, 1.8, 2.5]),
          h("5.3 Experiment Log", 1),
          para(f"We saved every experiment ({len(R['experiments'])} in total) in <i>reports/experiments.csv</i>. "
               "The table shows the experiments for the next-action question.")]
    rows = [["Model", "Features", "CV accuracy", "Validation", "Train"]]
    for e in R["experiments"]:
        if e["task"] == "A":
            rows.append([e["model"], e["features"], "-" if e["cv_mean"] is None else f"{pct(e['cv_mean'])} ± {pct(e['cv_std'])}",
                         pct(e["val"]), pct(e["train"]) if e["train"] is not None else "-"])
    s += [table(rows, [2.0, 1.5, 1.1, 0.8, 0.6]),
          h("5.4 How the Game Knowledge Helped", 1)]
    s += bullets(["We split by player, because the model must work for a new player.",
                  "We use the last two actions and the 4-second rule, because fights come in short groups of actions.",
                  "We add the history of each player, because every player has a style.",
                  "We give one slope to each boss attack, because a trick attack beats the shield but a jump beats a ground wave.",
                  "We test the online models step by step (predict first, then learn), like in the real game."])
    s.append(picture(IMG / "app_phase3.png", 6.0, crop=(0, 60, 1280, 1250), caption="Figure 5. Phase 3 page in the Flask app."))

    # ------------------------------------------------------------------ 6. phase 4
    lc = A["learning_curve"]
    ga = [e for e in R["experiments"] if e["task"] == "A" and e["gap_train_val"] is not None]
    worst = max(ga, key=lambda e: e["gap_train_val"])
    s += [h("6. Phase 4: Model Evaluation"),
          h("6.1 Metrics", 1),
          table([["Question", "Main metric", "Why"],
                 ["Next action", "Accuracy (and macro-F1, log loss)", "The boss uses the most probable action, so we count how many times the model is right. Macro-F1 helps because the classes are not balanced."],
                 ["Hit", "Brier score (and ROC-AUC)", "The boss compares probabilities, so they must be correct. A lower Brier score is better."],
                 ["Dodge", "Average error in pixels", "It is the distance between the boss aim and the real place."]],
                [1.1, 1.9, 3.0]),
          para("Our data has groups (players), so normal K-fold cross-validation is not correct. We use <b>GroupKFold by player</b>, "
               "a separate group of validation players, and step-by-step (prequential) testing for the online models."),
          h("6.2 Comparison of the Models", 1),
          picture(FIG / "fig_cmp_a.png", 5.8, caption="Figure 6. Next action: validation accuracy. Blue = chosen model, dark gray = baselines."),
          picture(FIG / "fig_cmp_b.png", 5.8, caption="Figure 7. Hit: validation Brier score (lower is better).")]
    rows = [["Model", "Validation accuracy", "Macro-F1", "Log loss"]]
    for k, v in A["candidates"].items():
        rows.append([k + ("  (chosen)" if k == A["chosen"] else ""), pct(v["val"]["accuracy"]), f3(v["val"]["macro_f1"]), f3(v["val"]["log_loss"])])
    s += [table(rows, [3.0, 1.2, 0.9, 0.9])]
    rows = [["Model", "Validation Brier", "ROC-AUC"]]
    for k, v in B["candidates"].items():
        rows.append([k + ("  (chosen)" if k == B["chosen"] else ""), f3(v["val"]["brier"]), f3(v["val"]["roc_auc"])])
    s += [Spacer(1, 8), table(rows, [3.6, 1.3, 1.1])]
    rows = [["Dodge model", "Validation error (px)"]]
    for k, v in C["candidates"].items():
        rows.append([k + ("  (chosen)" if k == C["chosen"] else ""), f3(v["val"]["mean_error_px"], 1)])
    s += [Spacer(1, 8), table(rows, [4.2, 1.8]),
          h("6.3 Overfitting, Underfitting and Bias", 1)]
    s += bullets([f"<b>Overfitting:</b> a model is much better on the training data than on new data. The biggest gap is "
                  f"{worst['model']} ({100 * worst['gap_train_val']:.1f} points). The learning curve of the best general model goes "
                  f"from {pct(lc['val'][0])} with {lc['sizes'][0]:,} rows to {pct(lc['val'][-1])} with {lc['sizes'][-1]:,} rows.",
                  f"<b>Underfitting:</b> a general model without the history of the player gets only "
                  f"{pct(A['candidates']['Logistic regression [seq]']['val']['accuracy'])} (the most common action gets "
                  f"{pct(A['candidates']['Majority class']['val']['accuracy'])}). Players are very different, so the model needs the history of each player.",
                  "<b>Bias:</b> we check the results for each difficulty and each game mode: " +
                  "; ".join(f"{k.split('=')[1]} {pct(v['accuracy'])}" for k, v in A["test_segments"].items()) + ".",
                  "<b>Rare actions:</b> the model finds rare actions (jump, hide) less often: " +
                  ", ".join(f"{k} {pct(v, 0)}" for k, v in A["test_recall"].items()) + "."])
    s += [picture(FIG / "fig_learning_curve.png", 5.4, caption="Figure 8. Learning curve."),
          picture(FIG / "fig_experience.png", 5.4, caption="Figure 9. How fast each model learns a new player."),
          picture(FIG / "fig_confusion.png", 4.4, caption="Figure 10. Confusion matrix on the test data."),
          picture(FIG / "fig_calibration.png", 3.9, caption="Figure 11. Calibration of the hit probability."),
          h("6.4 Review with the Domain Expert", 1),
          para(f"The domain expert is <b>J.A</b>, the creator of the game. On {E['date']} we explained the results in simple words "
               f"and he answered four questions. {'(The review used the first run, with simulated players.)' if 'SIMULATED' in E['data_shown'] else ''}"),
          table([["Question", "Answer (Spanish)", "Answer (English)"]] +
                [[q["question"], q["answer_es"], q["answer_en"]] for q in E["answers"]], [2.4, 1.3, 2.3]),
          para("Because of this review we keep the fairness limits in the game (the boss waits for your dodge at most 60% of the "
               "time and always shows its warning) and we check the results for each difficulty."),
          h("6.5 The Chosen Models and the Final Test", 1),
          para(f"Rule: the best validation result wins. The chosen models are <b>{A['chosen']}</b> (next action), "
               f"<b>{B['chosen']}</b> (hit) and <b>{C['chosen']}</b> (dodge). We trained them again with train + validation and we "
               f"tested them <b>one time</b> with the test data.")]
    rows = [["Question", "Model", "Test result"]]
    for k, v in A["test"].items():
        rows.append(["Next action", k, f"accuracy {pct(v['accuracy'])}, macro-F1 {f3(v['macro_f1'])}"])
    for k, v in B["test"].items():
        rows.append(["Hit", k, f"Brier {f3(v['brier'])}, AUC {f3(v['roc_auc'])}"])
    for k, v in C["test"].items():
        rows.append(["Dodge", k, f"{f3(v['mean_error_px'], 1)} px"])
    s += [table(rows, [1.0, 2.9, 2.1])]
    sv = R["serving"]
    crit = [("S1", at["accuracy"] - A["test"]["Majority class"]["accuracy"] >= 0.10, f"{pct(at['accuracy'])} vs {pct(A['test']['Majority class']['accuracy'])}"),
            ("S2", at["accuracy"] >= A["test"][IG]["accuracy"], f"{pct(at['accuracy'])} vs {pct(A['test'][IG]['accuracy'])}"),
            ("S3", bt["brier"] < B["test"][IB]["brier"] and bt["brier"] < B["test"]["Constant hit rate"]["brier"], f"{f3(bt['brier'])} vs {f3(B['test'][IB]['brier'])} and {f3(B['test']['Constant hit rate']['brier'])}"),
            ("S4", C["test"][ck]["mean_error_px"] < C["test"]["In-game running mean"]["mean_error_px"], f"{f3(C['test'][ck]['mean_error_px'], 1)} px vs {f3(C['test']['In-game running mean']['mean_error_px'], 1)} px"),
            ("S5", sv["next_ms"] < 50 and sv["hit_ms"] < 50, f"{f3(sv['next_ms'], 1)} ms and {f3(sv['hit_ms'], 1)} ms")]
    s += [Spacer(1, 8), table([["Criterion", "Result", "Met?"]] + [[c, r, "Yes" if ok else "No"] for c, ok, r in crit], [1.0, 4.0, 1.0]),
          picture(IMG / "app_phase4.png", 6.0, crop=(0, 860, 1280, 1700), caption="Figure 12. Phase 4 page in the Flask app.")]

    # ------------------------------------------------------------------ 7. phase 5
    s += [h("7. Phase 5: Model Deployment"),
          h("7.1 Saved Files", 1),
          table([["File", "What it has"],
                 ["models/next_action.joblib", "The chosen next-action model with its preprocessing (one sklearn Pipeline)."],
                 ["models/hit.joblib", "The chosen hit model with its preprocessing."],
                 ["models/dodge.joblib", "The dodge model (global average and its weight)."],
                 ["models/model_card.json", "Data source, date, features, test results and library version."]],
                [2.1, 3.9]),
          para("The file <i>src/umbral_ml/predict.py</i> creates the features in exactly the same way as in training. So the "
               f"training and the app cannot be different. One prediction takes about {f3(sv['next_ms'], 0)} ms."),
          h("7.2 Flask Web Application", 1),
          table([["Page", "What it does"],
                 ["/", "Cover of the project: the problem, the main results and links."],
                 ["/phase/1 to /phase/6", "One page for each CRISP-ML phase, with tables, charts and decisions."],
                 ["/demo", "Live demo: you write the last actions of a player and the situation; the app answers the next action and the hit probability."],
                 ["/api/predict/next, /api/predict/hit, /api/predict/dodge", "The same predictions in JSON, for other programs (like the game)."],
                 ["/api/health, /monitoring/live", "State of the app and the last predictions (for monitoring)."],
                 ["/report.pdf", "This report."]], [2.3, 3.7]),
          picture(IMG / "app_home.png", 6.0, crop=(0, 0, 1280, 1000), caption="Figure 13. Cover page of the Flask app."),
          picture(IMG / "app_demo.png", 6.0, crop=(0, 230, 1280, 1320), caption="Figure 14. The live demo with a prediction."),
          h("7.3 Inputs and Outputs", 1),
          table([["Input", "Example"], ["history", "[\"shoot\", \"shoot\", \"dodge\"] (last actions, oldest first)"],
                 ["w, mode, diff", "2, \"story\", 1 (world, game mode, difficulty)"], ["hp, ne, boss", "4, 160, true (life, distance, boss fight)"],
                 ["move + situation", "\"feint\", close = 1, blocking = 1 (for the hit question)"]], [1.6, 4.4]),
          para("Output: the predicted action and its probability, the probability of every action, what the game model would say, "
               "and the time of the prediction. For the hit question: the probability that the attack hits."),
          h("7.4 How to Run It", 1)]
    s += bullets(["<b>Dependencies:</b> Python 3.14, Flask, scikit-learn, pandas, numpy, matplotlib, joblib, reportlab, requests, gunicorn (file requirements.txt).",
                  "<b>Install:</b> pip install -r requirements.txt",
                  "<b>Train everything:</b> python pipeline.py (data, models, monitoring and this PDF, about 5 minutes).",
                  "<b>Run the app:</b> python -m flask --app app.app run",
                  "<b>Online:</b> Render runs gunicorn app.app:app (file render.yaml)."])
    s += [h("7.5 First Steps for a New User", 1)]
    s += bullets(["Open the cover page and read the six phases in order (menu at the top).",
                  "Open <b>Demo</b>. Write some actions with commas, for example: shoot, shoot, dodge.",
                  "Choose the world, the life and the distance, and press <b>Predict</b>.",
                  "Read the predicted action and compare it with the game model.",
                  "In Phase 5 you can send an opinion or report a problem."])
    s += [h("7.6 Plan to Connect with the System of Activity 2", 1),
          para("The system of Activity 2 is our game, its website and the Supabase database (accounts, saved games and records)."),
          table([["Step", "What", "Who", "State"],
                 ["1. Data", "The game sends anonymous play data to Supabase.", "J.P / J.A", "Done"],
                 ["2. Training", "pipeline.py reads the data and trains every Monday (GitHub Actions).", "J.A", "Done"],
                 ["3. Serving", "The Flask app on Render gives the predictions.", "J.Q", "Done"],
                 ["4. In the game", "The boss asks the API when a fight starts and every 2 seconds. If there is no internet, it uses the game model.", "J.P", "Planned"],
                 ["5. Feedback", "Accuracy per account, the report form of the game and the form of the app.", "J.Q / J.A", "Running"]],
                [1.0, 3.2, 0.9, 0.9]),
          h("7.7 Monitoring, Problems and Opinions", 1)]
    s += bullets(["Every prediction is saved with its time, model, inputs, output and speed (/monitoring/live).",
                  "Problems can be reported in the app, in the game (Pause, Report error) or on GitHub. A big problem (the app does not work) is fixed the same day, or we go back to the last model.",
                  "Every model is saved in Git with its model card, so we can always go back.",
                  "Users can give their opinion with a score from 1 to 5. We read them every week."])

    # ------------------------------------------------------------------ 8. phase 6
    th = M["thresholds"]
    s += [h("8. Phase 6: Monitoring and Maintenance"),
          h("8.1 What We Watch", 1),
          table([["Indicator", "Where", "Alarm"],
                 ["App working and speed", "/api/health, /monitoring/live", "More than 50 ms, or any error"],
                 ["Real events per day", "Supabase (REAL)", "0 events when people are playing"],
                 ["Next-action accuracy (2-week average)", "Real actions vs predictions", f"{int(th['acc_drop'] * 100)} points lower than the first {th['baseline_weeks']} weeks"],
                 ["Game model accuracy, all accounts", f"Supabase (REAL: {pct(real_acc)})", "Same rule"],
                 ["Changes in the data (PSI)", "New week vs first weeks", f"Warning over {th['psi_warn']}, alarm over {th['psi_alert']}"],
                 ["Opinions and problems", "Forms", "Any big problem"]], [2.2, 2.0, 1.8]),
          h("8.2 How We Find Drift", 1),
          para("We use the <b>Population Stability Index (PSI)</b>. It compares the data of a new week with the data of the first "
               "weeks. If PSI is small, the data is stable. If PSI is bigger than 0.20, the data changed a lot. We also measure the "
               "accuracy every week with the real actions."),
          h("8.3 When, Who and When to Train Again", 1),
          table([["What", "When", "Who"],
                 ["Automatic training and report (GitHub Actions)", "Every Monday", "Automatic; J.A checks it"],
                 ["Review the indicators", "Every week", "J.A (model and data), J.Q (web and reports)"],
                 ["Check the game (fair bosses, works offline)", "Every new game version", "J.P"],
                 ["Full review of the project", "Every semester or after a big update", "J.A, J.P and J.Q"]], [2.6, 1.6, 1.8]),
          para(f"We train again when: PSI is over {th['psi_alert']}; the accuracy falls {int(th['acc_drop'] * 100)} points; a game update "
               "changes the fights; or the real data doubles. We only use a new model if it passes the Phase 4 rule."),
          h("8.4 First Monitoring Test (SIMULATED data)", 1),
          para("This test uses <b>simulated</b> data. We created 8 weeks of new players. From week 4 we changed the way they play "
               "(more dodges and fights from farther away), like after a game update. Then we ran the same pipeline as in production."),
          picture(FIG / "fig_mon_psi.png", 5.6, caption="Figure 15. PSI per week (SIMULATED)."),
          picture(FIG / "fig_mon_accuracy.png", 5.6, caption="Figure 16. Accuracy per week (SIMULATED).")]
    rows = [["Week", "Accuracy", "2-week avg.", "PSI actions", "PSI distance", "State"]]
    for w in M["weeks"]:
        rows.append([f"{w['week']} (SIM)", pct(w["accuracy"]), pct(w["accuracy_2w"]), f3(w["psi_target"]), f3(w["psi_ne"]), w["status"]])
    s += [table(rows, [0.9, 0.9, 1.0, 1.0, 1.0, 1.2]),
          para(f"The PSI found the change: the first alarm is in week <b>{M['first_drift_alert']}</b>. "
               + (f"The accuracy also fell in week {M['first_perf_alert']}." if M["first_perf_alert"] else
                  "The accuracy did not fall below the alarm line. This shows why we watch the data and not only the accuracy: the data warns us earlier.")),
          picture(IMG / "app_phase6.png", 6.0, crop=(0, 1640, 1280, 2290), caption="Figure 17. Monitoring test in the Flask app.")]

    # ------------------------------------------------------------------ 9. debugging note
    s += [h("9. A Debugging Note: Two Real Bugs"),
          para("Like in our last project, the most important lessons came from bugs."),
          h("9.1 The Columns in the Wrong Order", 1),
          para("In the first version, the best model had only <b>10% accuracy</b> on the test data. This was worse than guessing. "
               "The reason: scikit-learn gives the probabilities in <b>alphabetical order</b> (block, dodge, hide...), but our code "
               "read them in the game order (shoot, melee, block...). So the model was right, but we read the wrong column. "
               f"We fixed it with one small function (<i>proba_a</i>) that orders the columns by name. After the fix the accuracy was {pct(at['accuracy'])}. "
               "The same problem also changed the log loss, so we fixed it there too."),
          h("9.2 The App Failed Only on Render", 1),
          para("On our computer the app worked. On Render, the demo showed \"Internal Server Error\". To find the reason, we "
               "added an error handler that writes the full error in the log. The error was an <b>ImportError</b> in scikit-learn. "
               "We had started a thread that loads the models in the background, and the first visitor loaded scikit-learn at the "
               "same time. The fix: load the models <b>before</b> the app starts. Now every error is also saved, which is part of "
               "the incident plan of Phase 5.")]

    # ------------------------------------------------------------------ 10. git
    s += [h("10. Git and GitHub Workflow"),
          para("We used Git and GitHub to keep the project organized. The project has its own repository, separate from the game."),
          para(f"<b>Repository:</b> {REPO_URL}<br/><b>Branch:</b> master"),
          para("Workflow: code on the computer, then commit, then push to master. Each push makes Render update the app. Every "
               "Monday, GitHub Actions trains the models again and makes a new commit (\"Weekly retrain\")."),
          table([["Commit", "What it did"],
                 ["UMBRAL CRISP-ML: six phases, pipeline, models, Flask app and report", "First version of everything."],
                 ["Weekly retrain", "First automatic training on GitHub (Linux): same results as on our computer."],
                 ["App: load the models at start-up", "Faster first prediction."],
                 ["Weekly retrain ready for the first real data", "Works with few players (split by session)."],
                 ["App: log every error / load the models before serving", "Found and fixed the Render bug (Section 9.2)."]],
                [3.0, 3.0]),
          para("In the game repository we also added the anonymous play log (<i>scripts/ml_log.gd</i>) and the database script "
               "(<i>tools/supabase_ml_log.sql</i>).")]

    # ------------------------------------------------------------------ 11-13
    s += [h("11. Discussion"),
          para("This project shows some important Machine Learning lessons:")]
    s += bullets([f"<b>The game model was already good.</b> On new players it gets {pct(A['test'][IG]['accuracy'])}, and the most common action "
                  f"gets only {pct(A['test']['Majority class']['accuracy'])}. With real players it is right {pct(real_acc)} of the time.",
                  "<b>Each player is different.</b> General models only work well when they know the history of the player.",
                  f"<b>Sharing helps for the hit question.</b> One model for all players ({B['chosen']}) is better than one small model "
                  f"for each player: Brier {f3(bt['brier'])} vs {f3(B['test'][IB]['brier'])}.",
                  "<b>Small bugs can change everything.</b> The wrong column order made a good model look terrible (Section 9).",
                  "<b>Watch the data, not only the accuracy.</b> In the monitoring test, the PSI found the change first."])
    s += [h("12. Conclusion"),
          para("This project shows a full CRISP-ML process, from the problem to a working web application. The main results are:")]
    s += bullets(["We added an anonymous play log to the game, with an option to turn it off.",
                  "We cleaned the data, created features and split it by player, without data leakage.",
                  f"We compared {len(R['experiments'])} experiments and chose the models with a fixed rule.",
                  f"The chosen next-action model got {pct(at['accuracy'])} on the test data (game model: {pct(A['test'][IG]['accuracy'])}).",
                  "The models are in a Flask app on Render, with a live demo and an API.",
                  "The system trains again every week and watches the data for drift."])
    s += [h("Future Work", 1)]
    s += bullets(["Train with real data when there are enough players, and repeat the expert review.",
                  "Connect the bosses of the game to the API (Section 7.6).",
                  "Try sequence models, such as a small recurrent neural network, with more data.",
                  "Ask players if the bosses feel fair, and measure it."])
    s += [h("13. References")]
    for ref in ["Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., & Müller, K.-R. (2021). Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology. <i>Machine Learning and Knowledge Extraction, 3</i>(2), 392–413.",
                "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
                "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. <i>Monthly Weather Review, 78</i>(1), 1–3.",
                "Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5–32.",
                "Yannakakis, G. N., & Togelius, J. (2018). <i>Artificial intelligence and games</i>. Springer.",
                "Siddiqi, N. (2006). <i>Credit risk scorecards</i>. Wiley. (Population Stability Index)",
                "Pallets Projects. (2024). <i>Flask documentation</i>. https://flask.palletsprojects.com/",
                "Render. (2026). <i>Blueprint specification</i>. https://render.com/docs/blueprint-spec"]:
        s.append(Paragraph(ref, ParagraphStyle("ref", parent=P, leftIndent=18, firstLineIndent=-18)))

    out = REPORTS / FILE_NAME
    for old in REPORTS.glob("*.pdf"):
        if old.name != FILE_NAME:
            old.unlink()
    Doc(str(out), title="R3A2 Traces in data", author=", ".join(AUTHORS)).multiBuild(s)
    print("PDF ->", out)


if __name__ == "__main__":
    main()
