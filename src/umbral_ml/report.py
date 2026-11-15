"""Technical report "R3A2 Traces in data" (PDF), APA 7 student-paper style.

Format asked by the team: Arial 12 in black everywhere, APA 7 (title page, page number top right, double
spacing, unnumbered headings by level, APA tables and figures, in-text citations, references with
hanging indent) and simple English (CEFR A2-B1), written in the team's own voice.

Every number comes from the files that the Flask app also shows (results.json, data_report.json,
monitoring, expert review), so the report and the app cannot disagree.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
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

TITLE = "Traces in Data: How the Bosses of UMBRAL Learn From the Player"
FILE_NAME = "R3A2 Traces in data.pdf"
APP_URL = "https://umbral-crisp-ml.onrender.com"
REPO_URL = "https://github.com/jeisonlondono35-cloud/umbral-crisp-ml"
GAME_URL = "https://juego-pixelart.onrender.com"
AUTHORS = ["Jeison Arboleda Londoño", "Juan Pablo Lugo Cárdenas", "Johan Sebastián Quintero Ramírez"]
INSTITUTION = "Universidad de Cundinamarca"
COURSE = "Machine Learning"
TEACHER = "Edison Gustavo Cañón Varela"
DATE = "November 14, 2026"
IMG = ROOT / "docs" / "img"
IG, IB = "In-game Markov-2 (decay 0.97)", "In-game bandit (SGD, lr 0.3)"
TMP = REPORTS / "_crops"
MARGIN = 1 * inch
BLACK = colors.black


# ------------------------------------------------------------------ Arial 12, black (Helvetica on Linux: same metrics)
def _fonts() -> tuple[str, str, str, str]:
    win = Path("C:/Windows/Fonts")
    try:
        from reportlab.lib.fonts import addMapping
        for name, f in (("Arial", "arial.ttf"), ("Arial-Bold", "arialbd.ttf"), ("Arial-Italic", "ariali.ttf"),
                        ("Arial-BoldItalic", "arialbi.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(win / f)))
        addMapping("Arial", 0, 0, "Arial")
        addMapping("Arial", 1, 0, "Arial-Bold")
        addMapping("Arial", 0, 1, "Arial-Italic")
        addMapping("Arial", 1, 1, "Arial-BoldItalic")
        return "Arial", "Arial-Bold", "Arial-Italic", "Arial-BoldItalic"
    except Exception:
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"


REG, BOLD, ITAL, BOLDITAL = _fonts()
SIZE = 12
DOUBLE = 24          # APA: double spacing
P = ParagraphStyle("p", fontName=REG, fontSize=SIZE, leading=DOUBLE, firstLineIndent=0.5 * inch, textColor=BLACK,
                   alignment=TA_LEFT, spaceAfter=0)
P0 = ParagraphStyle("p0", parent=P, firstLineIndent=0)
H1 = ParagraphStyle("h1", parent=P0, fontName=BOLD, alignment=TA_CENTER, keepWithNext=1)       # APA level 1
H2 = ParagraphStyle("h2", parent=P0, fontName=BOLD, keepWithNext=1)                            # APA level 2
LIST = ParagraphStyle("list", parent=P0, leftIndent=0.75 * inch, bulletIndent=0.5 * inch)
CELL = ParagraphStyle("cell", fontName=REG, fontSize=SIZE, leading=14.5, textColor=BLACK)
CELLH = ParagraphStyle("cellh", parent=CELL, fontName=BOLD)
CAPN = ParagraphStyle("capn", parent=P0, fontName=BOLD, keepWithNext=1)                          # "Table 1"
CAPT = ParagraphStyle("capt", parent=P0, fontName=ITAL, keepWithNext=1)                          # its title
NOTE = ParagraphStyle("note", parent=P0, leading=18, spaceAfter=12)
CEN = ParagraphStyle("cen", parent=P0, alignment=TA_CENTER)
REF = ParagraphStyle("ref", parent=P0, leftIndent=0.5 * inch, firstLineIndent=-0.5 * inch)
NUM = {"table": 0, "figure": 0}
H3 = ParagraphStyle("h3", parent=P0, fontName=BOLDITAL, keepWithNext=1)                         # APA level 3
# Used by the final report (report_final.py), which puts this report and the Activity 2 report inside one document:
# the numbers written in the text ("Table 5", "Figures 3 and 4") move by an offset, appendix letters can be renamed
# (A -> C) and headings go one level down. With the defaults nothing changes.
PART = {"t_off": 0, "f_off": 0, "letters": {}, "shift": 0}


def _retext(t: str) -> str:
    def nums(m):
        off = PART["t_off"] if m.group(1).startswith("Table") else PART["f_off"]
        return m.group(1) + " " + re.sub(r"\d+", lambda d: str(int(d.group()) + off), m.group(2))
    if PART["t_off"] or PART["f_off"]:
        t = re.sub(r"\b(Tables?|Figures?) (\d+(?:(?:,| and| to) \d+)*)", nums, t)
    for a, b in PART["letters"].items():
        t = re.sub(rf"\b(Appendix|Table|Figure) {a}(?=\d|\b)", rf"\1 {b}", t)
    return t


def _us_date(iso: str) -> str:
    from datetime import date
    d = date.fromisoformat(iso)
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def pct(v, d=1):
    return "-" if v is None else f"{100 * float(v):.{d}f}%"


def f3(v, d=3):
    return "-" if v is None else f"{float(v):.{d}f}"


def para(t):
    return Paragraph(_retext(t), P)


def seriation(items):
    """APA seriation: a bulleted list, same font and spacing as the text."""
    return [Paragraph(t, LIST, bulletText="•") for t in items]


def apa_table(title, rows, widths, note=None, label=None):
    """APA 7 table: bold number, italic title, black lines on every cell (asked by the team), optional note.
    `label` (e.g. "B1") is used in the appendices instead of the running number."""
    if label and label[0] in PART["letters"]:
        label = PART["letters"][label[0]] + label[1:]
    if not label:
        NUM["table"] += 1
    data = [[Paragraph(str(c), CELLH if i == 0 else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.75, BLACK), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
    out = [Spacer(1, 6), Paragraph(f"Table {label or NUM['table']}", CAPN), Paragraph(title, CAPT), t]
    out.append(Paragraph(f"<i>Note.</i> {note}", NOTE) if note else Spacer(1, 14))
    return out


def apa_figure(path: Path, title, width_in, crop=None, note=None, label=None):
    """APA 7 figure: bold number, italic title, the image, optional note (`label` for appendices)."""
    from PIL import Image as PI
    if not path.exists():
        return []
    if label and label[0] in PART["letters"]:
        label = PART["letters"][label[0]] + label[1:]
    note = _retext(note) if note else note
    if not label:
        NUM["figure"] += 1
    im = PI.open(path)
    if crop:
        TMP.mkdir(exist_ok=True)
        crop = (crop[0], crop[1], min(crop[2], im.width), min(crop[3], im.height))
        im = im.crop(crop)
        path = TMP / f"{path.stem}_{crop[1]}_{crop[3]}.png"
        im.save(path)
    w, h = im.size
    height = width_in * h / w
    if height > 6.6:
        width_in, height = width_in * 6.6 / height, 6.6
    img = Image(str(path), width=width_in * inch, height=height * inch, hAlign="LEFT")
    parts = [Spacer(1, 6), Paragraph(f"Figure {label or NUM['figure']}", CAPN), Paragraph(title, CAPT), img]
    parts.append(Paragraph(f"<i>Note.</i> {note}", NOTE) if note else Spacer(1, 14))
    return [KeepTogether(parts)]


class Doc(BaseDocTemplate):
    """APA page: 1-inch margins, page number top right; headings collected for the table of contents."""

    def __init__(self, path, **kw):
        super().__init__(path, pagesize=LETTER, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                         bottomMargin=MARGIN, **kw)
        frame = Frame(MARGIN, MARGIN, self.width, self.height, id="f", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=self._number)])

    @staticmethod
    def _number(canvas, doc):
        canvas.saveState()
        canvas.setFont(REG, SIZE)
        canvas.setFillColor(BLACK)
        canvas.drawRightString(LETTER[0] - MARGIN, LETTER[1] - 0.6 * inch, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and getattr(f, "toc_level", None) is not None:
            key = f"h{f.toc_key}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (f.toc_level, f.getPlainText(), self.page, key))


_HN = [0]


def h(text, level=0):
    level = min(level + PART["shift"], 2)
    p = Paragraph(_retext(text), (H1, H2, H3)[level])
    p.toc_level = level if level < 2 else None      # level 3 headings stay out of the contents (shorter list)
    _HN[0] += 1
    p.toc_key = _HN[0]
    return p


def main(part: bool = False):
    """Write the PDF. With part=True (final report) return {"body", "appendix"} instead: no title page, contents,
    introduction, concepts, closing sections or references, and the numbering continues from the final report."""
    if not part:
        NUM.update(table=0, figure=0)
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
    maj = A["test"]["Majority class"]["accuracy"]
    ra = D["real_available"]
    th = M["thresholds"]
    sv = R["serving"]
    by_player = D["split"]["group_by"] == "pid"
    s = []

    if not part:
        # ------------------------------------------------------------------ title page (APA student paper)
        if (IMG / "logo_udec.png").exists():
            s += [Spacer(1, 0.2 * inch), Image(str(IMG / "logo_udec.png"), width=1.1 * inch, height=1.1 * inch * 150 / 99)]
        s += [Spacer(1, 0.6 * inch), Paragraph(f"<b>{TITLE}</b>", CEN), Spacer(1, DOUBLE)]
        s += [Paragraph(a, CEN) for a in AUTHORS]
        s += [Paragraph(INSTITUTION, CEN), Paragraph(COURSE, CEN), Paragraph(TEACHER, CEN), Paragraph(DATE, CEN),
              Spacer(1, DOUBLE),
              Paragraph("Selected topic: technical report about how a video game learns from the way people play (CRISP-ML)", CEN),
              Spacer(1, DOUBLE),
              Paragraph(f'<link href="{REPO_URL}"><u>GitHub Repository</u></link>', CEN),
              Paragraph(f'<link href="{APP_URL}"><u>Render Deployment URL</u></link>', CEN),
              PageBreak()]

        # ------------------------------------------------------------------ contents
        toc = TableOfContents()
        toc.levelStyles = [ParagraphStyle("t0", fontName=REG, fontSize=SIZE, leading=20, leftIndent=0, textColor=BLACK),
                           ParagraphStyle("t1", fontName=REG, fontSize=SIZE, leading=20, leftIndent=0.35 * inch, textColor=BLACK)]
        toc.dotsMinLevel = 0
        s += [Paragraph("<b>Table of Contents</b>", CEN), toc, PageBreak()]

        # ------------------------------------------------------------------ introduction
        s += [Paragraph(f"<b>{TITLE}</b>", CEN),
              h("Introduction"),
              para("We are three students and we also make a video game together. The game is called UMBRAL. It is a pixel-art "
                   "action game made with Godot, and people can play it on the web, on Android phones and on computers. In "
                   f"the game you fight bosses, and the bosses learn from you while you fight. If you always use your shield, "
                   "the boss notices it and tries to trick you. If you always jump to the same side, the boss starts to aim there."),
              para("To do this, the game already had three small machine learning models inside it. One model guesses your next "
                   "action, one model learns where you usually go when you dodge, and one model calculates if an attack will hit "
                   "you. We wrote them some months ago, but we never checked carefully if they really work. This project was "
                   "our chance to do it in a serious way."),
              para("For this activity we followed CRISP-ML, a process model for machine learning projects that has six phases "
                   "(Studer et al., 2021). We recorded data from the game, cleaned it, trained and compared several models, chose "
                   "the best one with a rule that we wrote before training, put it in a Flask web application, and made a plan "
                   "to watch it over time. The name of the report, Traces in Data, comes from the idea that every player leaves "
                   "small traces while playing, and the computer can learn from them."),
              para("Our main goals were the following:")]
        s += seriation(["record real play data from the game without names or accounts;",
                        "prepare the data without mixing the training data and the test data;",
                        "compare several algorithms and choose the best one with a fixed rule;",
                        "publish the model in a web application with a demo;",
                        "plan how to watch the model and when to train it again."])
        if sim:
            s.append(para(f"We need to be honest about one thing. We started recording real data in November 2026. When we "
                          f"wrote this report we had {ra['actions']:,} real actions from {ra['players']} players, and our minimum "
                          f"was {ra['min_actions']:,} actions from {ra['min_players']} players. For this reason, the model results in "
                          f"this version come from simulated players, which are computer players that follow the rules of the game. "
                          f"We always write SIMULATED or REAL next to the numbers. The system trains again every Monday, and it "
                          f"changes to the real data by itself when there is enough of it."))

        # ------------------------------------------------------------------ concepts
        s += [h("Machine Learning Concepts"),
              para("Before the phases, we want to explain some words that we use many times in this report. Table 1 shows them "
                   "with the meaning that they have in our project.")]
        s += apa_table("Main Concepts Used in the Project", [
            ["Concept", "Meaning in this project"],
            ["Player action", "One thing the player does in a fight: shoot, hit, block, dodge, jump or hide."],
            ["Trace or event", "One line of data that the game saves, for example \"the player dodged at 120 px from the boss\"."],
            ["Feature", "An input for the model, for example the last two actions or the life of the player."],
            ["Target", "The answer that the model must predict, for example the next action."],
            ["Training", "The moment when the model learns from data."],
            ["Validation and test data", "Data that the model did not see. We use it to check if the model is really good."],
            ["Online learning", "The model learns during the game, one action at a time. The game models work like this."],
            ["Data leakage", "A mistake where information from the test data enters the training, so the results look too good."],
            ["Drift", "The data changes over time, for example after a game update, and the model can get worse."]],
            [1.9, 4.6])
        s += [para("CRISP-ML organizes the work in six phases. Each phase answers one question, and the results of one phase "
                   "are the starting point of the next one. Table 2 shows the phases and the question of each one.")]
        s += apa_table("The Six Phases of CRISP-ML", [
            ["Phase", "Main question"],
            ["1. Business and data understanding", "What is the problem, and what data do we have?"],
            ["2. Data engineering", "How do we clean and prepare the data?"],
            ["3. Model engineering", "Which algorithms do we try, and how do we train them?"],
            ["4. Model evaluation", "Which model is the best, and is it good enough?"],
            ["5. Model deployment", "How can other people use the model?"],
            ["6. Monitoring and maintenance", "How do we watch the model and update it?"]],
            [2.8, 3.7], note="Adapted from Studer et al. (2021).")


    i_body = len(s)
    # ------------------------------------------------------------------ phase 1
    s += [h("Phase 1: Business and Data Understanding"),
          h("The Problem and the People", 1),
          para("The problem is simple to say but not so simple to solve. We want bosses that learn the player, but we also "
               "want them to be fair. Player modeling, which means building a model of how a person plays, is a common idea "
               "in game research (Yannakakis & Togelius, 2018). In our case the approved project includes all the machine "
               "learning of the game, so we studied the three models together."),
          para("Several people are involved. The players use the game and create the data, and they want fights that change "
               "but stay fair. Jeison (J.A) created the game and he is the domain expert and the owner of the model and the "
               "data. Juan Pablo (J.P) works on the game code, so he needs a model that is fast on phones. Johan (J.Q) works on "
               "the website and the reports, so he needs clear numbers. Finally, our teacher evaluates the process."),
          h("Business Goals and Machine Learning Goals", 1),
          para("We translated each goal of the game into a goal that we can measure. Table 3 shows this translation.")]
    s += apa_table("Business Goals Translated Into Machine Learning Goals", [
        ["Business goal", "Machine learning goal", "Type of task"],
        ["The boss knows what you will do next.", "Predict the next action better than the game model and better than always guessing the most common action.", "Classification, 6 classes"],
        ["The boss chooses good attacks, but fair ones.", "Give a correct probability that an attack hits the player.", "Classification with probability"],
        ["The boss aims where you go.", "Predict where the player moves when the boss charges.", "Regression (x, y)"],
        ["It works on a phone during a fight.", "Fast predictions and small model files.", "Limit for every model"]],
        [1.9, 3.0, 1.6])
    s += [h("Data Sources and Variables", 1),
          para("We used three sources of data (Table 4). The most important one is a play log that we added to the game for "
               "this project. It saves every action in a fight, every boss attack and every dodge, but it does not save names, "
               "emails or accounts. It only uses random numbers to tell players apart, and players can turn it off in the "
               "options of the game.")]
    s += apa_table("Data Sources of the Project", [
        ["Source", "What it contains", "Status"],
        ["Anonymous play log (Supabase)", "Actions, boss attacks with 8 situation numbers, results and dodges.", f"REAL: {ra['actions']:,} actions, {ra['players']} players"],
        ["Accuracy per account (Supabase)", "How many times the game model was right for each account.", f"REAL: {PUB.get('players', 0)} players, {PUB.get('made', 0):,} predictions, {pct(real_acc)} correct"],
        ["Simulated players", "Computer players that follow the rules of the game, in the same format.", f"SIMULATED: {b['events']:,} events, {b['players']} players" if sim else "SIMULATED: only for the monitoring test"]],
        [1.9, 2.9, 1.7])
    s += [para(f"The three targets are the next action (six classes), whether an attack hit the player (yes or no) and the "
               f"place where the player moved during a charge (x and y in pixels). The inputs include the last actions, the time "
               f"between actions, the distance to the nearest enemy, the world, the game mode, the difficulty and the life of "
               f"the player. In this version the data has {D['rows']['actions']:,} actions, {D['rows']['hits']:,} boss attacks "
               f"and {D['rows']['dodges']:,} dodges. As Figure 1 shows, the classes are not balanced: shooting is much more common "
               f"than hiding or jumping. Around {pct(b['hit_rate'], 0)} of the boss attacks hit the player.")]
    s += apa_figure(FIG / "fig_classes.png", "Share of Each Action in the Data", 5.6)
    s += [h("Data Quality, Limits and Risks", 1),
          para("The game writes the data itself, so the values usually have the right type, but we still found some problems "
               "that we explain in Phase 2. The data is relevant because it is exactly what the game models see when they "
               "decide. Our main limit is the amount of real data, because at the beginning only our friends play the game. "
               "Other risks are that players change over time, privacy, and a boss that feels unfair or too difficult."),
          h("Success Criteria", 1),
          para("We wrote the rules in Table 5 before training. In this way we could not change them after we saw the results.")]
    s += apa_table("Success Criteria Defined Before Training", [
        ["Code", "Criterion"],
        ["S1", "Next action: test accuracy is at least 10 points higher than always guessing the most common action."],
        ["S2", "Next action: test accuracy is not lower than the game model."],
        ["S3", "Hit: the Brier score is lower than the game model and lower than a constant guess."],
        ["S4", "Dodge: the average error is lower than the game model."],
        ["S5", "Speed: one prediction takes less than 50 milliseconds."],
        ["S6", "The model is chosen with the validation data, and the test data is used only once, at the end."]],
        [0.8, 5.7])
    s.append(para("Figure A1 in Appendix A shows how this phase looks in our web application."))

    # ------------------------------------------------------------------ phase 2
    s += [h("Phase 2: Data Engineering"),
          para("In this phase we cleaned the raw data and created the inputs for the models. This part took more time than we "
               "expected, because the data comes in small packages from the game and some problems only appear when you look "
               "at all the packages together. Table 6 and Figure 2 show what we found and what we did."),
          h("Problems Found in the Data", 1)]
    s += apa_table("Data Problems Before and After Cleaning", [
        ["Check", "Before", "After", "What we did"],
        ["Total events", f"{b['events']:,}", f"{af['events']:,}", ""],
        ["Duplicated events", b["duplicate_events"], af["duplicate_events"], "We removed them. They come from a package that the game sent two times."],
        ["Distance = 9999", b["ne_sentinel_9999"], af["ne_sentinel_9999"], "9999 means that there was no enemy, so we changed it to a missing value and filled it with the median of the training data."],
        ["Unknown actions", b["invalid_action"], af["invalid_action"], "We remove them if they appear."],
        ["Pauses of 10 minutes or more", b["gap_over_10min"], af["gap_over_10min"], "We keep them, but with a limit of 10 minutes."]],
        [1.7, 0.75, 0.75, 3.3])
    s += apa_figure(FIG / "fig_before_after.png", "Number of Events and Problems Before and After Cleaning", 5.8,
                    note="The horizontal axis uses a logarithmic scale.")
    s += [h("Transformations and New Features", 1),
          para("We applied the transformations in a fixed order. First we removed duplicated events and rows that cannot "
               "come from the game. Then we changed the value 9999 to a missing value, we limited extreme values, and we "
               "sorted the events of each player by time. The encoding of categories (one-hot encoding) and the scaling of "
               "numbers happen inside each model pipeline, so they learn only from the training data."),
          para("We also created new features using what we know about the game. The features prev1 and prev2 are the last "
               "two actions of the player, and they reset after 4 seconds without actions, like in the game. The features "
               "hist_ tell how often the same player did each action before, and the features mk_ are the probabilities of "
               "the online model of that player. We also added a boss flag and, for the attacks, a combination of attack and "
               "situation. All of these features use only the past of the player and never the future."),
          h("Train, Validation and Test Split", 1),
          para(f"We split the data by {'player' if by_player else 'game session'}, so all the data of one "
               f"{'player' if by_player else 'session'} goes to only one part (Table 7). If the same player was in training and "
               "in test, the model could remember that person, and the result would look better than it really is. This was "
               "our main protection against data leakage.")]
    s += apa_table("Size of Each Part of the Data", [
        ["Part", "Groups", "Actions", "Attacks", "Use"],
        ["Training", D["split"]["train"], f"{D['split_rows']['actions']['train']:,}", f"{D['split_rows']['hits']['train']:,}", "Train the models and tune them with cross-validation"],
        ["Validation", D["split"]["val"], f"{D['split_rows']['actions']['val']:,}", f"{D['split_rows']['hits']['val']:,}", "Choose the best model"],
        ["Test", D["split"]["test"], f"{D['split_rows']['actions']['test']:,}", f"{D['split_rows']['hits']['test']:,}", "One final check at the end"]],
        [1.1, 0.8, 1.0, 0.9, 2.7], note=f"Groups are {'players' if by_player else 'game sessions'}.")
    s.append(para("To show the cleaning with real rows and not only with numbers, Appendix B has the same six actions of "
                  "one player twice: first as the game sent them (Table B1) and then as the models receive them (Table B2). "
                  "In those rows you can see a distance of 9999 that becomes a missing value, and a long pause that makes "
                  "the last two actions go back to start. Figure A2 shows the page of this phase in the application."))

    # ------------------------------------------------------------------ phase 3
    s += [h("Phase 3: Model Engineering"),
          h("Algorithms We Tried", 1),
          para("For the next action we tried seven kinds of models (Table 8). We started with very simple ones, because a "
               "complex model is only useful if it is better than a simple guess. We also included the model that the game "
               "uses today, because in the end the real question was whether we can do better than that."),
          ]
    s += apa_table("Models Compared for the Next Action", [
        ["Model", "Why we tried it"],
        ["Most common action", "The simplest guess. A useful model must be better than this."],
        ["Global Markov table", "A simple model of sequences, the same for all players."],
        ["Game model (Markov, decay 0.97)", "The model that the game uses now. It learns each player online and slowly forgets old data."],
        ["Tuned game model", "The same idea, with other values for order and decay chosen with data."],
        ["Logistic regression", "Simple and fast, with good probabilities, and easy to copy into the game code."],
        ["Random forest", "Many decision trees together (Breiman, 2001). It can find patterns that are not linear."],
        ["Gradient boosting", "Usually strong with tables of data, and it accepts missing values."]],
        [2.2, 4.3])
    s += [para("For the attacks we compared a constant hit rate, the game model (one logistic regression for each attack, "
               "trained online), logistic regression with and without the combination of attack and situation, a decision "
               "tree, a random forest and gradient boosting. For the dodge we compared no movement, the global average, the "
               "average of the game, and an average for each player that moves towards the global average when the player "
               "has little data."),
          h("Training Configuration", 1),
          para(f"We used scikit-learn {R['sklearn']} (Pedregosa et al., 2011). To choose the hyperparameters we used a grid "
               "search with group cross-validation, where each group is a player, so the same player never appears in two "
               "folds. Table 9 shows the main settings.")]
    s += apa_table("Training Settings", [
        ["Setting", "Value", "Reason"],
        ["Cross-validation", "GroupKFold, 5 folds", "The same player is never in two folds."],
        ["Logistic regression C", "0.1, 1 and 10", "Controls how strong the regularization is."],
        ["Random forest", "Depth 8 or 16, leaf 5 or 20", "Controls how complex the trees are."],
        ["Gradient boosting", "Learning rate 0.05 or 0.1, depth 3 or 6", "Small steps and short trees reduce overfitting."],
        ["Game model", "Order 1 or 2, decay 0.85 to 1.0", "How much history and memory the model uses."],
        ["Random seed", "42", "The results can be repeated."]],
        [1.7, 2.2, 2.6])
    s += [h("Experiment Log", 1),
          para(f"We saved every experiment in a file (reports/experiments.csv), {len(R['experiments'])} in total. Table 10 "
               "shows the experiments for the next action. The gap between training and validation helped us to see "
               "overfitting early.")]
    rows = [["Model", "Features", "CV (mean ± SD)", "Validation", "Training"]]
    for e in R["experiments"]:
        if e["task"] == "A":
            rows.append([e["model"], e["features"].replace("+", " + "), "-" if e["cv_mean"] is None else f"{pct(e['cv_mean'])} ± {pct(e['cv_std'])}",
                         pct(e["val"]), pct(e["train"]) if e["train"] is not None else "-"])
    s += apa_table("Experiment Log for the Next Action (Accuracy)", rows, [1.85, 1.5, 1.35, 1.0, 0.8],
                   note="CV is the accuracy in the five group cross-validation folds (mean ± standard deviation).")
    s += [h("How Our Knowledge of the Game Helped", 1),
          para("Knowing the game helped us in many decisions. We split the data by player because the model must work for a "
               "new person who opens the game. We used the last two actions and the 4-second rule because fights happen in "
               "short groups of actions. We added the history of each player because every player has a style. We gave each "
               "attack its own effect because, for example, a fake attack beats the shield but a jump beats a wave on the "
               "ground. Finally, we tested the online models step by step, first predicting and then learning, because that "
               "is exactly how they work in the game. Figure A3 shows the page of this phase in the application.")]

    # ------------------------------------------------------------------ phase 4
    lc = A["learning_curve"]
    ga = [e for e in R["experiments"] if e["task"] == "A" and e["gap_train_val"] is not None]
    worst = max(ga, key=lambda e: e["gap_train_val"])
    s += [h("Phase 4: Model Evaluation"),
          h("Metrics and Validation", 1),
          para("For the next action we used accuracy, because the boss acts on the most probable action, so what matters is "
               "how many times the model is right. We also looked at macro-F1, because the classes are not balanced. For the "
               "attacks we used the Brier score (Brier, 1950), which measures how good a probability is. A lower Brier score "
               "is better. We chose it because the boss compares probabilities between attacks, so a correct 30% is more "
               "useful than a good order. For the dodge we used the average error in pixels."),
          para("Our data has groups, because many actions come from the same player. For this reason normal K-fold "
               "cross-validation was not correct, and we used group cross-validation by player. The online models were "
               "tested step by step, which is called prequential evaluation (Dawid, 1984)."),
          h("Comparison of the Models", 1),
          para(f"Figures 3 and 4 compare the models with the validation players. For the next action, the chosen model "
               f"({A['chosen']}) and the game model were close. The difference became clearer in the final test.")]
    s += apa_figure(FIG / "fig_cmp_a.png", "Validation Accuracy for the Next Action", 6.2,
                    note="Blue shows the chosen model and dark gray shows the baselines.")
    s += apa_figure(FIG / "fig_cmp_b.png", "Validation Brier Score for the Hit Probability", 6.2,
                    note="A lower value is better. Blue shows the chosen model.")
    rows = [["Model", "Brier", "ROC-AUC"]] + [[k, f3(v["val"]["brier"]), f3(v["val"]["roc_auc"])] for k, v in B["candidates"].items()]
    s += apa_table("Validation Results for the Hit Probability", rows, [4.3, 1.1, 1.1])
    rows = [["Model", "Average error (px)"]] + [[k, f3(v["val"]["mean_error_px"], 1)] for k, v in C["candidates"].items()]
    s += apa_table("Validation Results for the Dodge", rows, [4.5, 2.0])
    s += [h("Overfitting, Underfitting and Bias", 1),
          para(f"A model overfits when it is much better with the training data than with new data. The biggest gap was "
               f"in the {worst['model'].lower()} ({100 * worst['gap_train_val']:.1f} points), which we expected because "
               f"trees can memorize details. The learning curve in Figure 5 shows the validation accuracy going from "
               f"{pct(lc['val'][0])} with {lc['sizes'][0]:,} rows to {pct(lc['val'][-1])} with {lc['sizes'][-1]:,} rows."),
          para(f"We also saw underfitting. A general model without the history of the player got only "
               f"{pct(A['candidates']['Logistic regression [seq]']['val']['accuracy'])}, very close to the "
               f"{pct(A['candidates']['Majority class']['val']['accuracy'])} of the simplest guess. This surprised us at first, "
               f"but it makes sense: players are very different, so one rule for everybody is not enough. When we added the "
               f"history of each player, the results improved a lot. Figure 6 shows how fast each model learns a new player.")]
    s += apa_figure(FIG / "fig_learning_curve.png", "Learning Curve of the Best General Model", 5.8)
    s += apa_figure(FIG / "fig_experience.png", "Accuracy According to the Number of Actions Already Seen From the Player", 5.8)
    s += [para("To check bias, we compared the results by difficulty and by game mode: " +
               "; ".join(f"{k.split('=')[1].lower()}, {pct(v['accuracy'])}" for k, v in A["test_segments"].items()) +
               ". The model also finds rare actions less often, as the confusion matrix in Figure 7 shows. This is a risk, "
               "because a boss could ignore players who like to hide. Figure 8 shows the calibration of the hit probability.")]
    s += apa_figure(FIG / "fig_confusion.png", "Confusion Matrix of the Chosen Model With the Test Data", 4.6,
                    note="Each row shows the real action and each column shows the predicted action, as a percentage of the row.")
    s += apa_figure(FIG / "fig_calibration.png", "Calibration of the Hit Probability With the Test Data", 4.3,
                    note="Points close to the dotted line mean that the predicted probability matches the real frequency.")
    s += [h("Review With the Domain Expert", 1),
          para(f"Jeison is the creator of the game, so he was our domain expert. On {_us_date(E['date'])} we explained the results to him in simple words and asked four questions. "
               + ("This review used the first run with simulated players. " if "SIMULATED" in E["data_shown"] else "") +
               "Table 13 shows his answers, first in Spanish as he said them and then in English.")]
    s += apa_table("Answers of the Domain Expert", [["Question", "Answer (Spanish)", "Answer (English)"]] +
                   [[q["question"], q["answer_es"], q["answer_en"]] for q in E["answers"]], [2.5, 1.5, 2.5])
    s += [para("His opinion changed some decisions. We kept the rule of choosing the model with the best validation "
               "result, and we kept the limits that make the bosses fair: the boss waits for a dodge at most 60% of the time "
               "and it always shows its warning before attacking. We also decided to check the results for each difficulty "
               "every week, because he was worried that the game could become too hard."),
          h("Chosen Models and Final Test", 1),
          para(f"Following our rule, we chose {A['chosen']} for the next action, {B['chosen']} for the attacks and "
               f"{C['chosen']} for the dodge. We trained them again with the training and validation data together, and then "
               f"we used the test data only once. Table 14 shows the final results, and Table 15 checks the success criteria.")]
    rows = [["Task", "Model", "Test result"]]
    for k, v in A["test"].items():
        rows.append(["Next action", k, f"Accuracy {pct(v['accuracy'])}"])
    for k, v in B["test"].items():
        rows.append(["Hit", k, f"Brier {f3(v['brier'])}"])
    for k, v in C["test"].items():
        rows.append(["Dodge", k, f"{f3(v['mean_error_px'], 1)} px"])
    s += apa_table("Final Results With the Test Data", rows, [1.2, 3.4, 1.9])
    crit = [("S1", at["accuracy"] - maj >= 0.10, f"{pct(at['accuracy'])} vs. {pct(maj)}"),
            ("S2", at["accuracy"] >= A["test"][IG]["accuracy"], f"{pct(at['accuracy'])} vs. {pct(A['test'][IG]['accuracy'])}"),
            ("S3", bt["brier"] < B["test"][IB]["brier"] and bt["brier"] < B["test"]["Constant hit rate"]["brier"],
             f"{f3(bt['brier'])} vs. {f3(B['test'][IB]['brier'])} and {f3(B['test']['Constant hit rate']['brier'])}"),
            ("S4", C["test"][ck]["mean_error_px"] < C["test"]["In-game running mean"]["mean_error_px"],
             f"{f3(C['test'][ck]['mean_error_px'], 1)} px vs. {f3(C['test']['In-game running mean']['mean_error_px'], 1)} px"),
            ("S5", sv["next_ms"] < 50 and sv["hit_ms"] < 50, f"{f3(sv['next_ms'], 1)} ms and {f3(sv['hit_ms'], 1)} ms")]
    s += apa_table("Check of the Success Criteria", [["Code", "Result", "Met"]] + [[c, r, "Yes" if ok else "No"] for c, ok, r in crit],
                   [0.8, 4.6, 1.1])
    s.append(para("All the success criteria were met with this data. Figure A4 shows the page of this phase in the application."
                  if all(ok for _, ok, _ in crit) else "Figure A4 shows the page of this phase in the application."))

    # ------------------------------------------------------------------ phase 5
    s += [h("Phase 5: Model Deployment"),
          h("Saved Model and Web Application", 1),
          para("We saved each chosen model together with its preprocessing in one file, using joblib. We also saved a small "
               "model card with the data source, the date, the features and the test results. The code that makes predictions "
               "creates the features in exactly the same way as the training code, so both parts cannot be different. One "
               f"prediction takes about {f3(sv['next_ms'], 0)} milliseconds."),
          para("The web application is made with Flask and it is published on Render. It has a cover page, one page for each "
               "phase of CRISP-ML, a live demo and a small API. Figure 9 shows the cover page and Figure 10 shows the demo, "
               "where you write the last actions of a player and the application answers.")]
    s += apa_figure(IMG / "app_home.png", "Cover Page of the Flask Application", 6.3, crop=(0, 0, 1280, 1000))
    s += apa_figure(IMG / "app_demo.png", "Live Demo With a Prediction", 6.3, crop=(0, 230, 1280, 1320),
                    note="The bars show the probability of each action. The application also shows what the current game model would say.")
    s += [h("Inputs, Outputs and How to Run It", 1),
          para("The demo and the API receive the last actions of the player (for example shoot, shoot, dodge), the world, "
               "the game mode, the difficulty, the life, the distance to the enemy and whether it is a boss fight. For the "
               "attacks they also receive the name of the attack and its situation. The output is the predicted action with "
               "its probability, the probability of every action, what the game model would say, and the time of the "
               "prediction. For an attack, the output is the probability that it hits."),
          para("To run the project, a person needs Python 3.14 and the libraries in the file requirements.txt. The command "
               "pip install -r requirements.txt installs them, python pipeline.py repeats the whole process in about five "
               "minutes, and python -m flask --app app.app run starts the application. On Render the application runs with "
               "gunicorn, and the configuration is in the file render.yaml."),
          para("For a new user we wrote simple first steps. First, open the cover page and read the phases in order. Then "
               "open the demo, write some actions separated by commas, choose the situation and press Predict. Finally, read "
               "the answer and compare it with the game model. On the page of Phase 5 there is also a form to send an "
               "opinion or report a problem."),
          h("Plan to Connect With the System of Activity 2", 1),
          para("The information system of Activity 2 is our game, its website and the Supabase database, where we keep the "
               "accounts, the saved games and the records. Table 16 shows how the model connects to it.")]
    s += apa_table("Integration Plan With the System of Activity 2", [
        ["Step", "What happens", "Person", "State"],
        ["Data", "The game sends the anonymous play data to Supabase.", "J.P and J.A", "Done"],
        ["Training", "The pipeline reads the data and trains every Monday with GitHub Actions.", "J.A", "Done"],
        ["Serving", "The Flask application on Render gives the predictions.", "J.Q", "Done"],
        ["In the game", "The boss asks the API when a fight starts and every 2 seconds. Without internet it uses the game model.", "J.P", "Planned"],
        ["Feedback", "Accuracy per account, the report form of the game and the form of the application.", "J.Q and J.A", "Running"]],
        [1.1, 3.3, 1.2, 0.9])
    s += [h("Monitoring, Problems and Opinions", 1),
          para("Every prediction is saved with its time, model, inputs, output and speed, and the page /monitoring/live "
               "shows a summary. People can report a problem in the application, in the game or on GitHub. A big problem, "
               "for example when the application does not answer, must be fixed the same day, or we go back to the last "
               "model. This is possible because every model is saved in Git with its model card. Users can also give their "
               "opinion with a score from 1 to 5, and we read the opinions every week. Figure A5 shows the page of this "
               "phase in the application.")]

    # ------------------------------------------------------------------ phase 6
    s += [h("Phase 6: Monitoring and Maintenance"),
          h("What We Watch", 1),
          para("A model is not finished when it is published. Players change, and the game changes too. Table 17 shows the "
               "indicators that we decided to watch and when each one gives an alarm.")]
    s += apa_table("Indicators for Monitoring", [
        ["Indicator", "Source", "Alarm"],
        ["Application working and speed", "/api/health and /monitoring/live", "More than 50 ms or any error"],
        ["Real events per day", "Supabase (REAL)", "No events while people are playing"],
        ["Accuracy of the next action, two-week average", "Real actions and predictions", f"{int(th['acc_drop'] * 100)} points below the first {th['baseline_weeks']} weeks"],
        ["Accuracy of the game model for all accounts", f"Supabase (REAL, {pct(real_acc)} now)", "Same rule"],
        ["Change in the data (PSI)", "New week and first weeks", f"Warning above {th['psi_warn']}, alarm above {th['psi_alert']}"],
        ["Opinions and problems", "Forms", "Any big problem"]],
        [2.4, 2.2, 1.9])
    s += [h("How We Detect Drift", 1),
          para("To see if the data changes we use the Population Stability Index, or PSI (Siddiqi, 2006). It compares the data "
               "of a new week with the data of the first weeks. A small PSI means that the data is stable, and a PSI above "
               "0.20 means a big change. We also measure the accuracy every week with the real actions, because the play log "
               "already contains the true next action."),
          h("Schedule, People and Retraining", 1),
          para("GitHub Actions trains the models again every Monday and creates a new version of this report. Jeison "
               "checks the model and the data every week, Johan checks the website and the reports, and Juan Pablo checks "
               "the game each time we publish a new version. Once a semester, or after a big update, the three of us review "
               f"the whole project. We train the model again when the PSI is above {th['psi_alert']}, when the accuracy falls "
               f"{int(th['acc_drop'] * 100)} points, when an update changes the fights, or when the real data doubles. A new "
               "model replaces the old one only if it passes the same rule of Phase 4."),
          h("First Monitoring Test With Simulated Data", 1),
          para("We wanted to test the plan before having real problems, so we made a test with simulated data. We created "
               "eight weeks of new players, and from week 4 we changed the way they play, with more dodges and fights from "
               "farther away, like after a game update. Then we ran the same pipeline as in production. Figures 11 and 12 and "
               "Table 18 show the results.")]
    s += apa_figure(FIG / "fig_mon_psi.png", "Population Stability Index per Week (Simulated Data)", 6.0)
    s += apa_figure(FIG / "fig_mon_accuracy.png", "Accuracy per Week (Simulated Data)", 6.0)
    rows = [["Week", "Accuracy", "Two-week average", "PSI actions", "PSI distance", "State"]]
    for w in M["weeks"]:
        rows.append([f"{w['week']}", pct(w["accuracy"]), pct(w["accuracy_2w"]), f3(w["psi_target"]), f3(w["psi_ne"]), w["status"]])
    s += apa_table("Results of the Monitoring Test", rows, [0.7, 1.0, 1.3, 1.1, 1.2, 1.2],
                   note="All the weeks use simulated data.")
    s += [para(f"The PSI found the change, and the first alarm appeared in week {M['first_drift_alert']}. "
               + (f"The accuracy also fell in week {M['first_perf_alert']}." if M["first_perf_alert"] else
                  "The accuracy did not fall below the alarm line, so if we had looked only at the accuracy, we would not have "
                  "seen the problem. This is why we watch the data and not only the results.")),
          h("Risks, Quality Controls and Model Updates", 1),
          para("We see six main risks. The first one is having few real players, and players who are friends of the team "
               "and not typical players. The second one is drift after a game update, when the fights change. The third "
               "one is bad data, for example a package that arrives twice or with strange values. The fourth one is a model "
               "that is too strong, so the bosses feel unfair or too difficult, which was the main worry of our domain "
               "expert. The fifth one is privacy, and the last one is the application being down."),
          para("For each risk we have a control. The database only accepts packages with the right format and size, and "
               "it accepts at most 8,000 events per player per hour. The cleaning rules of Phase 2 run every time we train. "
               "The pipeline checks that no player appears in two parts of the data and that the preprocessing learns only "
               "from the training data. The game model always stays inside the game as a backup, so if the API fails the "
               "bosses still work. We look at the results for each difficulty every week. The play log uses random numbers "
               "instead of names, and players can turn it off. Finally, the model, the data report and the model card are "
               "saved together in Git, so we always know which data made each model and we can go back to an older one."),
          para("The update process has five steps. First, GitHub Actions downloads the new real data and trains the "
               "candidates again. Second, the new model is compared with the current one on new validation players, using "
               "the same rule as in Phase 4. Third, if it is better and it passes the success criteria, it is saved with a "
               "commit and Render publishes it. Fourth, if it is not better, we keep the current model and look for the "
               "reason. Fifth, every run is written in the experiment log and in the model card. We have to be honest here: "
               "in this first version the weekly run replaces the model automatically, and Jeison checks it after. "
               "Automating the comparison of the second step is the next thing that we want to do. Figure A6 shows the "
               "page of this phase in the application."),
          ]

    # ------------------------------------------------------------------ debugging notes
    s += [h("Debugging Notes"),
          para("Like in our last project, some of the most important lessons came from bugs. We want to describe two of them."),
          h("The Probabilities in the Wrong Order", 1),
          para("In the first version, our best model got only about 10% accuracy with the test data. That was worse than "
               "guessing, and for some minutes we thought that the whole project was wrong. Then we found the reason: "
               "scikit-learn returns the probabilities in alphabetical order (block, dodge, hide, and so on), but our code "
               "read them in the order of the game (shoot, melee, block, and so on). The model was fine, but we were reading "
               f"the wrong column. We wrote a small function that orders the columns by name, and the accuracy went to "
               f"{pct(at['accuracy'])}. The same mistake was also changing the log loss, so we fixed it there too."),
          h("The Application Failed Only on Render", 1),
          para("On our computers the application worked, but on Render the demo showed the message Internal Server Error. To "
               "find the reason we added an error handler that writes the full error in the log. The error was an ImportError "
               "inside scikit-learn. We had added a thread that loaded the models in the background, and the first visitor "
               "loaded scikit-learn at the same time. The solution was to load the models before the application starts to "
               "receive visitors. Since then, every error is also saved, which is part of our plan for reporting problems.")]

    # ------------------------------------------------------------------ git
    s += [h("Git and GitHub Workflow"),
          para(f"We used Git and GitHub to keep the project organized. The project has its own repository ({REPO_URL}), "
               "separate from the repository of the game, and we work on the master branch. The workflow is simple: we "
               "change the code on the computer, make a commit and push it to master. Each push makes Render update the "
               "application, and every Monday GitHub Actions trains the models again and makes a new commit called Weekly "
               "retrain. The main commits were the first version of everything, the first automatic training on GitHub "
               "(with the same results as on our computers), the change that makes the first prediction faster, the "
               "preparation for the first real data, and the fix of the Render error."),
          para("In the repository of the game we also added the anonymous play log and the database script that creates the "
               "table and its functions in Supabase.")]

    # ------------------------------------------------------------------ discussion / conclusion
    i_close = len(s)
    s += [h("Discussion"),
          para(f"The first lesson for us is that the game model was already good. With new players it got "
               f"{pct(A['test'][IG]['accuracy'])}, while the simplest guess got {pct(maj)}, and with real players it is right "
               f"{pct(real_acc)} of the time. The new model is a little better ({pct(at['accuracy'])}) and it gives better "
               "probabilities, but the difference is not huge. We think this is an honest result and not a bad one."),
          para(f"The second lesson is that each player is different. General models only worked when they knew the history of "
               f"the player. For the attacks the result was the opposite: one model for all players ({B['chosen']}) was better "
               f"than one small model for each player, with a Brier score of {f3(bt['brier'])} against {f3(B['test'][IB]['brier'])}. "
               "Sharing information helps when each player has few attacks."),
          para("The last lessons are about the process. A very small bug made a good model look terrible, so now we always "
               "check the results with a simple baseline. And in the monitoring test the PSI found the change before the "
               "accuracy did, so watching the data is as important as watching the results."),
          h("Conclusion"),
          para(f"This project shows a complete CRISP-ML process, from the problem to a working web application. We added an "
               f"anonymous play log to the game, we cleaned the data and split it by player without data leakage, and we "
               f"compared {len(R['experiments'])} experiments. The chosen model for the next action reached "
               f"{pct(at['accuracy'])} with the test data, against {pct(A['test'][IG]['accuracy'])} for the game model. The "
               "models are published in a Flask application with a demo and an API, and the system trains again every week "
               "and watches the data for drift. For us, the most valuable part was learning to check our own work: a fixed "
               "rule, a simple baseline and data that the model never saw."),
          h("Future Work", 1),
          para("We want to train the models with real data when there are enough players and repeat the review with the "
               "domain expert. We also want to connect the bosses of the game to the API, try sequence models such as a small "
               "recurrent neural network when we have more data, and ask players if the bosses feel fair.")]

    # ------------------------------------------------------------------ references (APA 7, alphabetical, hanging indent)
    s += [PageBreak(), h("References")]
    for ref in [
        "Breiman, L. (2001). Random forests. <i>Machine Learning, 45</i>(1), 5–32. https://doi.org/10.1023/A:1010933404324",
        "Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. <i>Monthly Weather Review, 78</i>(1), 1–3. https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2",
        "Dawid, A. P. (1984). Present position and potential developments: Some personal views. Statistical theory: The prequential approach. <i>Journal of the Royal Statistical Society: Series A, 147</i>(2), 278–292. https://doi.org/10.2307/2981683",
        "Pallets Projects. (2024). <i>Flask documentation</i>. https://flask.palletsprojects.com/",
        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
        "Siddiqi, N. (2006). <i>Credit risk scorecards: Developing and implementing intelligent credit scoring</i>. Wiley.",
        "Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., &amp; Müller, K.-R. (2021). Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology. <i>Machine Learning and Knowledge Extraction, 3</i>(2), 392–413. https://doi.org/10.3390/make3020020",
        "Yannakakis, G. N., &amp; Togelius, J. (2018). <i>Artificial intelligence and games</i>. Springer. https://doi.org/10.1007/978-3-319-63519-4",
    ]:
        s.append(Paragraph(ref, REF))

    # ------------------------------------------------------------------ appendices (APA: after the references)
    i_app = len(s)
    if part:
        PART["shift"] = 0
    intro = (f"These screenshots come from the published application ({APP_URL}). Each phase has its own page, and "
             "the menu at the top lets the user move between them.")
    if sim:
        intro += " The yellow message at the top says that the model results come from simulated players."
    s += [PageBreak(), Paragraph("<b>Appendix A</b>", CEN), h("Screenshots of the Six Phases in the Application"), para(intro)]
    for i, (name, title, crop) in enumerate([
            ("app_phase1", "Page of Phase 1: Business and Data Understanding", (0, 60, 1280, 1250)),
            ("app_phase2", "Page of Phase 2: Data Engineering", (0, 60, 1280, 1320)),
            ("app_phase3", "Page of Phase 3: Model Engineering", (0, 60, 1280, 1250)),
            ("app_phase4", "Page of Phase 4: Model Evaluation", (0, 60, 1280, 1700)),
            ("app_phase5", "Page of Phase 5: Model Deployment", (0, 60, 1280, 1300)),
            ("app_phase6", "Page of Phase 6: Monitoring and Maintenance", (0, 60, 1280, 1250))], start=1):
        s += apa_figure(IMG / f"{name}.png", title, 6.4, crop=crop, label=f"A{i}")
    ev = json.loads((ROOT / "data" / "processed" / "evidence.json").read_text(encoding="utf-8"))
    txt = ("Table B1 shows six actions of one player exactly as the game sent them to the database. Table B2 shows "
           "the same six actions after Phase 2, as the models receive them. We chose this player because one of the "
           "actions has the value 9999 in the distance, so the cleaning is easy to see. Some columns are not shown "
           "because they do not fit on the page.")
    if sim:
        txt += " These rows come from the simulated data."
    s += [PageBreak(), Paragraph("<b>Appendix B</b>", CEN), h("The Same Data Before and After Processing"), para(txt)]

    def fmt(v):
        if v is None:
            return "missing"
        return f"{v:g}" if isinstance(v, float) else str(v)

    s += apa_table("Raw Events as Sent by the Game",
                   [["Player", "Time (ms)", "Action", "Gap (ms)", "Distance", "Life", "World", "Mode", "Diff."]] +
                   [[r["pid"].replace("\u2026", "..."), fmt(r["t"]), r["a"], fmt(r["gap"]), fmt(r["ne"]), fmt(r["hp"]),
                     fmt(r["w"]), r["m"], fmt(r["d"])] for r in ev["raw"]],
                   [0.9, 0.85, 0.7, 0.8, 0.95, 0.5, 0.6, 0.65, 0.55],
                   note="Player is a random id (shortened). Distance 9999 means that the game did not find an enemy.",
                   label="B1")
    s += apa_table("The Same Events After Processing",
                   [["Order", "prev2", "prev1", "Start", "Distance", "Boss", "hist shoot", "mk shoot", "Target"]] +
                   [[fmt(r["seq"]), r["prev2"], r["prev1"], fmt(r["is_start"]), fmt(r["ne"]), fmt(r["boss"]),
                     fmt(r["hist_shoot"]), fmt(r["mk_shoot"]), r["target"]] for r in ev["processed"]],
                   [0.65, 0.75, 0.75, 0.6, 0.85, 0.55, 0.75, 0.75, 0.65],
                   note="A missing distance is filled with the median of the training data inside the model. After a pause "
                        "of more than 4 seconds, prev1 and prev2 go back to start. The order numbers skip some values "
                        "because the player also had other events, such as boss attacks, between these actions.",
                   label="B2")

    if part:
        return {"body": s[i_body:i_close], "appendix": s[i_app:]}
    out = REPORTS / FILE_NAME
    Doc(str(out), title="R3A2 Traces in data", author=", ".join(AUTHORS)).multiBuild(s)
    print("PDF ->", out)


if __name__ == "__main__":
    main()
