"""Final technical report of the whole project (Activity 3): "R3A4 From Code To The World" (PDF, APA 7).

    python -m src.umbral_ml.report_final

It consolidates the three activities in one document:
  * the mandatory PGC sections (src/umbral_ml/pgc.py, the same text as the homepage),
  * Part I  = the CRISP-ML process of Activity 1 (report.py, as a part),
  * Part II = the intelligent information system of Activity 2 (report_system.py, as a part),
  * results and objectives, contributions of each member, limitations, conclusions, future work, references,
  * appendices A-B (Activity 1) and C-D (technical and user manuals).
Tables and figures are numbered once for the whole document (report.PART moves the numbers written in the text).
Optional: REPORT_ACCOUNTS="user:Role:password;..." adds the teacher accounts (private copy only, never commit it),
REPORT_OUT=<path> writes the PDF somewhere else.
"""
from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, Spacer
from reportlab.platypus.tableofcontents import TableOfContents

from . import pgc
from . import report as R0
from . import report_system as R2
from .config import REPORTS
from .report import (AUTHORS, CEN, COURSE, DATE, DOUBLE, INSTITUTION, REG, SIZE, TEACHER, Doc, apa_table, h, para,
                     seriation)

REPORT_NAME = "From Code To The World"
FILE_NAME = f"R3A4 {REPORT_NAME}.pdf"
IMG = R2.IMG


def main() -> None:
    R0.NUM.update(table=0, figure=0)
    R0.PART.update(t_off=0, f_off=0, letters={}, shift=0)
    C = pgc.context()
    n = C["n"]
    s: list = []

    # ---------------------------------------------------------------- title page and contents
    if (IMG / "logo_udec.png").exists():
        from reportlab.platypus import Image  # noqa: PLC0415
        s += [Spacer(1, 0.2 * inch), Image(str(IMG / "logo_udec.png"), width=1.1 * inch, height=1.1 * inch * 150 / 99)]
    s += [Spacer(1, 0.5 * inch), Paragraph(f"<b>{REPORT_NAME}</b>", CEN), Paragraph(f"<b>{C['title']}</b>", CEN),
          Spacer(1, DOUBLE)]
    s += [Paragraph(a, CEN) for a in AUTHORS]
    s += [Paragraph(INSTITUTION, CEN), Paragraph(COURSE, CEN), Paragraph(TEACHER, CEN), Paragraph(DATE, CEN),
          Spacer(1, DOUBLE),
          Paragraph("Final technical report and PGC documentation of the three activities: the CRISP-ML process, "
                    "the intelligent web information system and its validation", CEN),
          Spacer(1, DOUBLE),
          Paragraph(f'<link href="{R0.REPO_URL}"><u>GitHub Repository</u></link>', CEN),
          Paragraph(f'<link href="{R0.APP_URL}"><u>Published Project and PGC Documentation</u></link>', CEN)]
    if C["video"]:
        s.append(Paragraph(f'<link href="{C["video"]}"><u>Presentation Video</u></link>', CEN))
    s.append(PageBreak())
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle(f"t{i}", fontName=REG, fontSize=SIZE, leading=19, leftIndent=0.35 * inch * i,
                                      textColor=R0.BLACK) for i in range(3)]
    toc.dotsMinLevel = 0
    s += [Paragraph("<b>Table of Contents</b>", CEN), toc, PageBreak()]

    # ---------------------------------------------------------------- PGC sections
    s += [Paragraph(f"<b>{C['title']}</b>", CEN), h("Introduction"),
          para("This report consolidates the complete project that we developed in three activities. In Activity 1 we "
               "applied the six phases of CRISP-ML to the machine learning of our video game, UMBRAL. In Activity 2 we "
               "built an intelligent web information system that manages the players and accounts of the game and uses "
               "the chosen model inside a real decision. In Activity 3 we documented the whole project with the PGC "
               "structure, validated it again and presented it. The same text of the PGC sections is published on the "
               f"homepage of the project ({R0.APP_URL}), and the repository, the published system and this report "
               "represent the same final version."),
          h("Project Title"), para(f"<b>{C['title']}</b>."),
          para("The title says the three parts of the project: player modeling with machine learning, its purpose (boss "
               "fights that adapt but stay fair) and the information system where the model is used."),
          h("Problem Statement and Justification")]
    for head, paras in C["problem"]:
        s += [h(head, 1)] + [para(t) for t in paras]
    s += [h("General Objective"), para(C["general"]),
          h("Specific Objectives"),
          para("Table 1 lists the specific objectives with the indicator that verifies each one. The results are "
               "checked in the section Results and Achievement of the Objectives.")]
    s += apa_table("Specific Objectives and Their Indicators",
                   [["Code", "Area", "Objective", "Indicator"]] +
                   [[o["code"], o["area"], o["objective"], o["indicator"]] for o in C["specific"]],
                   [0.6, 1.2, 2.6, 2.1])
    s += [h("Reference Framework")]
    for head, paras in C["framework"]:
        s += [h(head, 1)] + [para(t) for t in paras]
    s += [h("Methodology and Development Schedule"),
          para("We followed CRISP-ML (Studer et al., 2021). Each phase below says what we really did; Parts I and II "
               "give the details and the evidence.")]
    for head, t in C["method"]:
        s += [h(head, 1), para(t)]
    s += [h("Development Schedule", 1),
          para("Table 2 shows the schedule with the person responsible, the planned and actual dates and the "
               "deliverable of each activity. The actual dates come from the Git history of the game and of the "
               "project repository.")]
    s += apa_table("Development Schedule", [["Activity", "Responsible", "Planned", "Actual", "Deliverable"]] +
                   [list(r) for r in C["schedule"]], [2.15, 1.0, 1.15, 1.05, 1.15])

    # ---------------------------------------------------------------- Part I and Part II
    s += [PageBreak(), h("Part I. The CRISP-ML Process (Activity 1)"),
          para("This part presents the six phases of CRISP-ML applied to the machine learning of the game: the data, "
               "the comparison of the models, the evaluation and the selection, the deployment and the monitoring plan.")]
    R0.PART.update(t_off=R0.NUM["table"] - 2, f_off=R0.NUM["figure"], letters={}, shift=1)   # Activity 1 had 2 concept tables
    p1 = R0.main(part=True)
    s += p1["body"]
    R0.PART.update(shift=0)
    s += [PageBreak(), h("Part II. The Intelligent Information System (Activity 2)"),
          para("This part presents the web information system: its processes and rules, the database, how the model "
               "works inside the play-session process, the tests, the deployment, the operational review and the "
               "monitoring. Its manuals are Appendices C and D.")]
    R0.PART.update(t_off=R0.NUM["table"] - 1, f_off=R0.NUM["figure"], letters={"A": "C", "B": "D"}, shift=1)
    p2 = R2.main(part=True)
    s += p2["body"]
    R0.PART.update(t_off=0, f_off=0, letters={}, shift=0)

    # ---------------------------------------------------------------- results, contributions, closing
    T, S, V = p2["tests"], p2["snap"], p2["review"]
    s += [PageBreak(), h("Results and Achievement of the Objectives"),
          para(f"Table {R0.NUM['table'] + 1} summarizes the main results of the three activities, and Table "
               f"{R0.NUM['table'] + 2} checks each specific objective. We separate what we demonstrated from what we "
               "expect: the model results come from simulated players that follow the rules of the game, because the "
               "real anonymous log has not reached its minimum yet, while the system results come from the published "
               "system.")]
    rows = [["Result", "Value", "Evidence"],
            ["Next-action model (test players)", f"{pgc.pct(n['acc'])} accuracy; game model {pgc.pct(n['acc_game'])}; "
                                                 f"majority {pgc.pct(n['acc_major'])}", "Part I, Phase 4 (SIMULATED)"],
            ["Hit model (test players)", f"Brier {n['brier']:.3f}; game model {n['brier_game']:.3f}",
             "Part I, Phase 4 (SIMULATED)"],
            ["Game model with real players", f"{pgc.pct(n['real_acc'])} of {n['real_made']:,} predictions "
                                             f"({n['real_players']} players)", "Supabase (REAL)"],
            ["Experiments recorded", str(n["n_exp"]), "reports/experiments.csv"],
            ["Automatic tests of the system", T["summary"], "tests/test_system.py, GitHub Actions"]]
    if V and V.get("environment") == "DEPLOYED":
        rows.append(["Operational review of the published system", f"{V['passed']} of {V['total']} checks",
                     "Part II, Phase 5"])
    if S and S.get("acc") is not None:
        rows.append(["Model inside the published system", f"{pgc.pct(S['acc'])} of {S['checked']} next actions; "
                                                          f"{S['inferences']} inferences, p95 {S['p95_ms']:.0f} ms",
                     "Part II, Phase 4"])
    s += apa_table("Main Results of the Project", rows, [2.2, 2.6, 1.7])
    s += apa_table("Achievement of the Specific Objectives", [["Code", "Result obtained", "Status"]] +
                   [[o["code"], o["result"], o["status"]] for o in C["specific"]], [0.7, 4.7, 1.1])
    s += [para("Six of the seven objectives were met. The monitoring objective is partly met: the monitoring works, but "
               "the drift indicator needs more real actions, and the models are still trained with simulated players.")]

    s += [h("Contributions of Each Member"),
          para(f"Table {R0.NUM['table'] + 1} shows what each member did, according to the Git history of the game and "
               "of the project, and the credits of the game.")]
    s += apa_table("Contributions of Each Member", [["Member", "Contribution"]] +
                   [[f"{ini} · {name}", t] for ini, name, t in C["contrib"]], [1.6, 4.9])

    s += [h("Limitations")]
    s += seriation(["The models were trained and tested with simulated players. The real anonymous log started in "
                    f"November 2026 and has {n['real']['actions']:,} actions, below the minimum of "
                    f"{n['real']['min_actions']:,} that we fixed in Phase 1; the pipeline changes to real data by itself.",
                    "The published system has few users and sessions, so its accuracy and drift numbers are not stable "
                    "yet.",
                    "The play sessions are uploaded by hand; the game does not send them to the system yet.",
                    "The system has no page to change or recover a password, and the weekly retraining replaces the "
                    "model before a person checks it.",
                    "We did not measure yet whether the difficulty changes make the fights feel fairer for real players."])
    s += [h("Conclusions"),
          para("We demonstrated a complete applied machine learning project. We turned the play traces of our game into "
               "an anonymous dataset, compared "
               f"{n['n_exp']} experiments with a split by player, and chose models that are better than the models that the "
               f"game already used ({pgc.pct(n['acc'])} against {pgc.pct(n['acc_game'])} for the next action). We then "
               "put the model inside a real process: an information system with accounts, roles and five business "
               "processes, where a play session is analyzed, the model proposes a fairer difficulty, a person decides, "
               "and every inference is stored with its model version."),
          para("The most important lessons were about checking our own work: a fixed rule before training, a simple "
               "baseline, players that the model never saw, the same features in training and in the system, and "
               "monitoring that watches the data and not only the results. What we demonstrated is that the process "
               "and the system work end to end; what we still expect, and must measure with real players, is that the "
               "fights really feel fairer."),
          h("Future Work")]
    s += seriation(["Train and evaluate the models again with the real log when it reaches its minimum, and repeat the "
                    "review with the domain expert.",
                    "Let the game upload the play sessions by itself and use the recommendations for real players.",
                    "Measure with players whether the difficulty changes make the fights feel fair (a short survey after "
                    "each session).",
                    "Compare the new model with the current one automatically before replacing it, and add password "
                    "change and recovery to the system."])

    # ---------------------------------------------------------------- references and appendices
    s += [PageBreak(), h("Bibliographic References")]
    s += [Paragraph(r, R0.REF) for r in C["refs"]]
    s += p1["appendix"]
    R0.PART.update(letters={"A": "C", "B": "D"})
    s += p2["appendix"]
    R0.PART.update(letters={})

    out = Path(os.environ["REPORT_OUT"]) if os.environ.get("REPORT_OUT") else REPORTS / FILE_NAME
    Doc(str(out), title=f"R3A4 {REPORT_NAME}: {C['title']}", author=", ".join(AUTHORS)).multiBuild(s)
    print("report written:", out, "| tables", R0.NUM["table"], "| figures", R0.NUM["figure"])


if __name__ == "__main__":
    main()
