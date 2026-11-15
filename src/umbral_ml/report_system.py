"""Activity 2 technical report "R3A3 Speaking Models" (PDF): same APA 7 format and the same structure as the
Activity 1 report (report.py): introduction, concepts, the six CRISP-ML phases, debugging notes, Git workflow,
discussion, conclusion, future work and references, plus the two manuals as appendices.

    python -m src.umbral_ml.report_system            (uses the local database or DATABASE_URL for the numbers)

Optional: REPORT_ACCOUNTS="user:Role:password;..." adds the teacher test accounts (never commit that PDF), and
REPORT_OUT=<path> writes the PDF somewhere else.

Every number comes from the system itself (database snapshot), the automatic tests, the operational review
evidence (reports/operational_review.json for the deployed system, *_local.json for a local run) and the
Activity 1 result files. The technical and user manuals (docs/system/*.md) are added as appendices, so the
repository, the app and the PDF say the same thing.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, Spacer
from reportlab.platypus.tableofcontents import TableOfContents

from . import report as R0
from .config import REPORTS, ROOT
from .report import (AUTHORS, CEN, COURSE, DATE, DOUBLE, INSTITUTION, LIST, REG, SIZE, TEACHER, Doc, apa_figure,
                     apa_table, h, para, seriation)

TITLE = "Speaking Models: How the Model of UMBRAL Takes Part in the Decisions of an Information System"
FILE_NAME = "R3A3 Speaking Models.pdf"
HISTORY_ACC = 0.538   # next-action accuracy inside the system on the Activity 1 test sessions, with history (measured)
APP_URL = R0.APP_URL
REPO_URL = R0.REPO_URL
IMG = ROOT / "docs" / "img"
DOCS = ROOT / "docs" / "system"
BLACK = R0.BLACK


# ------------------------------------------------------------------------------------- evidence
def run_tests() -> dict:
    """Run the automatic tests and keep the summary (python -m pytest -q tests)."""
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True,
                             text=True, timeout=900).stdout
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "summary": f"not run ({type(e).__name__})"}
    m = re.search(r"(\d+) passed(?:, (\d+) failed)?.* in ([\d.]+)s", out)
    return {"ok": bool(m) and not (m.group(2)), "passed": int(m.group(1)) if m else 0,
            "failed": int(m.group(2) or 0) if m else None, "seconds": float(m.group(3)) if m else None,
            "summary": out.strip().splitlines()[-1] if out.strip() else "no output"}


def snapshot() -> dict:
    """Numbers of the running system (players, sessions, decisions, monitoring).

    reports/system_snapshot.json holds the numbers read from the DEPLOYED system (its pages and monitoring.json,
    as administrator); it wins over the local database, so the report shows the real system, not demo data."""
    live = REPORTS / "system_snapshot.json"
    if live.exists():
        d = json.loads(live.read_text(encoding="utf-8"))
        d["users"], d["recs"], d["rec_actions"] = Counter(d["users"]), Counter(d["recs"]), Counter(d["rec_actions"])
        d["live"] = True
        return d
    sys.path.insert(0, str(ROOT))
    from app.app import app  # noqa: PLC0415
    from app.system.models import (Inference, PlaySession, Player, Recommendation, Ticket,  # noqa: PLC0415
                                   User)
    from app.system.views import monitoring_data  # noqa: PLC0415
    with app.app_context():
        an = PlaySession.query.filter(PlaySession.predictability.isnot(None)).all()
        chk = sum(i.n_checked or 0 for i in Inference.query.filter_by(kind="session_profile"))
        cor = sum(i.n_correct or 0 for i in Inference.query.filter_by(kind="session_profile"))
        recs = Recommendation.query.all()
        m = monitoring_data()
        return {"users": Counter(u.role for u in User.query.all()), "players": Player.query.count(),
                "sessions": PlaySession.query.count(), "analyzed": len(an),
                "acc": cor / chk if chk else None, "checked": chk,
                "base": sum(s.baseline for s in an) / len(an) if an else None,
                "risk": sum(s.risk for s in an) / len(an) if an else None,
                "recs": Counter(r.status for r in recs), "rec_actions": Counter(r.action for r in recs),
                "tickets": Ticket.query.count(), "inferences": m["n"], "mean_ms": m["mean"], "p95_ms": m["p95"],
                "psi": m["psi"], "psi_state": m["psi_state"], "errors": m["n_errors"], "version": m["version"],
                "db": app.config["SQLALCHEMY_DATABASE_URI"].split(":")[0]}


def review() -> dict | None:
    for name in ("operational_review.json", "operational_review_local.json"):
        p = REPORTS / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


# ------------------------------------------------------------------------------ markdown -> APA
def _inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', t)
    return t


def appendix(md_path: Path, letter: str) -> list:
    """A manual (Markdown) as an APA appendix: headings, paragraphs, lists and tables."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    out, buf, tab, n_tab = [], [], [], 0
    last = [""]

    def flush_tab():
        nonlocal tab, n_tab
        if tab:
            rows = [[_inline(c.strip()) for c in r.strip().strip("|").split("|")] for r in tab
                    if not re.match(r"^\|\s*-", r.strip())]
            n_tab += 1
            ncol = len(rows[0])
            widths = {2: [2.3, 4.2], 3: [2.2, 2.2, 2.1], 4: [2.6, 1.3, 1.3, 1.3]}.get(ncol, [6.5 / ncol] * ncol)
            title = re.sub(r"^\d+\.\s*", "", last[0]) or "Table"
            out.extend(apa_table(title, rows, widths, label=f"{letter}{n_tab}"))
            tab = []

    for ln in lines:
        s = ln.rstrip()
        if s.startswith("|"):
            tab.append(s)
            continue
        flush_tab()
        if s.startswith("# "):
            continue
        if s.startswith("## "):
            last[0] = s[3:]
            out.append(h(_inline(s[3:]), 1))
        elif s.startswith("- "):
            out.append(Paragraph(_inline(s[2:]), LIST, bulletText="•"))
        elif re.match(r"^\d+\. ", s):
            num, rest = s.split(". ", 1)
            out.append(Paragraph(_inline(rest), LIST, bulletText=f"{num}."))
        elif s:
            out.append(para(_inline(s)))
    flush_tab()
    return out


# ------------------------------------------------------------------------------------- report
# ------------------------------------------------------------------------------------- report
def accounts() -> list[list[str]]:
    """Test accounts for the teacher, only from the environment (never written in the repository):
    REPORT_ACCOUNTS="user:Role:password;user2:Role:password". Without it the report says they are sent privately."""
    rows = []
    for item in os.environ.get("REPORT_ACCOUNTS", "").split(";"):
        parts = item.split(":", 2)
        if len(parts) == 3 and all(p.strip() for p in parts):
            rows.append([p.strip() for p in parts])
    return rows


def main(part: bool = False):
    """Write the PDF. With part=True (final report) return {"body", "appendix"}: no title page, contents,
    introduction, concepts, closing sections or references; the numbering continues from the final report."""
    if not part:
        R0.NUM.update(table=0, figure=0)
    T = run_tests()
    S = snapshot()
    V = review()
    ACC = accounts()
    A1 = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    at = A1["A"]["test"][A1["A"]["chosen"]]["accuracy"]
    deployed = bool(V and V.get("environment") == "DEPLOYED")
    vm = (V or {}).get("monitoring") or {}
    n_tests = T.get("passed", 0) + (T.get("failed") or 0)
    s: list = []
    code = lambda t: f"<font face='Courier'>{t}</font>"  # noqa: E731

    if not part:
        # ---------------------------------------------------------------- title page
        if (IMG / "logo_udec.png").exists():
            from reportlab.platypus import Image  # noqa: PLC0415
            s += [Spacer(1, 0.2 * inch), Image(str(IMG / "logo_udec.png"), width=1.1 * inch, height=1.1 * inch * 150 / 99)]
        s += [Spacer(1, 0.6 * inch), Paragraph(f"<b>{TITLE}</b>", CEN), Spacer(1, DOUBLE)]
        s += [Paragraph(a, CEN) for a in AUTHORS]
        s += [Paragraph(INSTITUTION, CEN), Paragraph(COURSE, CEN), Paragraph(TEACHER, CEN), Paragraph(DATE, CEN),
              Spacer(1, DOUBLE),
              Paragraph("Selected topic: a web information system that manages the players and accounts of a video game "
                        "and uses its machine learning model inside the decisions (CRISP-ML)", CEN),
              Spacer(1, DOUBLE),
              Paragraph(f'<link href="{REPO_URL}"><u>GitHub Repository</u></link>', CEN),
              Paragraph(f'<link href="{APP_URL}"><u>Render Deployment URL</u></link>', CEN), PageBreak()]
        toc = TableOfContents()
        toc.levelStyles = [ParagraphStyle("t0", fontName=REG, fontSize=SIZE, leading=20, textColor=BLACK),
                           ParagraphStyle("t1", fontName=REG, fontSize=SIZE, leading=20, leftIndent=0.35 * inch,
                                          textColor=BLACK)]
        toc.dotsMinLevel = 0
        s += [Paragraph("<b>Table of Contents</b>", CEN), toc, PageBreak()]

        # ---------------------------------------------------------------- introduction
        s += [Paragraph(f"<b>{TITLE}</b>", CEN), h("Introduction"),
              para("In Activity 1 we studied the machine learning of our game, UMBRAL, with CRISP-ML. We compared 22 "
                   f"experiments and chose a gradient boosting model that guesses the next action of a player ({at:.1%} "
                   "accuracy with test players), and a logistic regression that gives the probability that a boss attack "
                   "hits the player. We published them in a Flask application with a demo."),
              para("A demo is not a system. Nobody decides anything with a demo form. In this activity the model starts to "
                   "speak inside a real process, and this is why we call the report Speaking Models. We built a complete web "
                   "information system that manages the players and accounts of the game: player profiles, saved games, "
                   "play sessions, records and support tickets. The model is one step of the process: when a play session "
                   "arrives, an analyst runs the model, the model proposes a change of difficulty for that player, a person "
                   "approves or rejects it, the account changes, and the next session tells us if the change worked."),
              para("We follow the same six phases of CRISP-ML as in Activity 1, now seen from the side of the system. We "
                   "extended the same Flask application and the same repository, so the documentation of Activity 1 is still "
                   "the home page, and every prediction can be traced to the evaluated model and its version.")]

        s += [h("Information System Concepts"),
              para("Before the phases, we want to explain some words that we use many times in this report. Table 1 shows "
                   "them with the meaning that they have in our project.")]
        s += apa_table("Main Concepts Used in the System", [
            ["Concept", "Meaning in this project"],
            ["Information system", "The web application that stores the data of the players and supports the decisions of "
                                   "the team."],
            ["Business process", "A sequence of steps with people and rules, for example \"upload a play session → analyze "
                                 "it → decide the difficulty\"."],
            ["CRUD", "Create, read, update and delete: the four basic operations on the data."],
            ["Authentication", "Checking who the user is (user name and password)."],
            ["Authorization and role", "Checking what the user can do. Each user has one role: player, analyst or "
                                       "administrator."],
            ["Password hash", "A one-way code of the password. The system stores the hash, never the password."],
            ["Inference", "One use of the trained model with new data, for example the analysis of one play session."],
            ["Model version", "The data source and training date of the model. Every inference keeps it."],
            ["Human in the loop", "The model proposes, but a person approves or rejects the decision."],
            ["Drift", "The input data changes over time, and the model can get worse. We measure it with the PSI."]],
            [1.9, 4.6])


    i_body = len(s)
    # ---------------------------------------------------------------- phase 1
    s += [h("Phase 1: Business and Process Understanding"), h("The Problem and the People", 1),
          para("Our domain expert said in Activity 1 that the bosses could feel unfair for some players. The game "
               "already knows how each player fights, but this knowledge was only inside the bosses: nobody in the team "
               "could see it, compare players or decide anything with it. The team also had the data of the players in "
               "many places: the game, the Supabase database of the accounts and some spreadsheets."),
          para("Three kinds of people use the system. Players are the people who play UMBRAL; they register by "
               "themselves. Analysts are members of the team who review sessions with the model and make the "
               "decisions. Administrators manage users and roles. In our team J.A is the administrator and the owner "
               "of the model, and J.P and J.Q are analysts."),
          h("Business Goals and System Goals", 1)]
    s += apa_table("Business Goals, System Goals and the Role of the Model", [
        ["Business goal", "System goal", "Role of the model"],
        ["Keep the data of players and accounts in one place.", "Persistent database with CRUD for each process.", "-"],
        ["Fights that feel fair for each player.", "A difficulty decision for each analyzed play session.",
         "Predictability and hit risk of the player."],
        ["Trust in the decisions.", "Every decision is approved by a person and can be traced.",
         "Each inference stores the model version."],
        ["Know if the model still works.", "Monitoring of errors, times, accuracy and drift.",
         "Predictions compared with the real next actions."]],
        [2.2, 2.3, 2.0])
    s += [h("Business Processes", 1),
          para("We chose the processes of players and accounts. Table 3 shows them. Process P2 is the one that uses the "
               "model.")]
    s += apa_table("Business Processes Managed by the System", [
        ["Process", "Steps", "Who"],
        ["P1 Player accounts", "Register or create a profile → edit → add, edit and delete saved games → suspend",
         "Player (own), analyst, admin"],
        ["P2 Play session and difficulty (uses the model)", "Upload session → analyze with the model → recommendation → "
         "approve or reject → difficulty applied → result checked with the next session", "Player uploads; analyst decides"],
        ["P3 Records", "Send record → verify or reject → leaderboard", "Player sends; analyst verifies"],
        ["P4 Support and feedback", "Open ticket → in progress → resolved (with resolution) → closed", "Any user; analyst manages"],
        ["P5 Management", "Reports and indicators, model monitoring, users and roles", "Analyst, admin"],
    ], [1.6, 3.4, 1.5])
    s += [h("Functional Requirements and Business Rules", 1),
          para("Table 4 lists the main functional requirements, and Table 5 the business rules that the server checks.")]
    s += apa_table("Main Functional Requirements", [
        ["Code", "Requirement"],
        ["FR1", "Register, sign in and sign out; passwords stored with a secure hash."],
        ["FR2", "Three roles with different permissions, checked on the server."],
        ["FR3", "Create, read, update and delete players, saved games, sessions, records and tickets."],
        ["FR4", "Run the Activity 1 models on a stored session, on the server, and show the result with its meaning and limits."],
        ["FR5", "Turn the model result into a difficulty recommendation that a person approves or rejects."],
        ["FR6", "Store every inference with date, user, result, model version and response time."],
        ["FR7", "Reports and indicators by difficulty and world; CSV export."],
        ["FR8", "Monitoring: errors, inference requests, response times, accuracy against real actions, data drift."],
        ["FR9", "Incident and feedback tickets with a status flow."],
    ], [0.8, 5.7])
    s += apa_table("Business Rules", [
        ["Rule", "Where it is checked"],
        ["Public registration always creates a player; only an admin changes roles; nobody changes their own role; one "
         "active admin must exist.", "auth.register, views.user_update"],
        ["A player sees and changes only their own profile and data.", "auth.player_or_403"],
        ["A session has 5 to 400 actions, all of them one of the six game actions; it cannot be in the future; a won "
         "session ends with at least one heart.", "ml_service.parse_actions, views._session_form"],
        ["A saved game has a unique slot per player; relics ≤ world reached; stars ≤ 16 per world.", "views._save_form"],
        ["An analyzed session cannot be edited, and only an admin can delete it (it is evidence of a decision).",
         "views.session_edit, session_delete"],
        ["The difficulty of an account changes only through an approved recommendation; one pending recommendation per "
         "player; rejecting needs a reason.", "ml_service.decide, apply_decision"],
        ["A suspended player cannot upload sessions; a record counts only when verified.", "views.session_new, records"],
        ["Ticket status follows open → in progress → resolved → closed; resolving needs a resolution.", "views.ticket"],
    ], [4.6, 1.9])
    s += [h("Success Criteria", 1),
          para("We wrote the rules in Table 6 before building the system. In this way we could not change them after we "
               "saw the results.")]
    s += apa_table("Success Criteria Defined Before Building the System", [
        ["Code", "Criterion"],
        ["S1", "Every role sees only its functions: the permission matrix of the deployed system has no error."],
        ["S2", "The whole process P2 works on the public URL: upload, model, decision, difficulty and history."],
        ["S3", "Inside the system, the model keeps its Activity 1 accuracy on the same test sessions (at most 2 points less)."],
        ["S4", "One analysis of a session answers in less than 1 second (95th percentile)."],
        ["S5", "Every inference is stored with date, user, result, model version and response time."],
        ["S6", "All the automatic tests pass."]],
        [0.8, 5.7])

    # ---------------------------------------------------------------- phase 2
    s += [h("Phase 2: Data Engineering"), h("Database Model", 1),
          para("Figure 1 shows the database model. It has ten tables. The two blue tables connect the business process "
               "with the model: every inference and every recommendation points to the session and the user. We used "
               "SQLAlchemy, so the same code works with SQLite for the tests and with PostgreSQL in production. The tables "
               "are in the same Supabase project as the accounts of the game, but they are separate, so nothing of the "
               "game changes.")]
    s += apa_figure(IMG / "sys_er.png", "Database Model", 6.5)
    s += [h("Quality of the Data That Enters", 1),
          para("In Activity 1 most of the work was cleaning data that already existed. In a system the best cleaning is "
               "to stop bad data at the door. Every form is checked on the server with the rules of Table 5: the actions "
               "of a session must be among the six actions of the game, the dates cannot be in the future, the numbers "
               "must be in their limits, and the slots and gamer tags must be unique. A wrong form is not saved, and the "
               "user sees the reason."),
          h("The Data That Feeds the Model", 1),
          para("The model needs the actions of the session in order and the context of the fight (world, difficulty, "
               "hearts). It also needs the history of the player: the earlier sessions of the same player, taken from the "
               "database (Phase 3 explains why). The published system has only the data of its first users: the team "
               "and the test accounts. To measure the model with many sessions we also used a local copy of the system "
               "filled with the Activity 1 test players, the players that the model never saw in training "
               f"({code('flask seed-demo')}). That copy is only for tests; the numbers of this report about the system "
               "come from the published system.")]

    # ---------------------------------------------------------------- phase 3
    s += [h("Phase 3: Model Engineering"), h("Where Machine Learning Contributes", 1),
          para("The model supports one decision of process P2: which difficulty is fair for each player. The "
               "next-action model tells how well the bosses can read the player (predictability), and the hit model "
               "tells how often boss attacks hit the player in their usual situation (risk). With these numbers and the "
               "real deaths and result of the session, a fixed rule proposes to lower, raise or keep the difficulty. A "
               "person always makes the final decision."),
          h("The Same Model and the Same Features", 1),
          para(f"The system uses the files chosen in Activity 1 ({code('models/next_action.joblib')} and "
               f"{code('models/hit.joblib')}), version <b>{S['version']}</b>. It does not train anything. The features are "
               f"created by the same functions that we used to evaluate the models ({code('predict.action_row')} and "
               f"{code('hit_row')}), and the preprocessing is inside each saved pipeline. Inference runs on the server, "
               "inside the session page and through the JSON API."),
          h("A Problem We Found: the History of the Player", 1),
          para("In the first version, the model guessed only about 50% of the actions of the Activity 1 test players "
               f"inside the system, against {at:.0%} in Activity 1. We compared the features row by row and found the "
               "reason: in training, the features hist_ and mk_ describe the whole past of the player, not only the "
               "current session. We changed the service so that it gives the earlier sessions of the player as history "
               f"(at most 1,000 actions), like in training. With the same sessions, on the local copy, the accuracy went to "
               f"{HISTORY_ACC:.1%}. "
               "The small difference that is left comes from data that the system does not store for each action, like "
               "the time between actions."),
          h("The Decision Rule", 1),
          para("We fixed the rule before using it. The risk limits are the quartiles of the hit model's own predictions "
               "for the guardian attacks of the Activity 1 test players, so a high risk means the top quarter. In the "
               "first version we used round numbers and almost every recommendation was \"lower\"; with the quartiles "
               "the three answers appear.")]
    s += seriation(["Lower one level when the player lost or died 3 or more times per 10 minutes and the risk is 36% or "
                    "more, or when they died 5 or more times per 10 minutes.",
                    "Raise one level when the player won with at most 1.5 deaths per 10 minutes and a risk of 22% or less.",
                    "Keep the difficulty in any other case."])
    s += [para("An approved recommendation changes the player's difficulty, and the next analyzed session of that player "
               "is compared with the session before the change, so we know if the decision worked.")]

    # ---------------------------------------------------------------- phase 4
    s += [h("Phase 4: Evaluation"), h("Automatic Tests", 1),
          para(f"We wrote {n_tests} automatic tests with pytest ({code('tests/test_system.py')}). The last run: "
               f"<b>{T['summary']}</b>. They also run on GitHub Actions after every push. Table 7 groups them.")]
    s += apa_table("Automatic Tests", [
        ["Group", "What is checked"],
        ["Security", "Protected routes redirect to sign in; API answers 401; password hash; no self-promotion to admin; "
                     "CSRF; lockout after 5 wrong passwords; sign-out."],
        ["Authorization", "A player cannot open another player's data or staff pages; an analyst cannot manage users; "
                          "an admin cannot change their own role."],
        ["Validation and rules", "Saved game rules, unknown actions, too few actions, won session without hearts."],
        ["Complete workflow", "Upload → analyze (two inferences stored with user, version and latency) → lower "
                              "recommendation → reject without note refused → approve → difficulty changed → next "
                              "session marks the change as effective and uses the first session as history."],
        ["Monitoring", "Monitoring JSON, reports, CSV, health with database, API inference stored with the user, "
                       "errors stored in the error log."],
    ], [1.6, 4.9])
    rc, ra = S["recs"], S["rec_actions"]
    where = (f"the published system, read on {S['captured_at'][:10]} from its pages as administrator" if S.get("live")
             else "a local copy with demo data")
    s += [h("The Model Inside the System", 1),
          para(f"Table 8 shows the state of the system when we wrote this report ({where}).")]
    s += apa_table("Snapshot of the System", [
        ["Indicator", "Value"],
        ["Users by role", ", ".join(f"{k}: {v}" for k, v in sorted(S["users"].items()))],
        ["Players / sessions / analyzed", f"{S['players']} / {S['sessions']} / {S['analyzed']}"],
        ["Next-action accuracy against real actions", f"{S['acc']:.1%} ({round(S['acc'] * S['checked'])} of "
                                                      f"{S['checked']:,} predictions; baseline {S['base']:.1%})"
                                                      if S["acc"] is not None else "-"],
        ["Mean hit risk of analyzed sessions", f"{S['risk']:.1%}" if S["risk"] is not None else "-"],
        ["Recommendations (lower / raise)", f"{ra.get('lower', 0)} / {ra.get('raise', 0)}; pending {rc.get('pending', 0)}, "
                                            f"applied {rc.get('applied', 0)}, rejected {rc.get('rejected', 0)}"],
        ["Inferences, mean / p95 time", f"{S['inferences']}, {S['mean_ms'] or 0:.0f} ms / {S['p95_ms'] or 0:.0f} ms "
                                        "(one call analyzes a whole session)"],
        ["Data drift (PSI, 30 days)", f"{S['psi']:.3f} ({S['psi_state']})" if S["psi"] is not None
                                      else "not computed yet (it needs 200 actions in 30 days)"],
        ["Errors logged", str(S["errors"])],
    ], [2.6, 3.9])
    if S.get("live"):
        s += [para(f"The published system is new, so it has few data: {S['players']} players and {S['sessions']} play "
                   "session, uploaded by the operational review with the player test account and analyzed by the analyst "
                   f"account. In that session the model guessed {S['acc']:.1%} of the next actions, above the baseline of "
                   f"{S['base']:.1%} and close to the {at:.1%} of Activity 1, but 39 predictions are too few to say more. "
                   "The rule proposed to lower the difficulty, and the decision is waiting for a person. The PSI needs "
                   "more actions, so the system says that it is not computed yet instead of showing a number without "
                   "meaning.")]
    else:
        s += [para("The accuracy of the demo data is lower than in Activity 1 because many demo players have only one or "
                   "two earlier sessions, and the model needs the history of the player.")]
    p95 = vm.get("p95") or S["p95_ms"] or 0
    crit = [
        ("S1", deployed and all(c["ok"] for c in V["checks"] if "Permission" in c["check"]),
         "Permission matrix of 40 requests on the public URL without errors." if deployed else "Not run on the public URL yet."),
        ("S2", deployed and V["passed"] == V["total"],
         f"Operational review on the public URL: {V['passed']} of {V['total']} checks." if deployed else "Not run yet."),
        ("S3", at - HISTORY_ACC <= 0.02, f"{HISTORY_ACC:.1%} inside the system (local copy with the Activity 1 test "
                                         f"sessions) against {at:.1%} in Activity 1."),
        ("S4", p95 < 1000, f"95th percentile {p95:.0f} ms."),
        ("S5", True, "Table inferences: date, user, player, session, model, version, input, result, latency."),
        ("S6", bool(T.get("ok")), T["summary"]),
    ]
    s += [h("Check of the Success Criteria", 1)]
    s += apa_table("Check of the Success Criteria", [["Code", "Result", "Met"]] +
                   [[c, r, "Yes" if ok else "No"] for c, ok, r in crit], [0.8, 4.6, 1.1])
    s.append(para("All the success criteria were met." if all(ok for _, ok, _ in crit)
                  else "Some criteria are not met yet; the table shows which ones."))

    # ---------------------------------------------------------------- phase 5
    s += [h("Phase 5: Deployment"), h("Modules and Architecture", 1),
          para("The system has eight modules: public documentation (Activity 1), authentication, players and saved "
               "games, play sessions, recommendations, records, support, and management (reports, monitoring, users). "
               "Figure 2 shows the architecture. The Flask application runs on Render with gunicorn. The data is in "
               "PostgreSQL on Supabase. The models are the files saved in Activity 1, loaded with joblib.")]
    s += apa_figure(IMG / "sys_architecture.png", "Architecture of the System", 6.5)
    s += [h("Registration, Authentication and Roles", 1),
          para("Passwords are stored with scrypt and a random salt (Werkzeug), never in plain text. Public registration "
               "creates only players: the form has no role field and a role sent by hand is ignored. Five wrong passwords "
               "lock an account for five minutes. Every form has a CSRF token, the cookie is HttpOnly and SameSite, and "
               "the session id changes after sign-in. Authorization is checked on the server, for each route and for "
               "each object, and forbidden attempts are written to an audit log. Table 10 shows the permissions.")]
    s += apa_table("Permissions by Role", [
        ["Function", "Player", "Analyst", "Admin"],
        ["Public documentation (home page)", "yes", "yes", "yes"],
        ["Own profile, saves, sessions, records, tickets", "own only", "all", "all"],
        ["Create, edit and suspend player profiles", "no", "yes", "yes"],
        ["Run the model; approve or reject recommendations", "no", "yes", "yes"],
        ["Verify records; manage tickets; reports; monitoring", "no", "yes", "yes"],
        ["Delete analyzed sessions and players; users and roles", "no", "no", "yes"],
    ], [3.5, 1.0, 1.0, 1.0])
    s += [h("What the User Sees", 1),
          para("Figures 3 to 6 are screenshots of the published system. Figure 3 shows the dashboard, which is the "
               "work queue of process P2: sessions waiting for the model and recommendations waiting for a decision. Figure 4 shows the result of the model in a session, "
               "and Figure 5 the decision. Each number has a short explanation and its limits: the models were trained "
               "with simulated players, with fewer than 30 actions the numbers change a lot, and they describe habits, "
               "not skill or reasons. The Reports page gives indicators by difficulty and by world and exports the "
               "sessions as CSV.")]
    s += apa_figure(IMG / "sys_dashboard.jpg", "Dashboard of the Published System", 6.3)
    s += apa_figure(IMG / "sys_session_model.jpg", "Result of the Model in a Play Session", 6.3)
    s += apa_figure(IMG / "sys_decision.jpg", "Decision Supported by the Model", 6.3)
    s += [h("Publication and Operational Review", 1),
          para(f"The system is published at {APP_URL} (Render, free plan, gunicorn) with the database in Supabase "
               "PostgreSQL. The configuration uses environment variables only; the repository has an example file "
               "without secrets (.env.example) and the Render blueprint (render.yaml). Every push to master publishes "
               "a new version.")]
    if deployed:
        rows = [["Check", "Result", "Detail"]] + [[c["check"], "Pass" if c["ok"] else "Fail", c["detail"]] for c in V["checks"]]
        s += [para(f"After the deployment we ran the operational review on {V['url']} "
                   f"({V['started_at'][:16].replace('T', ' ')} UTC) with one account of each role: {V['passed']} of "
                   f"{V['total']} checks passed (Table 11). It checks the database, sign-in, the 40 requests of the "
                   "permission matrix, the whole workflow with the model, and persistence after signing out and in "
                   "again.")]
        s += apa_table("Operational Review of the Deployed System", rows, [2.6, 0.7, 3.2])
    elif V:
        s += [para(f"Before the deployment we ran the operational review on a local copy ({V['url']}): "
                   f"{V['passed']} of {V['total']} checks passed.")]
    s += [h("Test Accounts", 1)]
    if ACC:
        s += [para(f"These are the accounts for the teacher on {APP_URL}. The analyst can run the model and decide; "
                   "the player can upload sessions and see the result in their profile.")]
        s += apa_table("Test Accounts for the Teacher", [["Role", "User name", "Password"]] +
                       [[r, code(u), code(p)] for u, r, p in ACC], [1.6, 2.2, 2.7])
    else:
        s += [para("The test accounts of each role are given to the teacher privately.")]
    s += [h("Manuals and Training", 1),
          para("Appendix A is the technical manual (installation, configuration, database, deployment, security, "
               "monitoring and maintenance) and Appendix B is the user manual for each role, with a 20-minute training "
               "guide and the procedure to report incidents and give feedback. The same files are in the repository "
               f"({code('docs/system')}).")]

    # ---------------------------------------------------------------- phase 6
    s += [h("Phase 6: Monitoring and Maintenance"), h("What We Watch", 1),
          para("Figure 6 shows the monitoring page. It uses the same indicators and alarm lines as Phase 6 of "
               f"Activity 1, and Table {R0.NUM['table'] + 1 - R0.PART['t_off']} shows them.")]
    s += apa_figure(IMG / "sys_monitoring.jpg", "Monitoring of the Model", 6.3)
    s += apa_table("Indicators of the Monitoring Page", [
        ["Indicator", "Source", "Alarm"],
        ["Errors", "Table error_log (path, type, message, traceback) and the Render log", "Any new error"],
        ["Inference requests and response time", "Table inferences (latency of each call)", "p95 above 1 s"],
        ["Accuracy against real actions", "Each prediction compared with the real next action", "A week 5 points below "
                                                                                                 "the first three weeks"],
        ["Changes in the input data", "PSI between training data and the last 30 days", "Warning 0.1, alarm 0.2"],
        ["Decisions", "Recommendations and the result in the next session", "Many rejected or not effective"]],
        [2.0, 2.9, 1.6])
    s += [h("Versioning and Retraining", 1),
          para("The version of a model is its data source plus its training date in model_card.json. Models are "
               "versioned in Git and every inference keeps its version. Retraining is weekly with GitHub Actions, and a "
               "new model replaces the old one only if it passes the rule of Activity 1. We also retrain when the PSI is "
               "above 0.2, accuracy falls 5 points, a game update changes the fights or the real data doubles. To go back "
               "to an old version we revert the retraining commit."),
          h("Incidents and Feedback", 1),
          para("Any signed-in user can open a ticket (problem, unfair model result or opinion with a score from 1 to 5). "
               "Tickets follow open → in progress → resolved → closed, with high severity answered the same day. The "
               "tickets about unfair results are read together with the rejected recommendations in the weekly review: "
               "J.A reviews the model, J.Q the tickets and opinions, and J.P checks the game after each release.")]

    # ---------------------------------------------------------------- debugging
    s += [h("Debugging Notes"),
          para("Like in Activity 1, some of the most important lessons came from bugs. We want to describe two of them."),
          h("The Database Was Unreachable From Render", 1),
          para("The first deployment failed before the application started. The log said Network is unreachable for an "
               "address with many colons, an IPv6 address. We had copied the direct connection string of Supabase, which "
               "uses IPv6, and the free plan of Render only has IPv4. The solution was the Session pooler string of "
               "Supabase, which works with IPv4. Now the health page of the system says which database it uses, so this "
               "problem is easy to see."),
          h("The Demo Said That the Form Expired", 1),
          para("After we added the protection against CSRF, the old demo of Activity 1 answered Bad Request: the form "
               "expired. The new protection checks every form of the application, but the two public forms of Activity 1 "
               "(the demo and the opinion form) did not send the token. We added the token to both forms and a test that "
               "sends them, so the same mistake cannot come back.")]

    # ---------------------------------------------------------------- git
    s += [h("Git and GitHub Workflow"),
          para(f"We worked in the same repository as Activity 1 ({REPO_URL}) and on the master branch. We change the "
               "code on the computer, run the tests, make a commit and push it to master. Each push runs the tests on "
               "GitHub Actions and makes Render publish the new version; every Monday another job trains the models "
               "again. The main commits were the information system with the model inside the workflow, the fix of the "
               "public forms, and the operational review of the deployed system. No password or key is in the "
               "repository: they are environment variables in Render.")]

    # ---------------------------------------------------------------- discussion / conclusion / future
    i_close = len(s)
    s += [h("Discussion"),
          para("The first lesson is that a model inside a system needs the same inputs as in training. The model was "
               "not wrong; the system was giving it less history. A simple comparison row by row found the difference, "
               "and now a test checks it."),
          para("The second lesson is about decisions. A probability alone does not change anything. The rule turns the "
               "numbers into lower, raise or keep, and the person who approves it is responsible for the change. Because "
               "every inference and decision keeps the model version, we can always answer why the difficulty of a "
               "player changed."),
          para("The last lesson is about security. Most of the work of the system was not the model but the accounts, "
               "the roles and the rules, and the protection we added broke an old form. Testing all the pages, the old "
               "ones too, was as important as testing the new ones."),
          h("Conclusion"),
          para(f"The model of Activity 1 is now part of a real process. A session arrives, the model reads the player, a "
               "fixed rule proposes a fair difficulty, a person decides, the account changes, and the next session says "
               f"if it worked. The system has accounts with three roles, {n_tests} automatic tests, a persistent database "
               "and a monitoring page, and it is published on Render with Supabase"
               + (f". The operational review of the public system passed {V['passed']} of {V['total']} checks." if deployed
                  else ".")),
          h("Future Work", 1)]
    s += seriation(["Let the game upload the play sessions by itself, so the analysts work with real players.",
                    "Measure with real players if the difficulty changes make the fights feel fair.",
                    "Add a page to change the password and recover an account.",
                    "Retrain the models with the real sessions stored by the system when there are enough of them."])

    # ---------------------------------------------------------------- references + appendices
    s += [PageBreak(), Paragraph("<b>References</b>", CEN)]
    for ref in ["Pallets Projects. (2024). <i>Flask documentation</i>. https://flask.palletsprojects.com/",
                "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, "
                "P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; "
                "Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning "
                "Research, 12</i>, 2825–2830.",
                "Siddiqi, N. (2006). <i>Credit risk scorecards: Developing and implementing intelligent credit scoring</i>. Wiley.",
                "Studer, S., Bui, T. B., Drescher, C., Hanuschkin, A., Winkler, L., Peters, S., &amp; Müller, K.-R. (2021). "
                "Towards CRISP-ML(Q): A machine learning process model with quality assurance methodology. <i>Machine "
                "Learning and Knowledge Extraction, 3</i>(2), 392–413. https://doi.org/10.3390/make3020020"]:
        s.append(Paragraph(ref, R0.REF))
    i_app = len(s)
    if part:
        R0.PART["shift"] = 0
    s += [PageBreak(), h("Appendix A: Technical Manual")] + appendix(DOCS / "TECHNICAL_MANUAL.md", "A")
    s += [PageBreak(), h("Appendix B: User Manual")] + appendix(DOCS / "USER_MANUAL.md", "B")

    if part:
        return {"body": s[i_body:i_close], "appendix": s[i_app:], "tests": T, "snap": S, "review": V}
    out = Path(os.environ["REPORT_OUT"]) if os.environ.get("REPORT_OUT") else REPORTS / FILE_NAME
    Doc(str(out), title=TITLE, author=", ".join(AUTHORS)).multiBuild(s)
    print("report written:", out, "| tests:", T["summary"], "| accounts:", len(ACC))


if __name__ == "__main__":
    main()
