"""Activity 2 technical report (PDF), same APA 7 format as the Activity 1 report (report.py).

    python -m src.umbral_ml.report_system            (uses the local database or DATABASE_URL for the numbers)

Every number comes from the system itself (database snapshot), the automatic tests, the operational review
evidence (reports/operational_review.json for the deployed system, *_local.json for a local run) and the
Activity 1 result files. The technical and user manuals (docs/system/*.md) are added as appendices, so the
repository, the app and the PDF say the same thing.
"""
from __future__ import annotations

import json
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

TITLE = "From Traces to Decisions: The UMBRAL Player Information System"
FILE_NAME = "R3 Activity 2 From traces to decisions.pdf"
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
    """Numbers of the running system (players, sessions, decisions, monitoring)."""
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
def main() -> None:
    R0.NUM.update(table=0, figure=0)
    T = run_tests()
    S = snapshot()
    V = review()
    A1 = json.loads((REPORTS / "results.json").read_text(encoding="utf-8"))
    at = A1["A"]["test"][A1["A"]["chosen"]]["accuracy"]
    s: list = []

    # ---------------------------------------------------------------- title page
    if (IMG / "logo_udec.png").exists():
        from reportlab.platypus import Image  # noqa: PLC0415
        s += [Spacer(1, 0.2 * inch), Image(str(IMG / "logo_udec.png"), width=1.1 * inch, height=1.1 * inch * 150 / 99)]
    s += [Spacer(1, 0.6 * inch), Paragraph(f"<b>{TITLE}</b>", CEN), Spacer(1, DOUBLE)]
    s += [Paragraph(a, CEN) for a in AUTHORS]
    s += [Paragraph(INSTITUTION, CEN), Paragraph(COURSE, CEN), Paragraph(TEACHER, CEN), Paragraph(DATE, CEN),
          Spacer(1, DOUBLE),
          Paragraph("Activity 2: a web information system that manages the players and accounts of UMBRAL and uses "
                    "the model of Activity 1 inside its processes", CEN),
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
          para("A demo is not a system. Nobody decides anything with a demo form. In this activity we built a "
               "complete web information system around the model. It manages the players and accounts of the game: "
               "player profiles, saved games, play sessions, records and support tickets. The model is not a separate "
               "page. It is one step of the process: when a play session arrives, an analyst runs the model, the model "
               "proposes a change of difficulty for that player, a person approves or rejects it, the account changes, "
               "and the next session tells us if the change worked."),
          para("We extended the same Flask application and the same repository, so the methodological documentation "
               "of Activity 1 is still the home page, and every prediction can be traced to the evaluated model and "
               "its version.")]

    # ---------------------------------------------------------------- 1. design
    s += [h("System Design"), h("Users", 1),
          para("Three kinds of people use the system. Players are the people who play UMBRAL; they register by "
               "themselves. Analysts are members of the team who review sessions with the model and make the "
               "decisions. Administrators manage users and roles. In our team J.A is the administrator and the owner "
               "of the model, and J.P and J.Q are analysts. Table 1 shows the business processes.")]
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
          para("Table 2 lists the main functional requirements, and Table 3 the business rules that the server "
               "checks. The main rules have automatic tests (Table 6).")]
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

    s += [h("Modules and Architecture", 1),
          para("The system has eight modules: public documentation (Activity 1), authentication, players and saved "
               "games, play sessions, recommendations, records, support, and management (reports, monitoring, users). "
               "Figure 1 shows the architecture. The Flask application runs on Render with gunicorn. The data is in "
               "PostgreSQL on Supabase, the same provider that the game already uses for its accounts, in separate "
               "tables. The models are the files saved in Activity 1, loaded with joblib.")]
    s += apa_figure(IMG / "sys_architecture.png", "Architecture of the System", 6.5)
    s += [h("Database Model", 1),
          para("Figure 2 shows the database model. The two blue tables connect the business process with the model: "
               "every inference and every recommendation points to the session and the user. We used SQLAlchemy, so "
               "the same code works with SQLite for the tests and PostgreSQL in production.")]
    s += apa_figure(IMG / "sys_er.png", "Database Model", 6.5)
    s += [h("Where Machine Learning Contributes", 1),
          para("The model supports one decision of process P2: which difficulty is fair for each player. The "
               "next-action model tells how well the bosses can read the player (predictability), and the hit model "
               "tells how often boss attacks hit the player in their usual situation (risk). With these numbers and "
               "the real deaths and result of the session, a fixed rule proposes to lower, raise or keep the "
               "difficulty. A person always makes the final decision. This follows the worry of our domain expert in "
               "Activity 1: that the bosses could feel unfair for some players.")]

    # ---------------------------------------------------------------- 2. auth and roles
    s += [h("Registration, Authentication and Roles"),
          para("Passwords are stored with scrypt and a random salt (Werkzeug), never in plain text. Public registration "
               "creates only players: the form has no role field and a role sent by hand is ignored. Five wrong "
               "passwords lock an account for five minutes. Every form has a CSRF token, the cookie is HttpOnly and "
               "SameSite, and the session id changes after sign-in. Authorization is checked on the server, for each "
               "route and for each object, and forbidden attempts are written to an audit log. Table 5 shows the "
               "permissions.")]
    s += apa_table("Permissions by Role", [
        ["Function", "Player", "Analyst", "Admin"],
        ["Public documentation (home page)", "yes", "yes", "yes"],
        ["Own profile, saves, sessions, records, tickets", "own only", "all", "all"],
        ["Create, edit and suspend player profiles", "no", "yes", "yes"],
        ["Run the model; approve or reject recommendations", "no", "yes", "yes"],
        ["Verify records; manage tickets; reports; monitoring", "no", "yes", "yes"],
        ["Delete analyzed sessions and players; users and roles", "no", "no", "yes"],
    ], [3.5, 1.0, 1.0, 1.0])

    # ---------------------------------------------------------------- 3. processes
    s += [h("Business Process Management"),
          para("All the data is in a persistent PostgreSQL database. Each process has the create, read, update and "
               "delete operations that it needs, with validation on the server and the rules of Table 3. Figure 3 "
               "shows the analyst dashboard, which is the work queue of process P2: sessions waiting for the model "
               "and recommendations waiting for a decision. The Reports page gives indicators by difficulty and by "
               "world (sessions, deaths per 10 minutes, win rate, model accuracy), the recommendations that worked, "
               "tickets and the mean opinion score, and exports the sessions as CSV.")]
    s += apa_figure(IMG / "sys_dashboard.jpg", "Dashboard of an Analyst", 6.3)

    # ---------------------------------------------------------------- 4. ML integration
    s += [h("Machine Learning Integration"), h("The Same Model and the Same Features", 1),
          para(f"The system uses the files chosen in Activity 1 (<font face='Courier'>models/next_action.joblib</font> "
               f"and <font face='Courier'>models/hit.joblib</font>), version <b>{S['version']}</b>. It does not train "
               "anything. The features are created by the same functions that we used to evaluate the models "
               "(<font face='Courier'>predict.action_row</font> and <font face='Courier'>hit_row</font>), and the "
               "preprocessing is inside each saved pipeline. Inference runs on the server, inside the session page and "
               "through the JSON API, and the input data is taken from the records of the system (the stored session "
               "and the earlier sessions of the same player).")]
    s += [h("A Problem We Found: the History of the Player", 1),
          para("In the first version, the model guessed only about 50% of the actions of the Activity 1 test players "
               "inside the system, against 55% in Activity 1. We compared the features row by row and found the reason: "
               "in training, the features hist_ and mk_ describe the whole past of the player, not only the current "
               "session. We changed the service so that it gives the earlier sessions of the player as history, like "
               "in training. With the same sessions the accuracy went back to 54%. The small difference that is left "
               "comes from data that the system does not store for each action, like the time between actions.")]
    s += [h("What the User Sees", 1),
          para("Figure 4 shows the result of the model in a session, and Figure 5 the decision. Each number has a "
               "short explanation and its limits: the models were trained with simulated players, with fewer than "
               "30 actions the numbers change a lot, and they describe habits, not skill or reasons.")]
    s += apa_figure(IMG / "sys_session_model.jpg", "Result of the Model in a Play Session", 6.3)
    s += apa_figure(IMG / "sys_decision.jpg", "Decision Supported by the Model", 6.3)
    s += [h("The Decision Rule", 1),
          para("We fixed the rule before using it. The risk limits are the quartiles of the hit model's own "
               "predictions for the guardian attacks of the Activity 1 test players, so a high risk means the top "
               "quarter.")]
    s += seriation(["Lower one level when the player lost or died 3 or more times per 10 minutes and the risk is 36% or "
                    "more, or when they died 5 or more times per 10 minutes.",
                    "Raise one level when the player won with at most 1.5 deaths per 10 minutes and a risk of 22% or less.",
                    "Keep the difficulty in any other case."])
    s += [para("Every inference is stored with its date, user, player, session, model name, model version, input, "
               "result and response time. An approved recommendation changes the player's difficulty, and the next "
               "analyzed session of that player is compared with the session before the change.")]

    # ---------------------------------------------------------------- 5. testing
    s += [h("Testing"),
          para(f"We wrote {T.get('passed', 0) + (T.get('failed') or 0)} automatic tests with pytest "
               f"(<font face='Courier'>tests/test_system.py</font>). The last run: <b>{T['summary']}</b>. They also run on "
               "GitHub Actions after every push. Table 6 groups them.")]
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

    # ---------------------------------------------------------------- 6. deployment
    s += [h("Deployment"),
          para(f"The system is published at {APP_URL} (Render, free plan, gunicorn) with the database in Supabase "
               "PostgreSQL. The configuration uses environment variables only; the repository has an example file "
               "without secrets (.env.example) and the Render blueprint (render.yaml). The test accounts of each role "
               "are given to the teacher privately.")]
    if V and V.get("environment") == "DEPLOYED":
        rows = [["Check", "Result", "Detail"]] + [[c["check"], "Pass" if c["ok"] else "Fail", c["detail"]] for c in V["checks"]]
        s += [para(f"After the deployment we ran the operational review on {V['url']} "
                   f"({V['started_at'][:16].replace('T', ' ')} UTC): {V['passed']} of {V['total']} checks passed "
                   "(Table 7). It checks the database, sign-in, the 40 requests of the permission matrix, the "
                   "whole workflow with the model, and persistence after signing out and in again.")]
        s += apa_table("Operational Review of the Deployed System", rows, [2.6, 0.7, 3.2])
    elif V:
        s += [para(f"Before the deployment we ran the operational review on a local copy ({V['url']}): "
                   f"{V['passed']} of {V['total']} checks passed. The same command is run on the public URL after "
                   "each deployment (Technical Manual, section 10), and its evidence replaces this paragraph.")]

    # ---------------------------------------------------------------- 7. monitoring
    s += [h("Monitoring and Maintenance"),
          para("Figure 6 shows the monitoring page. It uses the same indicators and alarm lines as Phase 6 of "
               "Activity 1.")]
    s += apa_figure(IMG / "sys_monitoring.jpg", "Monitoring of the Model", 6.3)
    s += seriation(["Errors are stored in the database with path, type, message and traceback, and printed in the "
                    "Render log.",
                    "Inference requests and response times are stored for every call (mean and 95th percentile).",
                    "Predictions are compared with real outcomes at once: in every analyzed session each prediction of "
                    "the next action is checked against the real next action. Accuracy is shown per week, with an alarm "
                    "5 points below the first three weeks. Difficulty changes are checked with the next session.",
                    "Changes in the input data are measured with the PSI between the training data and the last 30 "
                    "days (warning 0.1, alarm 0.2).",
                    "Versioning: the version is the data source plus the training date in model_card.json; models are "
                    "versioned in Git and every inference keeps its version. Retraining is weekly with GitHub Actions, "
                    "and a new model replaces the old one only if it passes the rule of Activity 1. We also retrain when "
                    "the PSI is above 0.2, accuracy falls 5 points, a game update changes the fights or the real data "
                    "doubles."])

    # ---------------------------------------------------------------- results
    rc, ra = S["recs"], S["rec_actions"]
    s += [h("Results"),
          para(f"Table 8 shows the state of the system when we wrote this report ({S['db']} database). The sessions are "
               "demo data built from the Activity 1 test players (SIMULATED), because the real play log does not have "
               "enough data yet.")]
    s += apa_table("Snapshot of the System", [
        ["Indicator", "Value"],
        ["Users by role", ", ".join(f"{k}: {v}" for k, v in sorted(S["users"].items()))],
        ["Players / sessions / analyzed", f"{S['players']} / {S['sessions']} / {S['analyzed']}"],
        ["Next-action accuracy against real actions", f"{S['acc']:.1%} of {S['checked']:,} predictions (baseline "
                                                      f"{S['base']:.1%})" if S["acc"] is not None else "-"],
        ["Mean hit risk of analyzed sessions", f"{S['risk']:.1%}" if S["risk"] is not None else "-"],
        ["Recommendations (lower / raise)", f"{ra.get('lower', 0)} / {ra.get('raise', 0)}; pending {rc.get('pending', 0)}, "
                                            f"applied {rc.get('applied', 0)}, rejected {rc.get('rejected', 0)}"],
        ["Inferences, mean / p95 time", f"{S['inferences']}, {S['mean_ms'] or 0:.0f} ms / {S['p95_ms'] or 0:.0f} ms "
                                        "(one call analyzes a whole session)"],
        ["Data drift (PSI, 30 days)", f"{S['psi']:.3f} ({S['psi_state']})" if S["psi"] is not None else "-"],
        ["Errors logged", str(S["errors"])],
    ], [2.6, 3.9])
    s += [para("The accuracy inside the system is lower than in Activity 1 because many demo players have only one or "
               "two earlier sessions, and the model needs the history of the player. This agrees with Figure 6 of "
               "Activity 1, where accuracy grows with the number of actions already seen.")]

    # ---------------------------------------------------------------- conclusion
    s += [h("Conclusion"),
          para("The model of Activity 1 is now part of a real process. A session arrives, the model reads the player, "
               "a fixed rule proposes a fair difficulty, a person decides, the account changes, and the next session "
               "says if it worked. Everything is stored and can be traced to the model version. The most useful lesson "
               "was the history problem: the system must give the model the same inputs as in training, and a simple "
               "comparison row by row found the difference. Next, we want real players, to measure if the difficulty "
               "changes really make the fights feel fair, and to let the game upload the sessions by itself.")]

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
    s += [PageBreak(), h("Appendix A: Technical Manual")] + appendix(DOCS / "TECHNICAL_MANUAL.md", "A")
    s += [PageBreak(), h("Appendix B: User Manual")] + appendix(DOCS / "USER_MANUAL.md", "B")

    out = REPORTS / FILE_NAME
    Doc(str(out), title=TITLE, author=", ".join(AUTHORS)).multiBuild(s)
    print("report written:", out, "| tests:", T["summary"])


if __name__ == "__main__":
    main()
