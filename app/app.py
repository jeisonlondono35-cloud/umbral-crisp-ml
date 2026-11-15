"""UMBRAL CRISP-ML - Flask application.

Cover page + the six CRISP-ML phases (each one reads the SAME result files that the PDF report uses:
reports/results.json, data/processed/data_report.json, monitoring/*.json, docs/expert_review.json)
+ a live demo and a JSON API that serve the saved models.

Run locally:   python -m flask --app app.app run      (or: python app/app.py)
Production:    gunicorn app.app:app
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_from_directory, url_for
from flask_login import current_user, login_required

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.umbral_ml import predict as P  # noqa: E402
from src.umbral_ml.config import (ACTIONS, ACTION_LABEL, DIFFS, FIG, HIT_X, MODES, MON, MOVES, MODELS,  # noqa: E402
                                  PROC, REPORTS, WORLDS)

app = Flask(__name__)

# Activity 2: the information system (database, accounts, roles, business processes, model in the workflow)
from app.system import init_system, log_error  # noqa: E402
from app.system import ml_service as MLS  # noqa: E402
from app.system.auth import role_required  # noqa: E402
from app.system.models import db as _db  # noqa: E402

init_system(app)


def _warm_up() -> None:
    """Load the models and make one prediction at start-up, BEFORE serving, so the first visitor does not
    wait (~13 s on Render's free instance). It must not run in a background thread: a request importing
    scikit-learn at the same time hit a half-initialised module (ImportError, Nov 2026)."""
    try:
        P.predict_next(["shoot", "dodge"], {"boss": True})
        P.predict_hit("dive", {"close": 1}, {})
    except Exception as e:  # never stop the app because of the warm-up
        print("warm-up failed:", e)


_warm_up()
LIVE_LOG = MON / "live_predictions.jsonl"
REPORT2 = "R3A3 Speaking Models.pdf"
FEEDBACK = MON / "feedback.jsonl"

PHASES = [
    (1, "Business & Data Understanding", "business"),
    (2, "Data Engineering", "data"),
    (3, "Model Engineering", "model"),
    (4, "Model Evaluation", "evaluation"),
    (5, "Model Deployment", "deployment"),
    (6, "Monitoring & Maintenance", "monitoring"),
]


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default if default is not None else {}


def ctx() -> dict:
    res = _json(REPORTS / "results.json")
    exps = res.get("experiments", [])
    return {"R": res, "D": _json(PROC / "data_report.json"), "M": _json(MON / "monitoring_simulated.json"),
            "E": _json(ROOT / "docs" / "expert_review.json"), "PUB": _json(ROOT / "data" / "raw" / "ml_public.json"),
            "EV": _json(PROC / "evidence.json"),
            "LS": _json(ROOT / "data" / "raw" / "log_stats.json"),
            "REPO": "https://github.com/jeisonlondono35-cloud/umbral-crisp-ml", "GAME": "https://juego-pixelart.onrender.com/", "card": _json(MODELS / "model_card.json"),
            "exps": exps, "PHASES": PHASES, "ACTIONS": ACTIONS, "AL": ACTION_LABEL, "WORLDS": WORLDS,
            "MODES": MODES, "DIFFS": DIFFS, "MOVES": MOVES, "HIT_X": HIT_X,
            "model_files": sorted((p.name, round(p.stat().st_size / 1024, 1)) for p in MODELS.iterdir()),
            "has_pdf": (REPORTS / "R3A2 Traces in data.pdf").exists()}


@app.context_processor
def base_ctx():
    """What base.html needs on EVERY page (the system pages do not pass the CRISP-ML context)."""
    return {"PHASES": PHASES, "R": _json(REPORTS / "results.json"), "D": _json(PROC / "data_report.json"),
            "has_pdf": (REPORTS / "R3A2 Traces in data.pdf").exists(), "has_pdf2": (REPORTS / REPORT2).exists()}


@app.template_filter("pct")
def pct(v, d=1):
    return "-" if v is None else f"{100 * float(v):.{d}f}%"


@app.template_filter("num")
def num(v, d=3):
    return "-" if v is None else f"{float(v):.{d}f}"


@app.errorhandler(Exception)
def on_error(e):
    """Incident logging (Phase 5): every unexpected error is printed with its traceback to the server log
    (Render → Logs) and the API answers with the error type instead of a bare 500 page."""
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    import traceback
    traceback.print_exc()
    log_error(e)                       # Activity 2: errors are also kept in the database (monitoring page)
    if request.path.startswith("/api/"):
        return jsonify({"error": "internal", "type": type(e).__name__, "detail": str(e)[:300]}), 500
    return render_template("base.html", title="Error", **ctx()).replace(
        "<main>", f"<main><p class='banner'><b>Something went wrong</b> ({type(e).__name__}: {str(e)[:200]}). "
                  "It was logged; please report it in Phase 5.</p>", 1), 500


@app.route("/")
def home():
    return render_template("home.html", **ctx())


@app.route("/phase/<int:n>")
def phase(n: int):
    if not 1 <= n <= 6:
        abort(404)
    return render_template(f"phase{n}.html", n=n, title=PHASES[n - 1][1], **ctx())


@app.route("/figures/<path:name>")
def figure(name):
    return send_from_directory(FIG, name)


@app.route("/report.pdf")
def report_pdf():
    return send_from_directory(REPORTS, "R3A2 Traces in data.pdf", download_name="R3A2 Traces in data.pdf")


@app.route("/report-activity2.pdf")
def report2_pdf():
    return send_from_directory(REPORTS, REPORT2, download_name=REPORT2)


@app.route("/experiments.csv")
def experiments_csv():
    return send_from_directory(REPORTS, "experiments.csv", mimetype="text/csv")


# ------------------------------------------------------------------------------------------- demo
def _log(kind: str, inp: dict, out: dict, ms: float) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind,
           "model": out.get("model"), "input": inp, "output": {k: v for k, v in out.items() if k != "features"},
           "latency_ms": round(ms, 2), "source": "LIVE demo request"}
    try:
        with open(LIVE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def _next_input(d: dict) -> tuple[list[str], dict]:
    hist = d.get("history", [])
    if isinstance(hist, str):
        hist = [h.strip() for h in hist.replace(";", ",").split(",") if h.strip()]
    c = {k: d.get(k) for k in ("w", "mode", "diff", "hp", "ne", "boss", "is_start")}
    c = {k: v for k, v in c.items() if v not in (None, "")}
    for k in ("w", "diff"):
        if k in c:
            c[k] = int(c[k])
    for k in ("hp", "ne"):
        if k in c:
            c[k] = float(c[k])
    for k in ("boss", "is_start"):
        if k in c:
            c[k] = str(c[k]).lower() in ("1", "true", "on", "yes")
    bad = [h for h in hist if h not in ACTIONS]
    if bad:
        raise ValueError(f"unknown actions: {bad}; use {ACTIONS}")
    return hist, c


def _db_log(kind: str, model: str, inp: dict, out: dict, result: str, ms: float) -> None:
    """Activity 2: every API inference is stored with date, user, model version, result and latency."""
    MLS._log(kind, current_user.id, None, None, model, inp, {k: v for k, v in out.items() if k != "features"},
             result, ms)
    _db.session.commit()


@app.route("/api/predict/next", methods=["POST"])
@login_required
def api_next():
    try:
        hist, c = _next_input(request.get_json(force=True, silent=True) or request.form.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    t = time.perf_counter()
    out = P.predict_next(hist, c)
    ms = 1000 * (time.perf_counter() - t)
    _log("next_action", {"history": hist, **c}, out, ms)
    _db_log("next_action", out["model"], {"history": hist, **c}, out,
            f"{out['prediction']} {out['probability']:.0%}", ms)
    return jsonify({**out, "latency_ms": round(ms, 2), "model_version": MLS.model_version()})


def _hit_input(d: dict) -> tuple[str, dict, dict]:
    move = str(d.get("move", "dive"))
    x = {}
    for name in HIT_X[1:]:
        v = d.get(name, 0)
        x[name] = float(v) if str(v).replace(".", "", 1).isdigit() else (1.0 if str(v).lower() in ("on", "true") else 0.0)
    c = {"w": int(d.get("w", 1) or 1), "diff": int(d.get("diff", 1) or 1), "mode": d.get("mode", "story") or "story"}
    return move, x, c


@app.route("/api/predict/hit", methods=["POST"])
@login_required
def api_hit():
    move, x, c = _hit_input(request.get_json(force=True, silent=True) or request.form.to_dict())
    t = time.perf_counter()
    out = P.predict_hit(move, x, c)
    ms = 1000 * (time.perf_counter() - t)
    _log("hit", {"move": move, **x, **c}, out, ms)
    _db_log("hit", out["model"], {"move": move, **x, **c}, out, f"{move} hits {out['p_hit']:.0%}", ms)
    return jsonify({**out, "latency_ms": round(ms, 2), "model_version": MLS.model_version()})


@app.route("/api/predict/dodge", methods=["POST"])
@login_required
def api_dodge():
    d = request.get_json(force=True, silent=True) or {}
    hist = [(float(a), float(b)) for a, b in d.get("history", [])]
    out = P.predict_dodge(hist)
    _log("dodge", {"history": hist}, out, 0)
    return jsonify(out)


@app.route("/api/health")
def health():
    card = _json(MODELS / "model_card.json")
    try:
        _db.session.execute(_db.text("select 1"))
        dbok = "ok"
    except Exception as e:  # noqa: BLE001
        dbok = f"error: {type(e).__name__}"
    warn = app.config.get("CONFIG_WARNINGS", [])
    return jsonify({"status": "ok" if dbok == "ok" and not warn else "degraded", "database": dbok,
                    "database_engine": _db.engine.dialect.name, "config_warnings": warn,
                    "model_version": MLS.model_version(), "data_source": card.get("source"), "trained_at": card.get("trained_at"),
                    "next_action_model": card.get("next_action", {}).get("model"),
                    "hit_model": card.get("hit", {}).get("model")})


@app.route("/demo", methods=["GET", "POST"])
def demo():
    out_next = out_hit = err = None
    run = request.method == "POST" or "history" in request.args      # a GET with ?history=... also predicts (shareable link)
    form = request.form.to_dict() if request.method == "POST" else request.args.to_dict() if run else {
        "history": "shoot, shoot, dodge, shoot, shoot, dodge, shoot, shoot, dodge, shoot", "w": "2", "mode": "story", "diff": "1", "hp": "4",
        "ne": "160", "boss": "on", "move": "feint", "close": "on", "blocking": "on"}
    if run:
        try:
            hist, c = _next_input(form)
            t = time.perf_counter()
            out_next = P.predict_next(hist, c)
            _log("next_action", {"history": hist, **c}, out_next, 1000 * (time.perf_counter() - t))
            move, x, hc = _hit_input({**form, "w": form.get("w"), "diff": form.get("diff"), "mode": form.get("mode")})
            t = time.perf_counter()
            out_hit = P.predict_hit(move, x, hc)
            _log("hit", {"move": move, **x, **hc}, out_hit, 1000 * (time.perf_counter() - t))
        except ValueError as e:
            err = str(e)
    return render_template("demo.html", form=form, out_next=out_next, out_hit=out_hit, err=err, n=0,
                           title="Live demo", **ctx())


@app.route("/feedback", methods=["POST"])
def feedback():
    d = request.form.to_dict()
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "type": d.get("type", "opinion"),
           "rating": d.get("rating"), "text": (d.get("text") or "")[:2000], "page": d.get("page", "")}
    try:
        with open(FEEDBACK, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass
    return redirect(url_for("phase", n=5, sent=1) + "#feedback")


@app.route("/monitoring/live")
@role_required("analyst", "admin")
def live():
    rows = []
    if LIVE_LOG.exists():
        rows = [json.loads(line) for line in LIVE_LOG.read_text(encoding="utf-8").splitlines()[-50:]]
    fb = []
    if FEEDBACK.exists():
        fb = [json.loads(line) for line in FEEDBACK.read_text(encoding="utf-8").splitlines()[-20:]]
    lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
    return jsonify({"requests_logged": len(rows), "mean_latency_ms": round(sum(lat) / len(lat), 2) if lat else None,
                    "last": rows[-10:], "feedback": fb})


if __name__ == "__main__":
    app.run(debug=True)
