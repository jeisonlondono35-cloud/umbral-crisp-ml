"""Activity 2 - operational modules (all of them need a signed-in user; see auth.py for the roles).

Process 1  Player accounts      players, saved games (CRUD) .......................... player (own) / analyst / admin
Process 2  Play sessions + ML   register -> analyze with the model -> recommendation -> decision -> applied
Process 3  Records              submit -> verify -> leaderboard
Process 4  Support              ticket open -> in progress -> resolved -> closed
Management                      reports and indicators, monitoring of the model, users and roles (admin)
"""
from __future__ import annotations

import csv
import io
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from flask import Blueprint, Response, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import func

from src.umbral_ml.config import ACTIONS, ACTION_LABEL

from . import ml_service as ML
from .auth import TAG_RE, can_see_player, player_or_403, role_required
from .models import (CHARACTERS, DIFFS, MODES, OUTCOMES, RECORD_MODES, ROLE_LABEL, ROLES, TICKET_CATEGORY,
                     TICKET_FLOW, TICKET_SEVERITY, TIME_MODES, WORLDS, AuditEvent, ErrorLog, Inference, PlaySession,
                     Player, Record, Recommendation, SavedGame, Ticket, User, audit, db, now)

bp = Blueprint("ops", __name__, url_prefix="/app")
STAFF = ("analyst", "admin")


@bp.app_context_processor
def inject():
    pending = 0
    if current_user.is_authenticated and current_user.staff:
        pending = Recommendation.query.filter_by(status="pending").count()
    return {"DIFFS": DIFFS, "WORLDS": WORLDS, "MODES": MODES, "OUTCOMES": OUTCOMES, "CHARACTERS": CHARACTERS,
            "RECORD_MODES": RECORD_MODES, "ROLE_LABEL": ROLE_LABEL, "TICKET_CATEGORY": TICKET_CATEGORY,
            "TICKET_SEVERITY": TICKET_SEVERITY, "AL": ACTION_LABEL, "ACTIONS": ACTIONS, "pending_recs": pending}


class Invalid(ValueError):
    pass


def num(name: str, lo: int, hi: int, label: str, default=None, optional=False) -> int | None:
    raw = (request.form.get(name) or "").strip()
    if raw == "":
        if optional:
            return None
        if default is not None:
            return default
        raise Invalid(f"{label} is required.")
    try:
        v = int(raw)
    except ValueError:
        raise Invalid(f"{label} must be a whole number.")
    if not lo <= v <= hi:
        raise Invalid(f"{label} must be between {lo} and {hi}.")
    return v


def choice(name: str, options, label: str) -> str:
    v = (request.form.get(name) or "").strip()
    if v not in options:
        raise Invalid(f"Choose a valid {label}.")
    return v


def my_player() -> Player | None:
    return Player.query.filter_by(owner_id=current_user.id).first()


# ------------------------------------------------------------------------------------------ dashboard
@bp.route("/")
@login_required
def dashboard():
    if not current_user.staff:
        p = my_player()
        return render_template("system/dash_player.html", p=p, title="My account")
    week = now() - timedelta(days=7)
    kpi = {"players": Player.query.count(), "sessions": PlaySession.query.count(),
           "sessions_week": PlaySession.query.filter(PlaySession.played_at >= week).count(),
           "to_analyze": PlaySession.query.filter_by(status="registered").count(),
           "pending": Recommendation.query.filter_by(status="pending").count(),
           "open_tickets": Ticket.query.filter(Ticket.status.in_(("open", "in_progress"))).count(),
           "unverified": Record.query.filter_by(verified=False).count()}
    acc = db.session.query(func.sum(Inference.n_correct), func.sum(Inference.n_checked)).filter(
        Inference.kind == "session_profile").one()
    kpi["live_acc"] = (acc[0] / acc[1]) if acc[1] else None
    queue = PlaySession.query.filter_by(status="registered").order_by(PlaySession.played_at).limit(8).all()
    recs = Recommendation.query.filter_by(status="pending").order_by(Recommendation.created_at).limit(8).all()
    return render_template("system/dash_staff.html", kpi=kpi, queue=queue, recs=recs, title="Dashboard")


# -------------------------------------------------------------------------------------------- players
@bp.route("/players")
@role_required(*STAFF)
def players():
    q = (request.args.get("q") or "").strip()
    query = Player.query
    if q:
        query = query.filter(Player.gamer_tag.ilike(f"%{q}%"))
    rows = query.order_by(Player.gamer_tag).all()
    stats = dict(db.session.query(PlaySession.player_id, func.count(PlaySession.id)).group_by(PlaySession.player_id).all())
    return render_template("system/players.html", rows=rows, q=q, stats=stats, title="Players")


def _player_form(p: Player, staff: bool) -> None:
    if staff:
        tag = (request.form.get("gamer_tag") or "").strip()
        if not TAG_RE.match(tag):
            raise Invalid("Gamer tag: 3-20 letters, numbers or _.")
        other = Player.query.filter(func.lower(Player.gamer_tag) == tag.lower(), Player.id != (p.id or 0)).first()
        if other:
            raise Invalid("That gamer tag is taken.")
        p.gamer_tag = tag
        p.status = choice("status", ("active", "suspended"), "status")
        if p.id is None:                                     # difficulty changes later only through the ML process
            p.difficulty = num("difficulty", 0, 3, "Difficulty", default=1)
    p.character = choice("character", CHARACTERS, "character")
    country = (request.form.get("country") or "").strip()
    if len(country) > 40:
        raise Invalid("Country: at most 40 characters.")
    p.country = country or None


@bp.route("/players/new", methods=["GET", "POST"])
@role_required(*STAFF)
def player_new():
    p = Player(character="jay", difficulty=1, status="active")
    if request.method == "POST":
        try:
            _player_form(p, True)
            p.created_by = current_user.id
            db.session.add(p)
            audit("player_created", p.gamer_tag, current_user.id)
            db.session.commit()
            flash("Player created.", "ok")
            return redirect(url_for("ops.player", pid=p.id))
        except Invalid as e:
            flash(str(e), "error")
    return render_template("system/player_form.html", p=p, staff=True, title="New player")


@bp.route("/players/<int:pid>")
@login_required
def player(pid: int):
    p = player_or_403(pid)
    best = {}
    for r in p.records:
        cur = best.get(r.mode)
        better = cur is None or (r.value < cur.value if r.mode in TIME_MODES else r.value > cur.value)
        if better:
            best[r.mode] = r
    analyzed = [s for s in p.sessions if s.status != "registered" and s.predictability is not None]
    prof = None
    if analyzed:
        allacts = Counter(a for s in analyzed for a in s.action_list)
        tot = sum(allacts.values())
        prof = {"acc": statistics.mean(s.predictability for s in analyzed),
                "base": statistics.mean(s.baseline or 0 for s in analyzed),
                "risk": statistics.mean(s.risk or 0 for s in analyzed),
                "share": {a: allacts[a] / tot for a in ACTIONS}, "n": len(analyzed)}
    return render_template("system/player.html", p=p, best=best, prof=prof, title=p.gamer_tag)


@bp.route("/players/<int:pid>/edit", methods=["GET", "POST"])
@login_required
def player_edit(pid: int):
    p = player_or_403(pid)
    if request.method == "POST":
        try:
            _player_form(p, current_user.staff)
            audit("player_edited", p.gamer_tag, current_user.id)
            db.session.commit()
            flash("Player saved.", "ok")
            return redirect(url_for("ops.player", pid=p.id))
        except Invalid as e:
            db.session.rollback()
            flash(str(e), "error")
    return render_template("system/player_form.html", p=p, staff=current_user.staff, title="Edit player")


@bp.route("/players/<int:pid>/delete", methods=["POST"])
@role_required("admin")
def player_delete(pid: int):
    p = db.session.get(Player, pid) or abort(404)
    if (request.form.get("confirm") or "") != p.gamer_tag:
        flash("Type the gamer tag to confirm the deletion.", "error")
        return redirect(url_for("ops.player", pid=pid))
    audit("player_deleted", p.gamer_tag, current_user.id)
    db.session.delete(p)
    db.session.commit()
    flash("Player and all their data deleted.", "ok")
    return redirect(url_for("ops.players"))


# ---------------------------------------------------------------------------------------- saved games
def _save_form(s: SavedGame) -> None:
    s.slot = num("slot", 0, 24, "Slot")
    s.world = num("world", 1, 6, "World")
    s.play_minutes = num("play_minutes", 0, 100000, "Play time (minutes)")
    s.stars = num("stars", 0, 80, "Stars")
    s.relics = num("relics", 0, 5, "Relics")
    s.diary_pages = num("diary_pages", 0, 15, "Diary pages")
    s.difficulty = num("difficulty", 0, 3, "Difficulty")
    # business rules of the game: one relic per world beaten, and stars are 16 per world
    if s.relics > s.world:
        raise Invalid("Relics cannot be more than the world reached (one relic per world).")
    if s.stars > 16 * s.world:
        raise Invalid(f"In world {s.world} a player can have at most {16 * s.world} stars.")
    clash = SavedGame.query.filter(SavedGame.player_id == s.player_id, SavedGame.slot == s.slot,
                                   SavedGame.id != (s.id or 0)).first()
    if clash:
        raise Invalid(f"Slot {s.slot} is already used by another saved game of this player.")


@bp.route("/players/<int:pid>/saves/new", methods=["GET", "POST"])
@login_required
def save_new(pid: int):
    p = player_or_403(pid)
    s = SavedGame(player_id=p.id, slot=0, world=1, play_minutes=0, stars=0, relics=0, diary_pages=0,
                  difficulty=p.difficulty)
    if request.method == "POST":
        try:
            _save_form(s)
            db.session.add(s)
            db.session.commit()
            flash("Saved game created.", "ok")
            return redirect(url_for("ops.player", pid=p.id) + "#saves")
        except Invalid as e:
            db.session.rollback()
            flash(str(e), "error")
    return render_template("system/save_form.html", p=p, s=s, title="Saved game")


@bp.route("/saves/<int:sid>/edit", methods=["GET", "POST"])
@login_required
def save_edit(sid: int):
    s = db.session.get(SavedGame, sid) or abort(404)
    p = player_or_403(s.player_id)
    if request.method == "POST":
        try:
            _save_form(s)
            db.session.commit()
            flash("Saved game updated.", "ok")
            return redirect(url_for("ops.player", pid=p.id) + "#saves")
        except Invalid as e:
            db.session.rollback()
            flash(str(e), "error")
    return render_template("system/save_form.html", p=p, s=s, title="Saved game")


@bp.route("/saves/<int:sid>/delete", methods=["POST"])
@login_required
def save_delete(sid: int):
    s = db.session.get(SavedGame, sid) or abort(404)
    p = player_or_403(s.player_id)
    db.session.delete(s)
    db.session.commit()
    flash("Saved game deleted.", "ok")
    return redirect(url_for("ops.player", pid=p.id) + "#saves")


# ------------------------------------------------------------------------------------------ sessions
@bp.route("/sessions")
@login_required
def sessions():
    st = request.args.get("status", "")
    mode = request.args.get("mode", "")
    query = PlaySession.query
    if not current_user.staff:
        p = my_player()
        query = query.filter(PlaySession.player_id == (p.id if p else -1))
    if st in ("registered", "analyzed", "reviewed"):
        query = query.filter_by(status=st)
    if mode in MODES:
        query = query.filter_by(mode=mode)
    rows = query.order_by(PlaySession.played_at.desc()).limit(300).all()
    return render_template("system/sessions.html", rows=rows, st=st, mode=mode, title="Play sessions")


def _session_form(s: PlaySession) -> None:
    s.mode = choice("mode", MODES, "mode")
    s.world = num("world", 1, 6, "World")
    s.difficulty = num("difficulty", 0, 3, "Difficulty")
    s.minutes = num("minutes", 1, 600, "Minutes played")
    s.deaths = num("deaths", 0, 500, "Deaths")
    s.outcome = choice("outcome", OUTCOMES, "outcome")
    s.hp_end = num("hp_end", 0, 12, "Hearts at the end")
    s.enemy_distance = num("enemy_distance", 0, 400, "Typical distance to the enemy", optional=True)
    s.boss_fight = bool(request.form.get("boss_fight"))
    raw = request.form.get("played_at") or ""
    try:
        s.played_at = datetime.fromisoformat(raw) if raw else now()
    except ValueError:
        raise Invalid("Date and time of the session is not valid.")
    if s.played_at > now() + timedelta(minutes=5):
        raise Invalid("A session cannot be in the future.")
    if s.outcome == "won" and s.hp_end == 0:
        raise Invalid("A won session must end with at least one heart.")
    try:
        s.actions = ",".join(ML.parse_actions(request.form.get("actions", "")))
    except ValueError as e:
        raise Invalid(str(e))
    notes = (request.form.get("notes") or "").strip()
    s.notes = notes[:300] or None


@bp.route("/players/<int:pid>/sessions/new", methods=["GET", "POST"])
@login_required
def session_new(pid: int):
    p = player_or_403(pid)
    if p.status != "active":
        flash("This player account is suspended: it cannot upload sessions.", "error")
        return redirect(url_for("ops.player", pid=p.id))
    s = PlaySession(player_id=p.id, mode="story", world=1, difficulty=p.difficulty, minutes=10, deaths=0,
                    outcome="lost", hp_end=3, boss_fight=True, actions="")
    if request.method == "POST":
        try:
            _session_form(s)
            s.created_by = current_user.id
            db.session.add(s)
            db.session.commit()
            flash("Session registered. An analyst will analyze it with the model.", "ok")
            return redirect(url_for("ops.session", sid=s.id))
        except Invalid as e:
            db.session.rollback()
            flash(str(e), "error")
    return render_template("system/session_form.html", p=p, s=s, title="New play session")


@bp.route("/sessions/<int:sid>")
@login_required
def session(sid: int):
    s = db.session.get(PlaySession, sid) or abort(404)
    player_or_403(s.player_id)
    infs = Inference.query.filter_by(session_id=s.id).order_by(Inference.created_at.desc()).all()
    prof = next((i for i in infs if i.kind == "session_profile"), None)
    hit = next((i for i in infs if i.kind == "hit"), None)
    recs = Recommendation.query.filter_by(session_id=s.id).all()
    share = Counter(s.action_list)
    return render_template("system/session.html", s=s, prof=prof.output() if prof else None, prof_row=prof,
                           hit=hit.output() if hit else None, hit_row=hit, recs=recs, share=share,
                           hit_in=json.loads(hit.input_json) if hit else {},
                           rule=ML.DECISION_RULE, title=f"Session {s.id}")


@bp.route("/sessions/<int:sid>/edit", methods=["GET", "POST"])
@login_required
def session_edit(sid: int):
    s = db.session.get(PlaySession, sid) or abort(404)
    p = player_or_403(s.player_id)
    if s.status != "registered":
        flash("Only sessions that were not analyzed yet can be edited (the analysis must match the data).", "error")
        return redirect(url_for("ops.session", sid=s.id))
    if request.method == "POST":
        try:
            _session_form(s)
            db.session.commit()
            flash("Session updated.", "ok")
            return redirect(url_for("ops.session", sid=s.id))
        except Invalid as e:
            db.session.rollback()
            flash(str(e), "error")
    return render_template("system/session_form.html", p=p, s=s, title="Edit session")


@bp.route("/sessions/<int:sid>/delete", methods=["POST"])
@login_required
def session_delete(sid: int):
    s = db.session.get(PlaySession, sid) or abort(404)
    p = player_or_403(s.player_id)
    if s.status != "registered" and current_user.role != "admin":
        flash("An analyzed session is evidence of a model decision: only an administrator can delete it.", "error")
        return redirect(url_for("ops.session", sid=s.id))
    audit("session_deleted", f"session {s.id} of {p.gamer_tag}", current_user.id)
    db.session.delete(s)
    db.session.commit()
    flash("Session deleted.", "ok")
    return redirect(url_for("ops.player", pid=p.id))


@bp.route("/sessions/<int:sid>/analyze", methods=["POST"])
@role_required(*STAFF)
def session_analyze(sid: int):
    s = db.session.get(PlaySession, sid) or abort(404)
    if s.status == "reviewed":
        flash("This session was already reviewed; its analysis is kept as evidence.", "error")
        return redirect(url_for("ops.session", sid=s.id))
    res = ML.analyze_session(s, current_user.id)
    db.session.commit()
    rec = res["recommendation"]
    flash(f"Analyzed in {res['latency_ms']} ms. " + (f"Recommendation: {rec.action.upper()} difficulty."
                                                       if rec else "No change of difficulty is needed."), "ok")
    return redirect(url_for("ops.session", sid=s.id))


# ----------------------------------------------------------------------------------- recommendations
@bp.route("/recommendations")
@role_required(*STAFF)
def recommendations():
    st = request.args.get("status", "pending")
    query = Recommendation.query
    if st in ("pending", "approved", "rejected", "applied"):
        query = query.filter_by(status=st)
    rows = query.order_by(Recommendation.created_at.desc()).limit(300).all()
    return render_template("system/recommendations.html", rows=rows, st=st, title="Recommendations")


@bp.route("/recommendations/<int:rid>/decide", methods=["POST"])
@role_required(*STAFF)
def recommendation_decide(rid: int):
    rec = db.session.get(Recommendation, rid) or abort(404)
    approve = request.form.get("decision") == "approve"
    note = (request.form.get("note") or "").strip()
    if not approve and not note:
        flash("Write why you reject the recommendation (it helps to improve the model).", "error")
        return redirect(request.referrer or url_for("ops.recommendations"))
    try:
        ML.apply_decision(rec, approve, current_user.id, note)
        db.session.commit()
        flash("Difficulty changed for the player." if approve else "Recommendation rejected.", "ok")
    except ValueError as e:
        db.session.rollback()
        flash(str(e), "error")
    return redirect(request.referrer or url_for("ops.recommendations"))


# -------------------------------------------------------------------------------------------- records
@bp.route("/records")
@login_required
def records():
    mode = request.args.get("mode", "ole1")
    if mode not in RECORD_MODES:
        mode = "ole1"
    rows = Record.query.filter_by(mode=mode, verified=True).all()
    best: dict[int, Record] = {}
    for r in rows:
        cur = best.get(r.player_id)
        if cur is None or (r.value < cur.value if mode in TIME_MODES else r.value > cur.value):
            best[r.player_id] = r
    board = sorted(best.values(), key=lambda r: r.value, reverse=mode not in TIME_MODES)[:20]
    unverified = Record.query.filter_by(verified=False).order_by(Record.achieved_at).all() if current_user.staff else []
    return render_template("system/records.html", board=board, mode=mode, unverified=unverified, title="Records")


@bp.route("/players/<int:pid>/records/new", methods=["POST"])
@login_required
def record_new(pid: int):
    p = player_or_403(pid)
    try:
        mode = choice("mode", RECORD_MODES, "record type")
        lim = {"panteon": (1, 12), "pan_t": (60, 36000), "rush": (60, 36000)}.get(mode, (1, 999))
        value = num("value", lim[0], lim[1], "Value")
        r = Record(player_id=p.id, mode=mode, value=value, verified=False)
        db.session.add(r)
        db.session.commit()
        flash("Record sent. It appears on the leaderboard when an analyst verifies it.", "ok")
    except Invalid as e:
        flash(str(e), "error")
    return redirect(url_for("ops.player", pid=p.id) + "#records")


@bp.route("/records/<int:rid>/verify", methods=["POST"])
@role_required(*STAFF)
def record_verify(rid: int):
    r = db.session.get(Record, rid) or abort(404)
    if request.form.get("decision") == "reject":
        audit("record_rejected", f"{r.player.gamer_tag} {r.mode}={r.value}", current_user.id)
        db.session.delete(r)
        flash("Record rejected and removed.", "ok")
    else:
        r.verified, r.verified_by = True, current_user.id
        audit("record_verified", f"{r.player.gamer_tag} {r.mode}={r.value}", current_user.id)
        flash("Record verified.", "ok")
    db.session.commit()
    return redirect(request.referrer or url_for("ops.records"))


@bp.route("/records/<int:rid>/delete", methods=["POST"])
@login_required
def record_delete(rid: int):
    r = db.session.get(Record, rid) or abort(404)
    p = player_or_403(r.player_id)
    if r.verified and not current_user.staff:
        flash("A verified record can only be removed by an analyst.", "error")
    else:
        db.session.delete(r)
        db.session.commit()
        flash("Record removed.", "ok")
    return redirect(url_for("ops.player", pid=p.id) + "#records")


# -------------------------------------------------------------------------------------------- tickets
@bp.route("/tickets")
@login_required
def tickets():
    st = request.args.get("status", "")
    query = Ticket.query if current_user.staff else Ticket.query.filter_by(created_by=current_user.id)
    if st in TICKET_FLOW:
        query = query.filter_by(status=st)
    rows = query.order_by(Ticket.updated_at.desc()).limit(300).all()
    return render_template("system/tickets.html", rows=rows, st=st, title="Support and feedback")


@bp.route("/tickets/new", methods=["GET", "POST"])
@login_required
def ticket_new():
    form = {"category": request.args.get("category", "bug"), "severity": "medium", "title": "", "description": "",
            "rating": "", "player_id": request.args.get("player", "")}
    if request.method == "POST":
        form = {k: (request.form.get(k) or "").strip() for k in form}
        try:
            cat = choice("category", TICKET_CATEGORY, "category")
            sev = choice("severity", TICKET_SEVERITY, "severity")
            if not 5 <= len(form["title"]) <= 120:
                raise Invalid("Title: 5 to 120 characters.")
            if not 10 <= len(form["description"]) <= 4000:
                raise Invalid("Describe what happened (10 to 4000 characters).")
            rating = num("rating", 1, 5, "Score", optional=True)
            if cat == "feedback" and rating is None:
                raise Invalid("An opinion needs a score from 1 to 5.")
            pid = None
            if form["player_id"]:
                p = player_or_403(int(form["player_id"])) if form["player_id"].isdigit() else abort(400)
                pid = p.id
            elif not current_user.staff and my_player():
                pid = my_player().id
            t = Ticket(created_by=current_user.id, category=cat, severity=sev, title=form["title"],
                       description=form["description"], rating=rating, player_id=pid)
            db.session.add(t)
            db.session.commit()
            flash(f"Ticket #{t.id} created. You can follow it here.", "ok")
            return redirect(url_for("ops.ticket", tid=t.id))
        except Invalid as e:
            flash(str(e), "error")
    return render_template("system/ticket_form.html", form=form, title="New ticket")


@bp.route("/tickets/<int:tid>", methods=["GET", "POST"])
@login_required
def ticket(tid: int):
    t = db.session.get(Ticket, tid) or abort(404)
    if not current_user.staff and t.created_by != current_user.id:
        abort(403)
    if request.method == "POST":
        if not current_user.staff:
            abort(403)
        new = request.form.get("status", t.status)
        res = (request.form.get("resolution") or "").strip()
        if new != t.status and new not in TICKET_FLOW[t.status]:
            flash(f"A ticket cannot go from {t.status} to {new}.", "error")
        elif new == "resolved" and not res:
            flash("Write the resolution before marking the ticket as resolved.", "error")
        else:
            t.status = new
            t.resolution = res[:500] or t.resolution
            if request.form.get("assign_me"):
                t.assigned_to = current_user.id
            audit("ticket_updated", f"#{t.id} -> {new}", current_user.id)
            db.session.commit()
            flash("Ticket updated.", "ok")
            return redirect(url_for("ops.ticket", tid=t.id))
    return render_template("system/ticket.html", t=t, flow=TICKET_FLOW, title=f"Ticket #{t.id}")


# ----------------------------------------------------------------------------------- inference history
@bp.route("/inferences")
@login_required
def inferences():
    query = Inference.query
    if not current_user.staff:
        p = my_player()
        query = query.filter(Inference.player_id == (p.id if p else -1))
    kind = request.args.get("kind", "")
    if kind in ("session_profile", "hit"):
        query = query.filter_by(kind=kind)
    rows = query.order_by(Inference.created_at.desc()).limit(300).all()
    return render_template("system/inferences.html", rows=rows, kind=kind, title="Inference history")


# -------------------------------------------------------------------------------------------- reports
def _report_data() -> dict:
    sess = PlaySession.query.all()
    by_diff = defaultdict(list)
    by_world = defaultdict(list)
    for s in sess:
        by_diff[s.difficulty].append(s)
        by_world[s.world].append(s)

    def summ(lst):
        return {"n": len(lst), "deaths10": statistics.mean(s.deaths_per_10 for s in lst) if lst else None,
                "win": sum(s.outcome == "won" for s in lst) / len(lst) if lst else None,
                "acc": statistics.mean(s.predictability for s in lst if s.predictability is not None)
                if any(s.predictability is not None for s in lst) else None}

    recs = Recommendation.query.all()
    rc = Counter(r.status for r in recs)
    eff = [r for r in recs if r.effective is not None]
    tk = Ticket.query.all()
    ratings = [t.rating for t in tk if t.rating]
    return {"by_diff": {d: summ(by_diff[d]) for d in DIFFS}, "by_world": {w: summ(by_world[w]) for w in WORLDS},
            "rec_status": rc, "rec_actions": Counter(r.action for r in recs),
            "effective": (sum(r.effective for r in eff), len(eff)),
            "tickets": Counter(t.status for t in tk), "ticket_cat": Counter(t.category for t in tk),
            "rating": statistics.mean(ratings) if ratings else None, "n_ratings": len(ratings),
            "players_by_diff": Counter(p.difficulty for p in Player.query.all()),
            "chars": Counter(p.character for p in Player.query.all())}


@bp.route("/reports")
@role_required(*STAFF)
def reports():
    return render_template("system/reports.html", d=_report_data(), title="Reports and indicators")


@bp.route("/reports/sessions.csv")
@role_required(*STAFF)
def sessions_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "player", "played_at", "mode", "world", "difficulty", "minutes", "deaths", "outcome", "hp_end",
                "n_actions", "status", "predictability", "baseline", "risk", "predicted_next"])
    for s in PlaySession.query.order_by(PlaySession.played_at).all():
        w.writerow([s.id, s.player.gamer_tag, s.played_at.isoformat(timespec="minutes"), s.mode, s.world,
                    s.difficulty, s.minutes, s.deaths, s.outcome, s.hp_end, len(s.action_list), s.status,
                    s.predictability, s.baseline, s.risk, s.predicted_next])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=umbral_sessions.csv"})


# ------------------------------------------------------------------------------------------ monitoring
def monitoring_data() -> dict:
    infs = Inference.query.order_by(Inference.created_at).all()
    lat = sorted(i.latency_ms for i in infs)
    p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None
    weeks: dict[str, list] = defaultdict(lambda: [0, 0])
    for i in infs:
        if i.kind == "session_profile" and i.n_checked:
            y, w, _ = i.created_at.isocalendar()
            weeks[f"{y}-W{w:02d}"][0] += i.n_correct
            weeks[f"{y}-W{w:02d}"][1] += i.n_checked
    wk = [{"week": k, "acc": c / n, "n": n} for k, (c, n) in sorted(weeks.items())]
    base = statistics.mean(x["acc"] for x in wk[:3]) if wk else None
    for x in wk:
        x["alarm"] = base is not None and x["acc"] < base - 0.05
    since = now() - timedelta(days=30)
    recent = PlaySession.query.filter(PlaySession.played_at >= since, PlaySession.status != "registered").all()
    cnt = Counter(a for s in recent for a in s.action_list)
    tot = sum(cnt.values())
    cur = {a: cnt[a] / tot for a in ACTIONS} if tot else None
    ref = ML.reference_shares()
    psi = ML.psi(ref, cur) if cur and tot >= 200 else None
    versions = db.session.query(Inference.model_version, func.count(Inference.id), func.min(Inference.created_at),
                                func.max(Inference.created_at)).group_by(Inference.model_version).all()
    recs = Recommendation.query.filter(Recommendation.effective.isnot(None)).all()
    return {"n": len(infs), "mean": statistics.mean(lat) if lat else None, "p95": p95,
            "slow": sum(1 for v in lat if v > 1000), "weeks": wk, "base": base, "ref": ref, "cur": cur,
            "n_actions": tot, "psi": psi, "psi_state": ML.psi_state(psi), "versions": versions,
            "effective": (sum(r.effective for r in recs), len(recs)),
            "errors": ErrorLog.query.order_by(ErrorLog.created_at.desc()).limit(20).all(),
            "n_errors": ErrorLog.query.count(), "card": ML.P.card(), "version": ML.model_version()}


@bp.route("/monitoring")
@role_required(*STAFF)
def monitoring():
    return render_template("system/monitoring.html", m=monitoring_data(), title="Model monitoring")


@bp.route("/monitoring.json")
@role_required(*STAFF)
def monitoring_json():
    m = monitoring_data()
    m["errors"] = [{"at": e.created_at.isoformat(), "path": e.path, "type": e.error_type} for e in m["errors"]]
    m["versions"] = [{"version": v, "n": n, "first": a.isoformat(), "last": b.isoformat()} for v, n, a, b in m["versions"]]
    m.pop("card", None)
    return json.dumps(m, default=str), 200, {"Content-Type": "application/json"}


# ------------------------------------------------------------------------------------- users (admin)
@bp.route("/users")
@role_required("admin")
def users():
    rows = User.query.order_by(User.role.desc(), User.username).all()
    events = AuditEvent.query.order_by(AuditEvent.created_at.desc()).limit(40).all()
    names = {u.id: u.username for u in rows}
    return render_template("system/users.html", rows=rows, events=events, names=names, ROLES=ROLES, title="Users and roles")


@bp.route("/users/<int:uid>", methods=["POST"])
@role_required("admin")
def user_update(uid: int):
    u = db.session.get(User, uid) or abort(404)
    if u.id == current_user.id:
        flash("You cannot change your own role or disable yourself (ask another administrator).", "error")
        return redirect(url_for("ops.users"))
    role = request.form.get("role", u.role)
    active = request.form.get("active") == "1"
    if role not in ROLES:
        abort(400)
    if u.role == "admin" and (role != "admin" or not active):
        admins = User.query.filter_by(role="admin", active=True).count()
        if admins <= 1:
            flash("The system must keep at least one active administrator.", "error")
            return redirect(url_for("ops.users"))
    if role != u.role:
        audit("role_changed", f"{u.username}: {u.role} -> {role}", current_user.id)
    if active != u.active:
        audit("user_enabled" if active else "user_disabled", u.username, current_user.id)
    u.role, u.active = role, active
    db.session.commit()
    flash(f"User {u.username} updated.", "ok")
    return redirect(url_for("ops.users"))
