"""Activity 2 - registration, sign-in, sign-out, roles and server-side authorization.

Security decisions:
  * Passwords: Werkzeug generate_password_hash (scrypt + random salt). The plain password is never stored.
  * Public registration ALWAYS creates the role "player": the form has no role field and any extra field
    sent by hand is ignored. Only an administrator can change roles (users_admin in views.py).
  * 5 wrong passwords lock the account for 5 minutes (same rule as the game's own accounts).
  * Every form that changes data carries a CSRF token checked on the server (csrf_protect).
  * Authorization is enforced on the server with @role_required and per-object checks (can_see_player);
    hiding a button is never the only protection.
"""
from __future__ import annotations

import re
import secrets
from datetime import timedelta
from functools import wraps

from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user

from .models import CHARACTERS, Player, User, audit, db, now

bp = Blueprint("auth", __name__)
login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "Please sign in to use the operational modules."

MAX_FAILS = 5
LOCK_MINUTES = 5
USER_RE = re.compile(r"^[a-z0-9_]{3,20}$")
TAG_RE = re.compile(r"^[A-Za-z0-9_]{3,20}$")


@login_manager.user_loader
def load_user(uid: str):
    return db.session.get(User, int(uid))


@login_manager.unauthorized_handler
def unauthorized():
    """The JSON API answers 401 (a program cannot follow a redirect to the sign-in form); pages redirect."""
    if request.path.startswith("/api/"):
        return {"error": "authentication required", "login": url_for("auth.login", _external=True)}, 401
    flash(login_manager.login_message, "error")
    return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))


# ----------------------------------------------------------------------------------------------- CSRF
def csrf_token() -> str:
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_urlsafe(32)
    return session["_csrf"]


def csrf_protect():
    """before_request hook: every POST must carry the session's token (forms) or be a JSON API call."""
    if request.method != "POST" or request.path.startswith("/api/"):
        return None
    sent = request.form.get("_csrf", "")
    if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
        abort(400, description="The form expired or was not sent from this site. Reload the page and try again.")
    return None


# ---------------------------------------------------------------------------------------- decorators
def role_required(*roles: str):
    def deco(fn):
        @wraps(fn)
        @login_required
        def inner(*a, **kw):
            if current_user.role not in roles:
                audit("forbidden", f"{request.method} {request.path} as {current_user.role}", current_user.id)
                db.session.commit()
                abort(403)
            return fn(*a, **kw)
        return inner
    return deco


def can_see_player(p: Player) -> bool:
    return current_user.staff or (p.owner_id is not None and p.owner_id == current_user.id)


def player_or_403(pid: int) -> Player:
    p = db.session.get(Player, pid) or abort(404)
    if not can_see_player(p):
        audit("forbidden", f"player {pid} by user {current_user.id}", current_user.id)
        db.session.commit()
        abort(403)
    return p


# ------------------------------------------------------------------------------------------- routes
def validate_password(pw: str, username: str) -> str | None:
    if len(pw) < 8:
        return "The password needs at least 8 characters."
    if pw.lower() == username.lower() or pw.isdigit() or pw.isalpha():
        return "Use letters and numbers, and not the user name."
    return None


@bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("ops.dashboard"))
    form = {"username": "", "display_name": "", "gamer_tag": "", "character": "jay"}
    if request.method == "POST":
        form = {k: (request.form.get(k) or "").strip() for k in form}
        pw, pw2 = request.form.get("password", ""), request.form.get("password2", "")
        u = form["username"].lower()
        err = None
        if not USER_RE.match(u):
            err = "User name: 3-20 characters, lowercase letters, numbers or _."
        elif not 1 <= len(form["display_name"]) <= 40:
            err = "Write the name other people will see (1-40 characters)."
        elif not TAG_RE.match(form["gamer_tag"]):
            err = "Gamer tag: 3-20 letters, numbers or _."
        elif form["character"] not in CHARACTERS:
            err = "Choose JAY or JAKE."
        elif pw != pw2:
            err = "The two passwords are different."
        else:
            err = validate_password(pw, u)
        if not err and User.query.filter_by(username=u).first():
            err = "That user name is taken."
        if not err and Player.query.filter(db.func.lower(Player.gamer_tag) == form["gamer_tag"].lower()).first():
            err = "That gamer tag is taken."
        if err:
            flash(err, "error")
        else:
            user = User(username=u, display_name=form["display_name"], role="player")   # never from the form
            user.set_password(pw)
            db.session.add(user)
            db.session.flush()
            db.session.add(Player(owner_id=user.id, gamer_tag=form["gamer_tag"], character=form["character"],
                                  difficulty=1, created_by=user.id))
            audit("register", u, user.id)
            db.session.commit()
            login_user(user)
            session.permanent = True
            flash("Welcome! Your player account was created.", "ok")
            return redirect(url_for("ops.dashboard"))
    return render_template("system/register.html", form=form, CHARACTERS=CHARACTERS, title="Create account")


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("ops.dashboard"))
    if request.method == "POST":
        u = (request.form.get("username") or "").strip().lower()
        pw = request.form.get("password") or ""
        user = User.query.filter_by(username=u).first()
        if user and user.locked_until and user.locked_until > now():
            mins = max(1, int((user.locked_until - now()).total_seconds() // 60) + 1)
            flash(f"Too many wrong passwords. Try again in {mins} min.", "error")
        elif user and user.active and user.check_password(pw):
            user.failed_logins, user.locked_until, user.last_login = 0, None, now()
            audit("login", u, user.id)
            db.session.commit()
            session.clear()                                 # new session id after login (no session fixation)
            login_user(user)
            session.permanent = True
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("ops.dashboard"))
        else:
            if user:
                user.failed_logins += 1
                if user.failed_logins >= MAX_FAILS:
                    user.locked_until = now() + timedelta(minutes=LOCK_MINUTES)
                    user.failed_logins = 0
                    audit("account_locked", u, user.id)
                db.session.commit()
            flash("Wrong user name or password.", "error")      # same message: do not reveal which one
    return render_template("system/login.html", title="Sign in")


@bp.route("/logout", methods=["POST"])
@login_required
def logout():
    audit("logout", current_user.username, current_user.id)
    db.session.commit()
    logout_user()
    session.clear()
    flash("You signed out.", "ok")
    return redirect(url_for("home"))
