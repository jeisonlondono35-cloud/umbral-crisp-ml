"""Activity 2 - the information system mounted on the Activity 1 Flask app.

init_system(app) configures the database, sign-in, CSRF, error logging and the first administrator.
Configuration only through environment variables (see .env.example; no secrets in the repository):
  DATABASE_URL     PostgreSQL connection (Supabase). Without it a local SQLite file is used (development).
  SECRET_KEY       signs the session cookie. Required in production.
  ADMIN_USERNAME / ADMIN_PASSWORD   creates the first administrator if there is none.
"""
from __future__ import annotations

import os
import secrets
import traceback
from datetime import timedelta
from pathlib import Path

import click
from flask import Flask, request
from flask_login import current_user

from . import auth, views
from .models import ErrorLog, Player, User, audit, db

ROOT = Path(__file__).resolve().parents[2]


def _db_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        (ROOT / "instance").mkdir(exist_ok=True)
        return f"sqlite:///{(ROOT / 'instance' / 'umbral_system.db').as_posix()}"
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def init_system(app: Flask) -> None:
    prod = bool(os.environ.get("RENDER") or os.environ.get("PRODUCTION"))
    warnings = []
    key = os.environ.get("SECRET_KEY")
    if not key:
        # never take the public documentation down for a missing variable: start, but say it loudly
        key = secrets.token_hex(32)              # sessions end when the server restarts
        if prod:
            warnings.append("SECRET_KEY is not set: sign-ins end at every restart (Render -> Environment).")
    if prod and not os.environ.get("DATABASE_URL"):
        warnings.append("DATABASE_URL is not set: using a temporary SQLite file, data is LOST at every deploy.")
    for w in warnings:
        print("CONFIG WARNING:", w)
    app.config["CONFIG_WARNINGS"] = warnings
    url = _db_url()
    engine_opts = {"pool_pre_ping": True, "pool_recycle": 280}
    if url.startswith("postgresql+psycopg"):
        # Supabase's pooler (port 6543, transaction mode) does not keep prepared statements
        engine_opts["connect_args"] = {"prepare_threshold": None, "connect_timeout": 10}
    app.config.update(SECRET_KEY=key, SQLALCHEMY_DATABASE_URI=url, SQLALCHEMY_ENGINE_OPTIONS=engine_opts,
                      SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", SESSION_COOKIE_SECURE=prod,
                      REMEMBER_COOKIE_HTTPONLY=True, PERMANENT_SESSION_LIFETIME=timedelta(hours=8))
    db.init_app(app)
    auth.login_manager.init_app(app)
    app.register_blueprint(auth.bp)
    app.register_blueprint(views.bp)
    app.before_request(auth.csrf_protect)
    app.jinja_env.globals["csrf_token"] = auth.csrf_token

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if request.path.startswith("/app") or request.path in ("/login", "/register"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    with app.app_context():
        db.create_all()
        _bootstrap_admin()

    @app.cli.command("seed-demo")
    @click.option("--analyze/--no-analyze", default=True, help="Analyze part of the demo sessions with the model.")
    def seed_demo(analyze: bool):
        """Create demo players, saved games and sessions (from the simulated players of Activity 1)."""
        from .seed import seed
        print(seed(analyze=analyze))

    @app.cli.command("create-user")
    @click.argument("username")
    @click.argument("role", type=click.Choice(["player", "analyst", "admin"]))
    @click.password_option()
    def create_user(username: str, role: str, password: str):
        """Create a user with a role (used to prepare the test accounts for the instructor)."""
        u = User.query.filter_by(username=username.lower()).first() or User(username=username.lower(),
                                                                             display_name=username)
        u.role = role
        u.active = True
        u.set_password(password)
        db.session.add(u)
        db.session.flush()
        if role == "player" and not u.player:
            db.session.add(Player(owner_id=u.id, gamer_tag=username[:20], character="jay", difficulty=1))
        audit("user_created_cli", f"{username} as {role}")
        db.session.commit()
        print(f"User {username} ({role}) ready.")


def _bootstrap_admin() -> None:
    name = (os.environ.get("ADMIN_USERNAME") or "").strip().lower()
    pw = os.environ.get("ADMIN_PASSWORD") or ""
    if not name or not pw or User.query.filter_by(role="admin").first():
        return
    u = User.query.filter_by(username=name).first() or User(username=name, display_name="Administrator")
    u.role, u.active = "admin", True
    u.set_password(pw)
    db.session.add(u)
    audit("admin_bootstrap", name)
    db.session.commit()


def log_error(e: Exception) -> None:
    """Called by the app's error handler: keep every unexpected error in the database (monitoring)."""
    try:
        db.session.rollback()
        db.session.add(ErrorLog(path=request.path[:200], method=request.method,
                                user_id=current_user.id if current_user.is_authenticated else None,
                                error_type=type(e).__name__, message=str(e)[:500],
                                trace=traceback.format_exc()[-8000:]))
        db.session.commit()
    except Exception:                       # the database itself may be the problem
        traceback.print_exc()
