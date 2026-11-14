"""Activity 2 - database model of the UMBRAL information system (players and accounts).

Entities (see docs/system/database.md for the diagram):
  User            people who sign in to the system (role: player, analyst, admin)
  Player          a game account / player profile (owned by a User of role player, or created by staff)
  SavedGame       a save slot of a player (world, stars, relics, diary pages...)
  PlaySession     one play session uploaded for a player, with the ordered list of combat actions
  Record          a leaderboard record (Waves per difficulty, Pantheon, the five in a row)
  Inference       every call to the Machine Learning model (date, user, model version, input, result, latency)
  Recommendation  the decision the model supports: change (or keep) the difficulty of a player
  Ticket          incidents, account problems and user feedback
  ErrorLog        unexpected server errors (monitoring)
  AuditEvent      security-relevant changes (role changes, logins locked, recommendations applied)

The same code runs on SQLite (local development, tests) and on PostgreSQL (Supabase, production).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()

ROLES = ("player", "analyst", "admin")
ROLE_LABEL = {"player": "Player", "analyst": "Analyst", "admin": "Administrator"}
CHARACTERS = {"jay": "JAY (archer)", "jake": "JAKE (archer)"}
DIFFS = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Legendary"}
WORLDS = {1: "I Lake", 2: "II Forest", 3: "III Volcano", 4: "IV City", 5: "V Sea", 6: "VI Threshold"}
MODES = {"story": "Story", "ole": "Waves", "pan": "Pantheon"}
OUTCOMES = {"won": "Won", "lost": "Lost", "quit": "Quit"}
SESSION_STATUS = ("registered", "analyzed", "reviewed")
REC_STATUS = ("pending", "approved", "rejected", "applied")
RECORD_MODES = {"ole0": "Waves · Easy", "ole1": "Waves · Normal", "ole2": "Waves · Hard", "ole3": "Waves · Legendary",
                "panteon": "Pantheon · bosses beaten", "pan_t": "Pantheon · time (s)", "rush": "Five in a row · time (s)"}
TIME_MODES = ("pan_t", "rush")                       # lower is better
TICKET_CATEGORY = {"bug": "Bug in the game", "account": "Account or saved game", "model": "Model / unfair boss",
                   "feedback": "Opinion"}
TICKET_SEVERITY = {"low": "Low", "medium": "Medium", "high": "High"}
TICKET_STATUS = ("open", "in_progress", "resolved", "closed")
TICKET_FLOW = {"open": ("in_progress", "closed"), "in_progress": ("resolved", "open"),
               "resolved": ("closed", "in_progress"), "closed": ()}


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)      # stored as naive UTC on both engines


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(32), unique=True, nullable=False, index=True)
    display_name = db.Column(db.String(40), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), nullable=False, default="player")
    active = db.Column(db.Boolean, nullable=False, default=True)
    failed_logins = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    last_login = db.Column(db.DateTime)

    player = db.relationship("Player", back_populates="owner", uselist=False, foreign_keys="Player.owner_id")

    def set_password(self, raw: str) -> None:
        self.password_hash = generate_password_hash(raw)          # scrypt with a random salt (Werkzeug default)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)

    @property
    def is_active(self) -> bool:                                  # Flask-Login: disabled users cannot sign in
        return bool(self.active)

    def has(self, *roles: str) -> bool:
        return self.role in roles

    @property
    def staff(self) -> bool:
        return self.role in ("analyst", "admin")


class Player(db.Model):
    __tablename__ = "players"
    id = db.Column(db.Integer, primary_key=True)
    owner_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), unique=True)
    gamer_tag = db.Column(db.String(20), unique=True, nullable=False, index=True)
    character = db.Column(db.String(8), nullable=False, default="jay")
    difficulty = db.Column(db.Integer, nullable=False, default=1)
    country = db.Column(db.String(40))
    status = db.Column(db.String(10), nullable=False, default="active")     # active / suspended
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"))

    owner = db.relationship("User", back_populates="player", foreign_keys=[owner_id])
    saves = db.relationship("SavedGame", back_populates="player", cascade="all, delete-orphan",
                            order_by="SavedGame.slot")
    sessions = db.relationship("PlaySession", back_populates="player", cascade="all, delete-orphan",
                               order_by="PlaySession.played_at.desc()")
    records = db.relationship("Record", back_populates="player", cascade="all, delete-orphan")
    recommendations = db.relationship("Recommendation", back_populates="player", cascade="all, delete-orphan",
                                      order_by="Recommendation.created_at.desc()")


class SavedGame(db.Model):
    __tablename__ = "saved_games"
    __table_args__ = (db.UniqueConstraint("player_id", "slot", name="uq_save_slot"),)
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    slot = db.Column(db.Integer, nullable=False)                 # 0 = AUTO, 1..24 = windows
    world = db.Column(db.Integer, nullable=False, default=1)
    play_minutes = db.Column(db.Integer, nullable=False, default=0)
    stars = db.Column(db.Integer, nullable=False, default=0)
    relics = db.Column(db.Integer, nullable=False, default=0)
    diary_pages = db.Column(db.Integer, nullable=False, default=0)
    difficulty = db.Column(db.Integer, nullable=False, default=1)
    updated_at = db.Column(db.DateTime, nullable=False, default=now, onupdate=now)

    player = db.relationship("Player", back_populates="saves")


class PlaySession(db.Model):
    __tablename__ = "play_sessions"
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    played_at = db.Column(db.DateTime, nullable=False, default=now, index=True)
    mode = db.Column(db.String(8), nullable=False, default="story")
    world = db.Column(db.Integer, nullable=False, default=1)
    difficulty = db.Column(db.Integer, nullable=False, default=1)
    minutes = db.Column(db.Integer, nullable=False)
    deaths = db.Column(db.Integer, nullable=False, default=0)
    outcome = db.Column(db.String(8), nullable=False, default="lost")
    boss_fight = db.Column(db.Boolean, nullable=False, default=True)
    hp_end = db.Column(db.Integer, nullable=False, default=3)
    enemy_distance = db.Column(db.Integer)                         # typical px to the nearest enemy (optional)
    actions = db.Column(db.Text, nullable=False)                   # "shoot,shoot,dodge,..." oldest first
    notes = db.Column(db.String(300))
    status = db.Column(db.String(12), nullable=False, default="registered")
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    # filled by the model analysis
    predictability = db.Column(db.Float)          # prequential accuracy of the next-action model on this session
    baseline = db.Column(db.Float)                # accuracy of always guessing the player's most common action
    risk = db.Column(db.Float)                    # mean P(boss attack hits) in this player's usual situation
    predicted_next = db.Column(db.String(8))
    predicted_p = db.Column(db.Float)

    player = db.relationship("Player", back_populates="sessions")

    @property
    def action_list(self) -> list[str]:
        return [a for a in (self.actions or "").split(",") if a]

    @property
    def deaths_per_10(self) -> float:
        return 10.0 * self.deaths / max(1, self.minutes)


class Record(db.Model):
    __tablename__ = "records"
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    mode = db.Column(db.String(8), nullable=False, index=True)
    value = db.Column(db.Integer, nullable=False)
    achieved_at = db.Column(db.DateTime, nullable=False, default=now)
    verified = db.Column(db.Boolean, nullable=False, default=False)
    verified_by = db.Column(db.Integer, db.ForeignKey("users.id"))

    player = db.relationship("Player", back_populates="records")


class Inference(db.Model):
    __tablename__ = "inferences"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=now, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    player_id = db.Column(db.Integer, db.ForeignKey("players.id", ondelete="SET NULL"))
    session_id = db.Column(db.Integer, db.ForeignKey("play_sessions.id", ondelete="SET NULL"))
    kind = db.Column(db.String(16), nullable=False)               # session_profile / next_action / hit
    model_name = db.Column(db.String(80), nullable=False)
    model_version = db.Column(db.String(60), nullable=False)      # "<source> <trained_at>" from models/model_card.json
    input_json = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(120), nullable=False)            # short human result ("shoot 61%")
    output_json = db.Column(db.Text, nullable=False)
    latency_ms = db.Column(db.Float, nullable=False)
    n_checked = db.Column(db.Integer)             # predictions compared with the real next action
    n_correct = db.Column(db.Integer)

    user = db.relationship("User")

    def output(self) -> dict:
        return json.loads(self.output_json or "{}")


class Recommendation(db.Model):
    __tablename__ = "recommendations"
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey("players.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = db.Column(db.Integer, db.ForeignKey("play_sessions.id", ondelete="SET NULL"))
    inference_id = db.Column(db.Integer, db.ForeignKey("inferences.id", ondelete="SET NULL"))
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    current_difficulty = db.Column(db.Integer, nullable=False)
    suggested_difficulty = db.Column(db.Integer, nullable=False)
    action = db.Column(db.String(8), nullable=False)              # lower / keep / raise
    reason = db.Column(db.String(400), nullable=False)
    status = db.Column(db.String(10), nullable=False, default="pending")
    decided_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    decided_at = db.Column(db.DateTime)
    decision_note = db.Column(db.String(300))
    # outcome tracking: the next analyzed session after the change is applied
    before_deaths10 = db.Column(db.Float)
    after_deaths10 = db.Column(db.Float)
    after_session_id = db.Column(db.Integer, db.ForeignKey("play_sessions.id", ondelete="SET NULL"))
    effective = db.Column(db.Boolean)

    player = db.relationship("Player", back_populates="recommendations")
    session = db.relationship("PlaySession", foreign_keys=[session_id])
    decider = db.relationship("User", foreign_keys=[decided_by])


class Ticket(db.Model):
    __tablename__ = "tickets"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=now)
    updated_at = db.Column(db.DateTime, nullable=False, default=now, onupdate=now)
    created_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    player_id = db.Column(db.Integer, db.ForeignKey("players.id", ondelete="SET NULL"))
    category = db.Column(db.String(10), nullable=False)
    severity = db.Column(db.String(8), nullable=False, default="medium")
    title = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer)                                # 1..5, for opinions
    status = db.Column(db.String(12), nullable=False, default="open")
    assigned_to = db.Column(db.Integer, db.ForeignKey("users.id"))
    resolution = db.Column(db.String(500))

    author = db.relationship("User", foreign_keys=[created_by])
    assignee = db.relationship("User", foreign_keys=[assigned_to])
    player = db.relationship("Player")


class ErrorLog(db.Model):
    __tablename__ = "error_log"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=now, index=True)
    path = db.Column(db.String(200))
    method = db.Column(db.String(8))
    user_id = db.Column(db.Integer)
    error_type = db.Column(db.String(80))
    message = db.Column(db.String(500))
    trace = db.Column(db.Text)


class AuditEvent(db.Model):
    __tablename__ = "audit_events"
    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, nullable=False, default=now, index=True)
    user_id = db.Column(db.Integer)
    event = db.Column(db.String(40), nullable=False)
    detail = db.Column(db.String(400))


def audit(event: str, detail: str = "", user_id: int | None = None) -> None:
    db.session.add(AuditEvent(event=event, detail=detail[:400], user_id=user_id))
