"""Activity 2 - architecture and database diagrams for the report (python tools/gen_system_figures.py).
Drawn with matplotlib (already a dependency), Arial-like font, black lines, like the other report figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"], "font.size": 9})
BLUE, GRAY, LINE = "#e3eefb", "#f2f1ee", "#222222"


def box(ax, x, y, w, h, title, lines=(), fill=GRAY, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=LINE, lw=1))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontweight="bold" if bold else "normal", fontsize=9.5)
    for i, t in enumerate(lines):
        ax.text(x + 0.12, y + h - 0.6 - 0.27 * i, t, ha="left", va="top", fontsize=8)


def arrow(ax, x1, y1, x2, y2, label="", dy=0.12):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=LINE, lw=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center", fontsize=7.5, color="#333")


def architecture():
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    box(ax, 0.2, 4.6, 2.4, 1.9, "Users (browser)", ["Player", "Analyst", "Administrator", "(HTTPS, session cookie)"])
    box(ax, 3.4, 0.4, 5.2, 6.2, "Flask app on Render (gunicorn)", [], fill="#fbfbfa")
    box(ax, 3.7, 4.6, 2.2, 1.6, "Public docs", ["Cover + 6 CRISP-ML", "phases, demo, PDFs"], fill=BLUE)
    box(ax, 6.1, 4.6, 2.3, 1.6, "Auth (auth.py)", ["register / login", "roles, CSRF, lockout", "scrypt hashes"], fill=BLUE)
    box(ax, 3.7, 2.5, 4.7, 1.8, "Operational modules (views.py)", ["players · saved games · sessions · records", "tickets · recommendations · reports",
                                                                   "monitoring · users and roles"], fill=BLUE)
    box(ax, 3.7, 0.6, 4.7, 1.6, "Model service (ml_service.py)", ["features = src/umbral_ml/predict.py (same as training)",
                                                                  "decision rule · inference log · drift (PSI)"], fill=BLUE)
    box(ax, 9.3, 3.9, 2.5, 2.4, "Supabase PostgreSQL", ["users, players, saves", "sessions, records", "inferences, decisions",
                                                        "tickets, errors, audit"])
    box(ax, 9.3, 0.5, 2.5, 2.7, "Activity 1 models", ["models/*.joblib", "model_card.json", "(version = source +", "training date)",
                                                      "weekly retrain: GitHub", "Actions"])
    arrow(ax, 2.6, 5.5, 3.7, 5.4, "pages")
    arrow(ax, 2.6, 5.0, 6.1, 4.9, "sign in", dy=-0.25)
    arrow(ax, 6.0, 4.6, 6.0, 4.3)
    arrow(ax, 6.0, 2.5, 6.0, 2.2, "")
    arrow(ax, 8.4, 3.4, 9.3, 4.6, "SQLAlchemy", dy=0.15)
    arrow(ax, 8.4, 1.4, 9.3, 1.6, "joblib", dy=0.12)
    arrow(ax, 8.4, 1.0, 9.3, 4.0, "inference log", dy=-0.55)
    fig.tight_layout()
    fig.savefig(OUT / "sys_architecture.png", dpi=170)
    plt.close(fig)


def er():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    W = 2.8
    E = {
        "users": (0.2, 6.0, ["id PK", "username UNIQUE", "password_hash (scrypt)", "role", "active", "failed_logins", "locked_until"]),
        "players": (4.1, 6.0, ["id PK", "owner_id FK users UNIQUE", "gamer_tag UNIQUE", "character", "difficulty", "status"]),
        "saved_games": (8.6, 7.0, ["id PK", "player_id FK", "slot (UNIQUE per player)", "world, stars, relics, diary"]),
        "records": (8.6, 4.75, ["id PK", "player_id FK", "mode, value", "verified, verified_by FK"]),
        "tickets": (0.2, 0.7, ["id PK", "created_by FK users", "player_id FK", "category, severity", "status, assigned_to", "rating"]),
        "play_sessions": (4.1, 0.7, ["id PK", "player_id FK", "played_at, mode, world", "difficulty, minutes, deaths",
                                     "outcome, hp_end, actions", "status", "predictability, risk"]),
        "inferences": (8.6, 0.25, ["id PK", "created_at, user_id FK", "session_id FK, player_id FK", "model_name, model_version",
                                   "result, latency_ms", "n_checked, n_correct"]),
        "recommendations": (8.6, 2.75, ["id PK", "player_id FK, session_id FK", "inference_id FK", "current -> suggested", "status, decided_by"]),
    }
    P = {}
    for name, (x, y, cols) in E.items():
        h = 0.55 + 0.27 * len(cols)
        box(ax, x, y, W, h, name, cols, fill=BLUE if name in ("inferences", "recommendations") else GRAY)
        P[name] = (x, y, W, h)

    def side(a, b, label, ya=None, yb=None):          # a's right edge -> b's left edge, elbow in the gap
        xa, y0, _, ha = P[a]
        xb, y1, _, hb = P[b]
        y_a = ya if ya is not None else y0 + ha / 2
        y_b = y1 + hb / 2 if yb is None else yb
        xm = (xa + W + xb) / 2
        ax.plot([xa + W, xm, xm, xb], [y_a, y_a, y_b, y_b], color=LINE, lw=0.9)
        ax.text(xm + 0.08, y_b + 0.08, label, fontsize=7.5, color="#333")

    def down(a, b, label):                            # a's bottom -> b's top
        xa, ya, _, _ = P[a]
        xb, yb, _, hb = P[b]
        ax.plot([xa + W / 2, xb + W / 2], [ya, yb + hb], color=LINE, lw=0.9)
        ax.text(xa + W / 2 + 0.08, (ya + yb + hb) / 2, label, fontsize=7.5, color="#333")

    side("users", "players", "1 : 0..1")
    side("players", "saved_games", "1 : N", ya=P["players"][1] + 1.6)
    side("players", "records", "1 : N", ya=P["players"][1] + 0.6)
    down("players", "play_sessions", "1 : N")
    down("users", "tickets", "1 : N")
    side("play_sessions", "inferences", "1 : N", ya=P["play_sessions"][1] + 0.5)
    side("play_sessions", "recommendations", "1 : N", ya=P["play_sessions"][1] + 1.8)
    ax.text(0.2, 0.2, "Also: error_log and audit_events (monitoring and security).", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "sys_er.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    er()
    print("figures written to", OUT)
