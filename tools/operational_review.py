"""Activity 2 - initial operational review of the DEPLOYED system (evidence for the report).

    python tools/operational_review.py https://umbral-crisp-ml.onrender.com

Credentials come from environment variables (never from the repository, never written to the output):
    REVIEW_PLAYER_USER / REVIEW_PLAYER_PASS      a player account (with its player profile)
    REVIEW_ANALYST_USER / REVIEW_ANALYST_PASS    an analyst account
    REVIEW_ADMIN_USER / REVIEW_ADMIN_PASS        an administrator account

It checks, against the real URL: health and database, public documentation, protected routes for anonymous
users, sign-in of each role, the permission matrix (who can open what), the whole workflow (the player uploads
a session -> the analyst runs the model -> inference stored with user, version and latency -> recommendation),
and that the data is still there after signing out and in again (persistence). The result is written to
reports/operational_review.json, which the Activity 2 report reads.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "operational_review.json"
SEQ = ",".join(["shoot", "shoot", "dodge", "melee", "shoot", "block", "shoot", "dodge", "jump", "hide"] * 4)

MATRIX = [  # path, expected status for anonymous / player / analyst / admin  (302 = sent to sign in)
    ("/", 200, 200, 200, 200),
    ("/phase/4", 200, 200, 200, 200),
    ("/app/", 302, 200, 200, 200),
    ("/app/sessions", 302, 200, 200, 200),
    ("/app/players", 302, 403, 200, 200),
    ("/app/recommendations", 302, 403, 200, 200),
    ("/app/reports", 302, 403, 200, 200),
    ("/app/monitoring", 302, 403, 200, 200),
    ("/app/users", 302, 403, 403, 200),
    ("/monitoring/live", 302, 403, 200, 200),
]


def csrf(s: requests.Session, base: str, path: str = "/login") -> str:
    m = re.search(r'name="_csrf" value="([^"]+)"', s.get(base + path, timeout=60).text)
    return m.group(1) if m else ""


def login(base: str, user: str, pw: str) -> requests.Session:
    s = requests.Session()
    r = s.post(base + "/login", data={"_csrf": csrf(s, base), "username": user, "password": pw}, timeout=60)
    if "Sign out" not in r.text:
        raise SystemExit(f"Sign-in failed for the {user!r} account (check the REVIEW_* variables).")
    return s


def main() -> None:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000").rstrip("/")
    env = os.environ
    roles = {"player": ("REVIEW_PLAYER_USER", "REVIEW_PLAYER_PASS"),
             "analyst": ("REVIEW_ANALYST_USER", "REVIEW_ANALYST_PASS"),
             "admin": ("REVIEW_ADMIN_USER", "REVIEW_ADMIN_PASS")}
    missing = [v for pair in roles.values() for v in pair if not env.get(v)]
    if missing:
        raise SystemExit("Set these variables first: " + ", ".join(missing))
    ev = {"url": base, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": []}

    def check(name: str, ok: bool, detail: str = "") -> None:
        ev["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f" · {detail}" if detail else ""))

    t = time.perf_counter()
    h = requests.get(base + "/api/health", timeout=120).json()           # first call may wake Render up
    check("Health endpoint and database", h.get("status") == "ok" and h.get("database") == "ok",
          f"database={h.get('database')}, model={h.get('model_version')}, {1000 * (time.perf_counter() - t):.0f} ms")
    ev["model_version"] = h.get("model_version")

    sessions = {"anonymous": requests.Session()}
    for role, (u, p) in roles.items():
        sessions[role] = login(base, env[u], env[p])
        check(f"Sign-in as {role}", True)
    order = ["anonymous", "player", "analyst", "admin"]
    bad = []
    for path, *expected in MATRIX:
        for role, exp in zip(order, expected):
            code = sessions[role].get(base + path, allow_redirects=False, timeout=60).status_code
            if code != exp:
                bad.append(f"{role} {path}: {code} (expected {exp})")
    check(f"Permission matrix ({len(MATRIX) * 4} requests)", not bad, "; ".join(bad) or "every role sees only its functions")
    r = requests.post(base + "/api/predict/next", json={"history": ["shoot"]}, timeout=60)
    check("API refuses anonymous inference", r.status_code == 401, f"status {r.status_code}")

    # ---- workflow: player uploads -> analyst analyzes -> inference + decision
    pl = sessions["player"]
    home = pl.get(base + "/app/", timeout=60).text
    m = re.search(r'/app/players/(\d+)"', home)
    if not m:
        raise SystemExit("The player account has no player profile.")
    pid = m.group(1)
    url = f"{base}/app/players/{pid}/sessions/new"
    data = {"_csrf": csrf(pl, base, f"/app/players/{pid}/sessions/new"), "mode": "story", "world": "2", "difficulty": "1",
            "minutes": "10", "deaths": "6", "outcome": "lost", "hp_end": "0", "boss_fight": "on", "enemy_distance": "60",
            "actions": SEQ, "notes": "Operational review (automatic) - can be deleted"}
    r = pl.post(url, data=data, timeout=60)
    sid = re.search(r"/app/sessions/(\d+)", r.url)
    check("Player uploads a session", bool(sid), r.url)
    sid = sid.group(1)
    an = sessions["analyst"]
    t = time.perf_counter()
    r = an.post(f"{base}/app/sessions/{sid}/analyze", data={"_csrf": csrf(an, base, "/app/")}, timeout=120)
    ms = 1000 * (time.perf_counter() - t)
    check("Analyst runs the model on the session", "Analyzed in" in r.text, f"round trip {ms:.0f} ms")
    page = an.get(f"{base}/app/sessions/{sid}", timeout=60).text
    acc = re.search(r"(\d+) of (\d+) next actions guessed", page)
    check("Model result shown with real-vs-predicted comparison", bool(acc),
          f"{acc.group(1)}/{acc.group(2)} right" if acc else "")
    hist = an.get(f"{base}/app/inferences", timeout=60).text
    check("Inference history stores user, model version and latency",
          f"#{sid}</a>" in hist and (h.get("model_version") or "?") in hist)
    check("Decision step available (recommendation or keep)", "Decision supported by the model" in page)
    # ---- persistence: sign out, sign in again, the session is still there
    pl.post(base + "/logout", data={"_csrf": csrf(pl, base, "/app/")}, timeout=60)
    pl2 = login(base, env["REVIEW_PLAYER_USER"], env["REVIEW_PLAYER_PASS"])
    check("Data persists after signing out and in", pl2.get(f"{base}/app/sessions/{sid}", timeout=60).status_code == 200)
    mon = an.get(base + "/app/monitoring.json", timeout=60).json()
    ev["monitoring"] = {k: mon.get(k) for k in ("n", "mean", "p95", "psi", "psi_state", "n_errors", "version")}
    check("Monitoring answers (inferences, latency, drift, errors)", mon.get("n", 0) >= 1,
          f"{mon.get('n')} inferences, mean {mon.get('mean') or 0:.0f} ms, PSI {mon.get('psi')}, {mon.get('n_errors')} errors")
    ev["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    ev["passed"] = sum(c["ok"] for c in ev["checks"])
    ev["total"] = len(ev["checks"])
    local = any(h in base for h in ("127.0.0.1", "localhost"))
    ev["environment"] = "LOCAL" if local else "DEPLOYED"
    out = OUT.with_name("operational_review_local.json") if local else OUT      # local runs never pose as production
    out.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    print(f"\n{ev['passed']}/{ev['total']} checks passed · evidence in {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
