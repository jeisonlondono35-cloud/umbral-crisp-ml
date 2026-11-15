"""List the English text segments of the site that are not yet in app/i18n/es.json.

    python tools/i18n_extract.py            -> writes reports/_i18n_missing.json (key -> "") and prints how many

It builds a throw-away copy of the system (temporary SQLite, demo data from `seed`), signs in as a player, an
analyst and an administrator, opens every GET page of the app (parameters filled with real ids) and the flash
messages written in the code, and collects every segment that app/i18n.py would translate.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = Path(tempfile.mkdtemp()) / "i18n.db"
os.environ.update(DATABASE_URL=f"sqlite:///{DB.as_posix()}", SECRET_KEY="i18n", ADMIN_USERNAME="root",
                  ADMIN_PASSWORD="Root12345x")
os.environ.pop("RENDER", None)
sys.path.insert(0, str(ROOT))

from app import i18n  # noqa: E402
from app.app import app  # noqa: E402
from app.system.models import (PlaySession, Player, Recommendation, Record, SavedGame, Ticket, User,  # noqa: E402
                               db)
from app.system.seed import seed  # noqa: E402

PW = "Passw0rd99"


def setup():
    with app.app_context():
        seed(analyze=True)
        for name, role in (("ana", "analyst"), ("pia", "player")):
            u = User(username=name, display_name=name.title(), role=role)
            u.set_password(PW)
            db.session.add(u)
            db.session.flush()
            if role == "player":
                db.session.add(Player(owner_id=u.id, gamer_tag="PiaTag", character="jay", difficulty=1))
        db.session.commit()


def login(c, user, pw):
    html = c.get("/login").get_data(as_text=True)
    tok = re.search(r'name="_csrf" value="([^"]+)"', html).group(1)
    c.post("/login", data={"_csrf": tok, "username": user, "password": pw})


def ids():
    with app.app_context():
        first = lambda m: (m.query.first().id if m.query.first() else 1)  # noqa: E731
        own = Player.query.filter_by(gamer_tag="PiaTag").first()
        return {"pid": first(Player), "sid": first(PlaySession), "tid": first(Ticket), "rid": first(Recommendation),
                "gid": first(SavedGame), "uid": first(User), "n": 1, "name": "user-manual", "code": "en",
                "own": own.id if own else 1, "record_id": first(Record)}


def urls():
    v = ids()
    out = set()
    for rule in app.url_map.iter_rules():
        if "GET" not in rule.methods or rule.endpoint in ("static", "set_lang") or rule.rule.endswith((".pdf", ".csv", ".json")):
            continue
        u = rule.rule
        for arg in rule.arguments:
            val = v.get(arg, v["pid"] if "pid" in arg or "player" in arg else v["sid"] if "sid" in arg or "session" in arg
                        else v["tid"] if "tid" in arg or "ticket" in arg else 1)
            u = re.sub(rf"<(?:\w+:)?{arg}>", str(val), u)
        out.add(u)
    out |= {f"/phase/{i}" for i in range(1, 7)} | {"/docs/technical-manual", "/app/sessions?status=analyzed",
                                                   "/app/recommendations?status=all", "/demo"}
    return sorted(out)


def code_messages() -> set[str]:
    msgs = set()
    for f in (ROOT / "app" / "system").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r'(?:flash|_err|errors\.append|abort_msg)\(\s*f?"([^"]{6,})"', src):
            msgs.add(re.sub(r"\{[^}]*\}", "{n}", m.group(1)))
    return msgs


def main():
    setup()
    found: dict[str, int] = {}
    pages = urls()
    for who in (None, ("pia", PW), ("ana", PW), ("root", "Root12345x")):
        c = app.test_client()
        if who:
            login(c, *who)
        for u in pages:
            r = c.get(u)
            if r.mimetype != "text/html":
                continue
            for s in i18n.segments(r.get_data(as_text=True)):
                found[s] = found.get(s, 0) + 1
        r = c.post("/demo", data={"_csrf": re.search(r'name="_csrf" value="([^"]+)"', c.get("/demo").get_data(as_text=True)).group(1),
                                  "history": "shoot, shoot, dodge", "w": "3", "mode": "ole", "diff": "1", "hp": "4",
                                  "ne": "160", "boss": "on", "move": "special", "close": "on"})
        for s in i18n.segments(r.get_data(as_text=True)):
            found[s] = found.get(s, 0) + 1
    for m in code_messages():
        found.setdefault(i18n.key_of(m)[0], 1)
    have = i18n.table()
    missing = {k: "" for k in sorted(found) if k not in have}
    out = ROOT / "reports" / "_i18n_missing.json"
    out.write_text(json.dumps(missing, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(found)} segments, {len(missing)} missing -> {out}")


if __name__ == "__main__":
    main()
