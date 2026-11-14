"""Activity 2 - automatic tests of the information system (python -m pytest -q).

They run against a temporary SQLite database and check: registration and sign-in, password hashing, no
self-promotion to admin, CSRF, lockout, authorization of every role (server side, also for other players'
data), validation and business rules, and the whole workflow session -> model -> recommendation -> decision
-> applied difficulty -> outcome compared with the next session.
"""
from __future__ import annotations

import os
import re
import tempfile
from datetime import timedelta

import pytest

_DB = os.path.join(tempfile.mkdtemp(), "test.db")
os.environ.update(DATABASE_URL=f"sqlite:///{_DB}", SECRET_KEY="test-secret", ADMIN_USERNAME="root",
                  ADMIN_PASSWORD="RootPass123")
os.environ.pop("RENDER", None)

from app.app import app  # noqa: E402
from app.system.models import (PlaySession, Player, Recommendation, Inference, SavedGame, User,  # noqa: E402
                               db, now)

app.config.update(TESTING=True)


@app.route("/_boom")
def _boom():
    raise RuntimeError("test error")
SEQ = ",".join(["shoot", "shoot", "dodge", "melee", "shoot", "block", "shoot", "dodge", "jump", "shoot"] * 4)


def tok(c, path="/login"):
    return re.search(r'name="_csrf" value="([^"]+)"', c.get(path).text).group(1)


def login(c, user, pw):
    return c.post("/login", data={"_csrf": tok(c), "username": user, "password": pw}, follow_redirects=True)


def register(c, user, tag, pw="Passw0rd99", **extra):
    data = {"_csrf": tok(c, "/register"), "username": user, "display_name": user.title(), "gamer_tag": tag,
            "character": "jake", "password": pw, "password2": pw, **extra}
    return c.post("/register", data=data, follow_redirects=True)


def make_user(name, role, pw="Passw0rd99"):
    with app.app_context():
        u = User(username=name, display_name=name, role=role)
        u.set_password(pw)
        db.session.add(u)
        db.session.commit()


@pytest.fixture(scope="module", autouse=True)
def staff():
    make_user("ana", "analyst")
    yield


def client():
    return app.test_client()


# ------------------------------------------------------------------------------------- registration
def test_public_docs_and_protected_modules():
    c = client()
    assert c.get("/").status_code == 200 and c.get("/phase/3").status_code == 200
    for path in ("/app/", "/app/players", "/app/sessions", "/app/monitoring", "/app/users", "/monitoring/live"):
        r = c.get(path)
        assert r.status_code == 302 and "/login" in r.headers["Location"], path
    r = c.post("/api/predict/next", json={"history": ["shoot"]})
    assert r.status_code == 401


def test_register_hashes_password_and_never_admin():
    c = client()
    r = register(c, "eve", "EveTag", role="admin")          # an attacker adds role=admin by hand
    assert "My account" in r.text
    with app.app_context():
        u = User.query.filter_by(username="eve").one()
        assert u.role == "player"
        assert u.password_hash != "Passw0rd99" and u.password_hash.startswith(("scrypt", "pbkdf2"))
        assert u.player and u.player.gamer_tag == "EveTag"
    assert c.get("/app/users").status_code == 403


def test_register_validation():
    c = client()
    assert "different" in register(c, "bob1", "Bob1", pw="Passw0rd99", password2="Other999").text
    assert "8 characters" in register(c, "bob2", "Bob2", pw="a1").text
    assert "taken" in register(c, "eve", "Another").text


def test_csrf_required():
    c = client()
    r = c.post("/login", data={"username": "root", "password": "RootPass123"})
    assert r.status_code == 400


def test_lockout_after_five_wrong_passwords():
    make_user("lock", "player")
    c = client()
    for _ in range(5):
        login(c, "lock", "wrong-pass1")
    r = login(c, "lock", "Passw0rd99")
    assert "Too many wrong passwords" in r.text


def test_logout():
    c = client()
    login(c, "ana", "Passw0rd99")
    assert c.get("/app/").status_code == 200
    c.post("/logout", data={"_csrf": tok(c, "/app/")})
    assert c.get("/app/").status_code == 302


# ------------------------------------------------------------------------------------- authorization
def test_player_sees_only_own_data():
    a, b = client(), client()
    register(a, "pla", "PlayerA")
    register(b, "plb", "PlayerB")
    with app.app_context():
        pa = Player.query.filter_by(gamer_tag="PlayerA").one().id
        pb = Player.query.filter_by(gamer_tag="PlayerB").one().id
    assert a.get(f"/app/players/{pa}").status_code == 200
    assert a.get(f"/app/players/{pb}").status_code == 403
    assert a.get(f"/app/players/{pb}/sessions/new").status_code == 403
    for path in ("/app/players", "/app/recommendations", "/app/reports", "/app/monitoring", "/app/users",
                 "/monitoring/live"):
        assert a.get(path).status_code == 403, path
    # a forged POST to a staff action is refused on the server
    assert a.post("/app/sessions/1/analyze", data={"_csrf": tok(a, "/app/")}).status_code in (403, 404)


def test_analyst_cannot_manage_users_admin_can():
    c = client()
    login(c, "ana", "Passw0rd99")
    assert c.get("/app/players").status_code == 200
    assert c.get("/app/users").status_code == 403
    r = c.post("/app/users/1", data={"_csrf": tok(c, "/app/"), "role": "admin", "active": "1"})
    assert r.status_code == 403
    d = client()
    login(d, "root", "RootPass123")
    assert d.get("/app/users").status_code == 200
    with app.app_context():
        root_id = User.query.filter_by(username="root").one().id
    r = d.post(f"/app/users/{root_id}", data={"_csrf": tok(d, "/app/"), "role": "player", "active": "1"},
               follow_redirects=True)
    assert "cannot change your own role" in r.text


# ------------------------------------------------------------------------------- validation and rules
def test_saved_game_rules():
    c = client()
    register(c, "saver", "Saver")
    with app.app_context():
        pid = Player.query.filter_by(gamer_tag="Saver").one().id
    base = {"slot": "1", "world": "2", "play_minutes": "30", "stars": "10", "relics": "1", "diary_pages": "2",
            "difficulty": "1"}
    url = f"/app/players/{pid}/saves/new"
    assert "Relics cannot be more" in c.post(url, data={**base, "relics": "4", "_csrf": tok(c, url)},
                                             follow_redirects=True).text
    assert "at most 32 stars" in c.post(url, data={**base, "stars": "60", "_csrf": tok(c, url)},
                                        follow_redirects=True).text
    c.post(url, data={**base, "_csrf": tok(c, url)})
    assert "already used" in c.post(url, data={**base, "_csrf": tok(c, url)}, follow_redirects=True).text
    with app.app_context():
        assert SavedGame.query.filter_by(player_id=pid).count() == 1


def test_session_validation():
    c = client()
    register(c, "val", "ValTag")
    with app.app_context():
        pid = Player.query.filter_by(gamer_tag="ValTag").one().id
    url = f"/app/players/{pid}/sessions/new"
    data = {"mode": "story", "world": "1", "difficulty": "1", "minutes": "10", "deaths": "1", "outcome": "won",
            "hp_end": "2", "actions": "shoot,fly,dodge,shoot,shoot"}
    assert "Unknown actions: fly" in c.post(url, data={**data, "_csrf": tok(c, url)}, follow_redirects=True).text
    assert "at least 5" in c.post(url, data={**data, "actions": "shoot,dodge", "_csrf": tok(c, url)},
                                  follow_redirects=True).text
    assert "at least one heart" in c.post(url, data={**data, "hp_end": "0", "actions": SEQ, "_csrf": tok(c, url)},
                                          follow_redirects=True).text


# ---------------------------------------------------------------------------- the complete workflow
def _upload(c, pid, outcome, deaths, minutes=10, when=None):
    url = f"/app/players/{pid}/sessions/new"
    data = {"mode": "story", "world": "2", "difficulty": "2", "minutes": str(minutes), "deaths": str(deaths),
            "outcome": outcome, "hp_end": "0" if outcome == "lost" else "3", "actions": SEQ, "boss_fight": "on",
            "enemy_distance": "50", "_csrf": tok(c, url)}
    if when:
        data["played_at"] = when
    c.post(url, data=data)
    with app.app_context():
        return PlaySession.query.filter_by(player_id=pid).order_by(PlaySession.id.desc()).first().id


def test_workflow_session_model_recommendation_outcome():
    pc, an = client(), client()
    register(pc, "flow", "FlowTag")
    login(an, "ana", "Passw0rd99")
    with app.app_context():
        p = Player.query.filter_by(gamer_tag="FlowTag").one()
        p.difficulty = 2
        db.session.commit()
        pid = p.id
    past = (now() - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M")
    sid = _upload(pc, pid, "lost", deaths=9, when=past)         # 9 deaths in 10 min: too hard
    assert pc.post(f"/app/sessions/{sid}/analyze", data={"_csrf": tok(pc, "/app/")}).status_code == 403
    an.post(f"/app/sessions/{sid}/analyze", data={"_csrf": tok(an, "/app/")})
    with app.app_context():
        s = db.session.get(PlaySession, sid)
        assert s.status == "analyzed" and 0 <= s.predictability <= 1 and 0 < s.risk < 1
        infs = Inference.query.filter_by(session_id=sid).all()
        assert {i.kind for i in infs} == {"session_profile", "hit"}
        assert all(i.user_id and i.model_version and i.latency_ms >= 0 for i in infs)
        assert infs[0].n_checked == len(SEQ.split(",")) - 1
        rec = Recommendation.query.filter_by(session_id=sid).one()
        assert rec.action == "lower" and rec.status == "pending" and rec.suggested_difficulty == 1
        rid = rec.id
    # a player can never decide; rejecting needs a note; approving changes the account
    assert pc.post(f"/app/recommendations/{rid}/decide", data={"_csrf": tok(pc, "/app/"), "decision": "approve"}).status_code == 403
    an.post(f"/app/recommendations/{rid}/decide", data={"_csrf": tok(an, "/app/"), "decision": "reject"})
    with app.app_context():
        assert db.session.get(Recommendation, rid).status == "pending"
    an.post(f"/app/recommendations/{rid}/decide", data={"_csrf": tok(an, "/app/"), "decision": "approve",
                                                        "note": "Agreed with the tester."})
    with app.app_context():
        assert db.session.get(Player, pid).difficulty == 1
        assert db.session.get(Recommendation, rid).status == "applied"
        assert db.session.get(PlaySession, sid).status == "reviewed"
    # the next session after the change is compared with the one before it
    sid2 = _upload(pc, pid, "won", deaths=1)
    an.post(f"/app/sessions/{sid2}/analyze", data={"_csrf": tok(an, "/app/")})
    with app.app_context():
        rec = db.session.get(Recommendation, rid)
        assert rec.after_session_id == sid2 and rec.effective is True
        # the second analysis used the first session as the player's history (as in training)
        prof = Inference.query.filter_by(session_id=sid2, kind="session_profile").one()
        assert '"past_actions": 40' in prof.input_json
    # the player sees the result and the inference history of his own profile
    assert "Difficulty decisions" in pc.get(f"/app/players/{pid}").text
    assert pc.get("/app/inferences").status_code == 200
    # an analyzed session cannot be edited, and only an admin can delete it
    assert "Only sessions that were not analyzed" in pc.get(f"/app/sessions/{sid2}/edit", follow_redirects=True).text


def test_monitoring_and_reports_for_staff():
    c = client()
    login(c, "ana", "Passw0rd99")
    r = c.get("/app/monitoring.json")
    assert r.status_code == 200 and r.json["n"] >= 2
    assert c.get("/app/reports").status_code == 200
    assert c.get("/app/reports/sessions.csv").text.startswith("id,player")
    assert c.get("/api/health").json["database"] == "ok"


def test_api_inference_is_logged_with_user():
    c = client()
    login(c, "ana", "Passw0rd99")
    r = c.post("/api/predict/next", json={"history": ["shoot", "dodge", "shoot"], "w": 1, "diff": 1})
    assert r.status_code == 200 and r.json["prediction"] in ("shoot", "melee", "block", "dodge", "jump", "hide")
    with app.app_context():
        i = Inference.query.filter_by(kind="next_action").order_by(Inference.id.desc()).first()
        assert i.user.username == "ana" and i.model_version


def test_errors_are_logged():
    c = client()
    assert c.get("/_boom").status_code == 500
    from app.system.models import ErrorLog
    with app.app_context():
        assert ErrorLog.query.filter_by(error_type="RuntimeError").count() == 1
