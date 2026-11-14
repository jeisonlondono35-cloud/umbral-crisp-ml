"""Demo data for the information system (flask --app app.app seed-demo).

The action sequences are NOT invented here: they are whole sessions of the TEST players of Activity 1
(data/processed/actions.csv, split == "test", SIMULATED), so the demo shows the model working on data it
never saw in training. Deaths, minutes and outcomes are drawn with a fixed seed (42). Every demo row says
"DEMO" in its notes so it can be told apart from real use.
"""
from __future__ import annotations

import random
from datetime import timedelta

import pandas as pd

from src.umbral_ml.config import PROC

from . import ml_service as ML
from .models import Player, PlaySession, Record, SavedGame, Ticket, User, db, now

TAGS = ["EcoRunner", "LagoAzul", "Brasa_7", "NieblaFox", "ArcoLento", "Penitente2", "Oryonita", "SalDelMar",
        "ZarpaRoja", "LuzPropia", "Ceniza_K", "Umbralero"]


def seed(analyze: bool = True) -> str:
    if Player.query.filter(Player.gamer_tag.in_(TAGS)).first():
        return "Demo data already exists (nothing done)."
    rng = random.Random(42)
    staff = User.query.filter(User.role.in_(("analyst", "admin"))).order_by(User.role).first()
    df = pd.read_csv(PROC / "actions.csv", usecols=["pid", "sid", "seq", "target", "w", "mode", "diff", "hp", "ne",
                                                    "boss", "split"])
    df = df[df["split"] == "test"].sort_values(["pid", "sid", "seq"])
    # one simulated TEST player per demo player, with his sessions in their real order (the model uses the
    # player's earlier sessions as history, exactly like in training)
    by_player = []
    for _, pl in df.groupby("pid"):
        sess = [g for _, g in pl.groupby("sid", sort=False) if 20 <= len(g) <= 250]
        if len(sess) >= 2:
            by_player.append(sess)
    rng.shuffle(by_player)
    out_s = 0
    for i, tag in enumerate(TAGS):
        diff = rng.choice([0, 1, 1, 1, 2, 2, 3])
        p = Player(gamer_tag=tag, character=rng.choice(["jay", "jake"]), difficulty=diff,
                   country=rng.choice(["Colombia", "Mexico", "Peru", "Chile", None]),
                   created_at=now() - timedelta(days=60 - i * 3), created_by=staff.id if staff else None)
        db.session.add(p)
        db.session.flush()
        world = rng.randint(1, 6)
        relics = rng.randint(max(0, world - 2), min(5, world - 1 if world > 1 else 0))
        db.session.add(SavedGame(player_id=p.id, slot=0, world=world, play_minutes=rng.randint(30, 900),
                                 stars=rng.randint(0, 16 * world), relics=relics, diary_pages=rng.randint(0, 15),
                                 difficulty=diff))
        if rng.random() < 0.5:
            db.session.add(SavedGame(player_id=p.id, slot=1, world=max(1, world - 1), play_minutes=rng.randint(10, 300),
                                     stars=rng.randint(0, 16 * max(1, world - 1)), relics=max(0, relics - 1),
                                     diary_pages=rng.randint(0, 8), difficulty=diff))
        own = by_player[i % len(by_player)] if by_player else []
        start = now() - timedelta(days=40)
        for k, g in enumerate(own[:5]):
            acts = g["target"].tolist()
            n = len(acts)
            minutes = max(3, round(n / rng.uniform(5, 9)))
            skill = rng.random()
            deaths = max(0, round(minutes / 10 * (1 + 3 * (1 - skill)) * (0.6 + 0.3 * diff) + rng.uniform(-1, 1)))
            outcome = "won" if skill > 0.55 and deaths < minutes / 4 else rng.choice(["lost", "lost", "quit"])
            ne = g["ne"].median()
            s = PlaySession(player_id=p.id, played_at=start + timedelta(days=7 * k + rng.randint(0, 5), hours=rng.randint(0, 23)),
                            mode=str(g["mode"].iloc[0]), world=int(g["w"].iloc[0]), difficulty=diff, minutes=minutes,
                            deaths=deaths, outcome=outcome, boss_fight=bool(g["boss"].max()),
                            hp_end=0 if outcome == "lost" else rng.randint(1, 5),
                            enemy_distance=None if pd.isna(ne) else int(min(400, ne)), actions=",".join(acts),
                            notes="DEMO · actions of a simulated TEST player of Activity 1",
                            created_by=staff.id if staff else None)
            db.session.add(s)
            db.session.flush()
            out_s += 1
            if analyze and staff and rng.random() < 0.6:
                ML.analyze_session(s, staff.id)
        if rng.random() < 0.7:
            mode = rng.choice(["ole0", "ole1", "ole2", "panteon", "rush"])
            val = {"panteon": rng.randint(2, 12), "rush": rng.randint(400, 1500)}.get(mode, rng.randint(5, 45))
            db.session.add(Record(player_id=p.id, mode=mode, value=val, verified=rng.random() < 0.7,
                                  verified_by=staff.id if staff else None))
    if staff:
        db.session.add(Ticket(created_by=staff.id, category="model", severity="medium",
                              title="DEMO · The Forest boss always reads my shield",
                              description="DEMO ticket: a tester says the boss feints every time they block."))
    db.session.commit()
    return f"Demo data: {len(TAGS)} players, {out_s} sessions" + (" (part of them analyzed)." if analyze and staff
                                                                    else ". Create an analyst to analyze them.")
