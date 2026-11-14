"""Shared constants and paths. Everything (scripts, Flask app, PDF) reads from here so the
problem, data, experiments and results stay traceable between the report, the repo and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
SIM = DATA / "simulated"
PROC = DATA / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
MON = ROOT / "monitoring"

SEED = 42

# Public Supabase project of the game (the PUBLISHABLE/anon key is public by design: it can only
# call the security-definer functions; it is the same one shipped inside the game, scripts/cloud.gd).
SUPABASE_URL = "https://ozxjelnjwggigvfrnwyh.supabase.co"
SUPABASE_KEY = "sb_publishable_bCzV7gFdJAAxZfZVzUMmXw_azOuHu8g"

# The six player actions the in-game predictor learns (scripts/ml_brain.gd ACTIONS)
ACTIONS = ["shoot", "melee", "block", "dodge", "jump", "hide"]
ACTION_LABEL = {"shoot": "Shoot (bow)", "melee": "Melee hit", "block": "Block (shield)",
                "dodge": "Dodge", "jump": "Jump", "hide": "Hide"}
GAP_MS = 4000          # in-game: 4 s without actions starts a new "phrase"
DECAY = 0.97           # in-game forgetting factor of the Markov predictor
MIN_SUPPORT = 3.0      # in-game minimum evidence in a context

# The 8 situation features the bosses use (Game.ml_features)
HIT_X = ["bias", "close", "far", "in_air", "blocking", "moving", "low_hp", "expects_evade"]
# Boss attacks that feed the bandit (guardian.gd _use_move, lobo_boss.gd _ml_move, espejo_boss.gd)
MOVES = ["dive", "feint", "volley", "special", "summon", "reposition", "repel",
         "lobo_charge", "lobo_bite", "lobo_zarpas", "lobo_breath", "lobo_howl", "lobo_jump",
         "lobo2_charge", "lobo2_bite", "lobo2_zarpas", "lobo2_breath", "lobo2_howl", "lobo2_dive",
         "espejo_shot", "espejo_rush", "espejo_guard", "espejo_hide"]
HIT_LR = 0.3           # in-game SGD learning rate

WORLDS = {1: "I Lake (Sadness)", 2: "II Forest (Denial)", 3: "III Volcano (Anger)",
          4: "IV City (Bargaining)", 5: "V Sea / Threshold (Depression)"}
MODES = {"story": "Story", "ole": "Waves", "pan": "Pantheon"}
DIFFS = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Legendary"}

# Minimum real data to use the real source (otherwise the pipeline says so and uses simulated)
MIN_REAL_ACTIONS = 1500
MIN_REAL_PLAYERS = 3

for p in (RAW, SIM, PROC, MODELS, FIG, MON):
    p.mkdir(parents=True, exist_ok=True)
