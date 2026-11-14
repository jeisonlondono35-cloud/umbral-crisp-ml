# UMBRAL · CRISP-ML

**Teaching a game's bosses to read the player.** The machine learning of the game
[UMBRAL](https://juego-pixelart.onrender.com/) (J.A · J.P · J.Q), developed and documented with the six phases of
CRISP-ML in a Flask application.

| Deliverable | Where |
|---|---|
| Flask app (cover + six phases + live demo + API) | https://umbral-crisp-ml.onrender.com |
| Technical report (PDF) | [`reports/UMBRAL_CRISP-ML_report.pdf`](reports/UMBRAL_CRISP-ML_report.pdf) (also `/report.pdf` in the app) |
| This repository | code, training scripts, dependencies, saved models + preprocessing, results |

## The problem in one paragraph

UMBRAL's bosses adapt to whoever is playing. Inside the game (`scripts/ml_brain.gd` of the game repository) three small
models learn **online** during combat: an order-2 Markov chain with forgetting that predicts the player's **next action**
(shoot, melee, block, dodge, jump, hide), a running mean of the **dodge** direction, and a contextual bandit — one
logistic regression per boss attack — that estimates whether an attack will **hit**. This project measures how good
those models really are, compares them with other algorithms on recorded play data, picks the best one with a rule fixed
in advance, deploys it and plans its monitoring.

## Data

* **REAL — anonymous gameplay log.** Added to the game for this project (`scripts/ml_log.gd` in the game, SQL in
  `tools/supabase_ml_log.sql`): every action in combat (with what the in-game predictor expected), every boss attack with
  its 8 situation features and whether it hit, every dodge. No user, name or e-mail: random player and session ids.
  Stored in the team's Supabase project; read with the **public** key through `umbral_ml_log_export`.
* **REAL — in-game accuracy per account** (`umbral_ml_public`): production evidence.
* **SIMULATED — players generated from the game's rules** (`src/umbral_ml/simulate.py`), same format as the real log.
  Used while the real log is below the minimum fixed in Phase 1 (1,500 actions from 3 players) and for the monitoring
  test. **Every page, table and figure says which source it comes from.**

`python pipeline.py` (source `auto`) uses the real data as soon as there is enough of it.

## Reproduce everything

Requires Python 3.14 (the saved models need the exact versions in `requirements.txt`).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py                # fetch -> simulate -> prepare -> train/evaluate -> monitor -> PDF   (~5 min)
python -m flask --app app.app run # http://127.0.0.1:5000
```

Options: `python pipeline.py --source real|simulated|auto` and `--no-fetch` (offline). Individual steps:

| Step (CRISP-ML phase) | Command | Output |
|---|---|---|
| Download real data (1) | `python -m src.umbral_ml.fetch` | `data/raw/*.json` |
| Simulated players (1) | `python -m src.umbral_ml.simulate` | `data/simulated/sim_export.json` |
| Data engineering (2) | `python -m src.umbral_ml.prepare [auto\|real\|simulated]` | `data/processed/*.csv`, `data_report.json` |
| Model engineering + evaluation (3, 4) | `python -m src.umbral_ml.train` | `models/*.joblib`, `model_card.json`, `reports/results.json`, `reports/experiments.csv`, `reports/figures/` |
| Monitoring test (6) | `python -m src.umbral_ml.monitor` | `monitoring/monitoring_simulated.json`, `.jsonl`, figures |
| Report | `python -m src.umbral_ml.report` | `reports/UMBRAL_CRISP-ML_report.pdf` |

The seed is fixed (42), so the simulated run reproduces the same numbers.

## Repository map

```
pipeline.py                 runs every phase in order
src/umbral_ml/
  config.py                 paths, constants, public Supabase endpoint (anon key only)
  fetch.py                  real data from Supabase
  simulate.py               simulated players (clearly marked)
  online.py                 exact replicas of the in-game online models (the deployed baseline)
  prepare.py                Phase 2: quality profile, cleaning, features, player split (no leakage)
  features.py               feature step that lives inside a saved Pipeline
  train.py                  Phases 3-4: candidates, GroupKFold grid search, experiment log, selection, test, figures
  predict.py                Phase 5: loads the models and rebuilds features exactly as in training
  monitor.py                Phase 6: PSI drift, weekly performance, alerts (simulated operations)
  report.py                 the PDF
app/                        Flask app: app.py + templates (cover, phase1..6, demo)
models/                     saved models + preprocessing (sklearn Pipelines) + model card
data/                       raw, simulated and processed data (+ data_report.json)
reports/                    results.json, experiments.csv, figures, PDF
monitoring/                 monitoring test log (SIMULATED) and live prediction log
docs/expert_review.json     domain expert review (Phase 4)
.github/workflows/weekly.yml weekly retrain (Phase 6)
render.yaml                 deployment on Render
```

## API

```bash
curl -X POST https://umbral-crisp-ml.onrender.com/api/predict/next -H "Content-Type: application/json" \
     -d '{"history":["shoot","shoot","dodge"],"w":2,"mode":"story","diff":1,"hp":4,"ne":160,"boss":true}'
curl -X POST https://umbral-crisp-ml.onrender.com/api/predict/hit -H "Content-Type: application/json" \
     -d '{"move":"feint","close":1,"blocking":1,"w":2}'
curl https://umbral-crisp-ml.onrender.com/api/health
```

Inputs and outputs are documented in Phase 5 of the app.

## Deploy (Render)

1. Render → **New + → Blueprint** → connect this GitHub repository (it reads `render.yaml`).
2. It installs `requirements.txt` and starts `gunicorn app.app:app`. Health check: `/api/health`.
3. Every push to `main` (including the weekly retrain commit) redeploys the app.

## Traceability

The report, this repository and the app read the same files: `data/processed/data_report.json`,
`reports/experiments.csv`, `reports/results.json`, `models/model_card.json`, `monitoring/monitoring_simulated.json`,
`docs/expert_review.json`. Regenerating them with `pipeline.py` updates the three deliverables together.

## Privacy

The gameplay log contains no account, name, e-mail or device information — only a random player id, a random session
id and combat events. Players can switch it off in the game (Options → Accessibility → Share gameplay (ML)). Only the
public (publishable) Supabase key is used here; it can call the export function and nothing else.
