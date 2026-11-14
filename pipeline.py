"""Runs the whole CRISP-ML pipeline, in order, so the report, the repository and the app always show
the same numbers:

    python pipeline.py              # auto: REAL data if there is enough, otherwise SIMULATED
    python pipeline.py --source real
    python pipeline.py --source simulated
    python pipeline.py --no-fetch   # do not download from Supabase (offline)
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from src.umbral_ml import fetch, monitor, prepare, report, simulate, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="auto", choices=["auto", "real", "simulated"])
ap.add_argument("--no-fetch", action="store_true")
a = ap.parse_args()
if not a.no_fetch:
    try:
        fetch.main()
    except Exception as e:  # offline: keep the last download
        print("fetch skipped:", e)
simulate.main()
prepare.main(a.source)
train.main()
monitor.run()
report.main()
print("Done. Run the app with:  python -m flask --app app.app run")
