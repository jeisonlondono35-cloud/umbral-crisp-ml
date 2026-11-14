"""Download the REAL data from the game's Supabase project with the PUBLIC key only.

- umbral_ml_log_export: anonymous gameplay batches recorded by scripts/ml_log.gd (no user, no name)
- umbral_ml_log_stats:  events / players per day (monitoring)
- umbral_ml_public:     aggregate accuracy of the in-game predictor over all accounts (production evidence)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import requests

from .config import RAW, SUPABASE_KEY, SUPABASE_URL


def rpc(fn: str, args: dict | None = None, timeout: int = 30):
    r = requests.post(f"{SUPABASE_URL}/rest/v1/rpc/{fn}", json=args or {}, timeout=timeout,
                      headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
                               "Content-Type": "application/json"})
    r.raise_for_status()
    return r.json()


def fetch_log() -> list[dict]:
    out, after = [], 0
    while True:
        page = rpc("umbral_ml_log_export", {"p_after": after, "p_limit": 1000})
        if not page:
            return out
        out += page
        after = page[-1]["id"]


def main() -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    public = rpc("umbral_ml_public")
    (RAW / "ml_public.json").write_text(json.dumps({"fetched_at": stamp, **public}, indent=1), encoding="utf-8")
    print("umbral_ml_public:", public)
    try:
        batches = fetch_log()
        stats = rpc("umbral_ml_log_stats")
    except requests.HTTPError as e:      # SQL not installed yet (404)
        print("Real gameplay log not available yet:", e)
        batches, stats = [], {}
    (RAW / "real_export.json").write_text(json.dumps({"source": "REAL", "fetched_at": stamp,
                                                      "batches": batches}), encoding="utf-8")
    (RAW / "log_stats.json").write_text(json.dumps({"fetched_at": stamp, **(stats or {})}, indent=1),
                                        encoding="utf-8")
    print(f"REAL: {len(batches)} batches, {sum(b['n'] for b in batches)} events")


if __name__ == "__main__":
    main()
