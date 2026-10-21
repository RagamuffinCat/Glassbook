"""Per-wallet checkpoints for the collection scripts, and atomic file writes."""

import json
import os
from pathlib import Path


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    os.replace(tmp, path)


def atomic_jsonl(path, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


class Checkpoint:
    """plan.json for the run + one file per wallet, written as each wallet finishes.
    A breaker trip loses at most the wallet in progress."""

    def __init__(self, run_id, base, sub="wallets"):
        self.run_id = run_id
        self.base = Path(base)
        self.dir = self.base / run_id
        self.wdir = self.dir / sub          # "wallets" for TWAP fills, "fills" for collect_mixed.py

    @staticmethod
    def latest(base):
        p = Path(base) / "latest"
        return p.read_text().strip() if p.exists() else None

    def exists(self):
        return (self.dir / "plan.json").exists()

    def create(self, plan):
        self.wdir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.dir / "plan.json", plan)
        (self.base / "latest").write_text(self.run_id)

    def plan(self):
        return json.loads((self.dir / "plan.json").read_text())

    def update_plan(self, **kw):
        p = self.plan()
        p.update(kw)
        atomic_json(self.dir / "plan.json", p)

    def wallet_path(self, w):
        return self.wdir / f"{w}.json"

    def load_wallet(self, w):
        p = self.wallet_path(w)
        return json.loads(p.read_text()) if p.exists() else None

    def save_wallet(self, w, rec):
        atomic_json(self.wallet_path(w), rec)

    def states(self, wallets):
        return {w: (self.load_wallet(w) or {}).get("state") for w in wallets}
