"""Weekly update of the decline-cap penalty λ from logged traffic, without labels.

    uv run python -m fraud.serving.cap --log scored_last_week.parquet

The log needs the API's `fraud_probability` and the transaction amount
(`TransactionAmt`). λ is set so the expected share of good customers declined,
sum of (1 - p) over declines / sum of (1 - p), equals the controller target in
`configs/costs.yaml`. The previous value is kept in `penalty_history`. Restart
or reload the API to use the new value.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.decision import bayes_policy, expected_good_decline_rate, load_costs, penalty_for_expected_cap
from fraud.serving.bundle import BUNDLE_DIR


def update_penalty(bundle_dir: Path, p, amount, target: float) -> dict:
    """Recompute λ on a batch of scored traffic and write it into the bundle's decision.json."""
    path = Path(bundle_dir) / "decision.json"
    decision = json.loads(path.read_text())
    p, amount = np.asarray(p, dtype=float), np.asarray(amount, dtype=float)
    lam = penalty_for_expected_cap(p, amount, decision["costs"], target)
    expected = expected_good_decline_rate(bayes_policy(p, amount, decision["costs"], fp_penalty=lam), p)
    decision.setdefault("penalty_history", []).append({
        "previous": decision["fp_penalty"], "new": lam, "target": target, "n_transactions": int(len(p)),
        "expected_good_decline_rate": expected, "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    decision["fp_penalty"] = lam
    path.write_text(json.dumps(decision, indent=1))
    return decision["penalty_history"][-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="parquet or csv with fraud_probability and TransactionAmt")
    parser.add_argument("--bundle", type=Path, default=BUNDLE_DIR)
    parser.add_argument("--target", type=float, default=None)
    args = parser.parse_args()
    log = pd.read_parquet(args.log) if args.log.suffix == ".parquet" else pd.read_csv(args.log)
    target = args.target if args.target is not None else load_costs()["constraints"]["controller_target"]
    print(update_penalty(args.bundle, log["fraud_probability"], log["TransactionAmt"], target))


if __name__ == "__main__":
    main()
