"""Audit -- and, only when asked, rebuild -- the persisted LinUCB routing state (``bandit_arms`` / ``routing_rewards``).

The learner loads its arms (A_a, b_a) as stored; it does not rebuild them from the reward log. Rewards
credited before 2026-10-26 during the 20-incident bootstrap were applied at once with the 600 s
fallback L_max instead of the paper's 90th-percentile L_max, and those updates are still in the
stored arms. This script shows what is stale and what a rebuild would change:

* every stored reward is rescored under the current protocol (bootstrap incomplete: all become
  provisional and the arms return to their prior; complete: rescored with the final L_max);
* the arms are rebuilt by replaying the rescored rewards oldest first.

By default nothing is written: the audit runs on a temporary copy of the database. ``--apply`` first
copies the database file next to itself (``<name>.bak-<timestamp>``) and refuses when the stored arms
are not a replay of the stored rewards, unless ``--force``. Rewards stored before 2026-10-26 have
their IC_exec and latency rounded (4 and 3 decimals), so a rebuild reproduces them to that precision.

    python scripts/rebuild_bandit_state.py                                   # audit the configured database
    python scripts/rebuild_bandit_state.py --database sqlite:///./cade_e2e.db
    python scripts/rebuild_bandit_state.py --database sqlite:///./cade_e2e.db --apply
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy.engine import make_url  # noqa: E402

from app.core.config import Settings  # noqa: E402
from app.routing.learning import RoutingLearner  # noqa: E402
from app.storage.bandit_repository import BanditRepository  # noqa: E402
from app.storage.database import Database  # noqa: E402


def sqlite_path(url: str) -> Path:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite" or not parsed.database or parsed.database == ":memory:":
        raise SystemExit("only file-based SQLite databases are supported (the file is backed up before --apply)")
    path = Path(parsed.database)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def learner_for(path: Path, settings: Settings) -> tuple[RoutingLearner, Database]:
    database = Database(f"sqlite:///{path.as_posix()}")
    database.create_schema()  # adds the bandit tables to an older database; existing tables are not altered
    return RoutingLearner.from_settings(settings, BanditRepository(database)), database


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--database", help="SQLAlchemy URL (default: CADE_DATABASE_URL)")
    parser.add_argument("--apply", action="store_true", help="back up the database file, then rebuild in place")
    parser.add_argument("--force", action="store_true", help="with --apply: rebuild even if the stored arms are unexplained")
    parser.add_argument("--json", type=Path, help="write the audit and the planned changes as JSON")
    args = parser.parse_args()
    settings = Settings()
    url = args.database or settings.database_url
    path = sqlite_path(url)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")

    if args.apply:
        backup = path.with_name(f"{path.name}.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        shutil.copy2(path, backup)
        print(f"backed up {path} -> {backup}")
        learner, database = learner_for(path, settings)
    else:
        copy = Path(tempfile.mkdtemp(prefix="cade_bandit_audit_")) / path.name
        shutil.copy2(path, copy)
        learner, database = learner_for(copy, settings)
    try:
        audit = learner.last_audit
        print(f"database {path.name}: {audit.rewards} stored rewards ({audit.provisional} provisional), "
              f"bootstrap {learner.bootstrap}, latencies seen {len(learner.repository.first_latencies(learner.bootstrap))}, "
              f"protocol L_max {audit.latency_max if audit.latency_max is not None else 'not yet known'}")  # fmt: skip
        print(f"  stale (applied with a superseded L_max): {list(audit.stale) or 'none'}")
        print(f"  stored arms {'equal' if audit.consistent else 'DIFFER FROM'} a replay of the stored rewards "
              f"(max diff {audit.max_arm_difference:.3g})")  # fmt: skip
        plan = learner.rebuild(apply=args.apply, force=args.force)
        for change in plan.changes:
            print(f"  decision {change['decision_id']} ({change['arm']}): reward {change['old_reward']:.4f} -> "
                  f"{change['new_reward']:.4f}, L_max {change['old_latency_max']} -> {change['new_latency_max']}, "
                  f"provisional {change['old_provisional']} -> {change['new_provisional']}")  # fmt: skip
        print(f"  arms before: {plan.arms_before}")
        print(f"  arms after:  {plan.arms_after}")
        print("  APPLIED" if plan.applied else "  dry run: nothing was written (the audit used a temporary copy)")
        if args.json:
            args.json.write_text(json.dumps({"database": path.name, "audit": {**audit.__dict__, "stale": list(audit.stale)},
                                             "changes": plan.changes, "arms_before": plan.arms_before,
                                             "arms_after": plan.arms_after, "applied": plan.applied}, indent=1), encoding="utf-8")  # fmt: skip
    finally:
        database.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
