"""Persistence for the learned router: LinUCB arm statistics and the reward log."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from app.core.errors import StorageError
from app.core.logging import get_logger, log_event
from app.routing.bandit import ArmState
from app.schemas.routing import RoutingAction, RoutingReward
from app.storage.database import Database
from app.storage.models import BanditArmRecord, RoutingRewardRecord

logger = get_logger("storage")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _error(operation: str, exc: Exception) -> StorageError:
    log_event(logger, "database operation failed", level=logging.ERROR, operation=operation, stage="storage",
              outcome="failure", error_category="storage_error", exception_type=type(exc).__name__)  # fmt: skip
    return StorageError("database operation failed")


class BanditRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def load_arms(self) -> dict[RoutingAction, ArmState]:
        try:
            with self.db.session() as session:
                return {RoutingAction(r.arm): ArmState(a=json.loads(r.a_json), b=json.loads(r.b_json), pulls=r.pulls)
                        for r in session.scalars(select(BanditArmRecord))}  # fmt: skip
        except SQLAlchemyError as exc:
            raise _error("load_bandit_arms", exc) from None

    def credit(self, rewards: list[RoutingReward], arms: dict[RoutingAction, ArmState]) -> list[RoutingReward]:
        """Store ``rewards`` and the updated ``arms`` in one transaction; decisions already credited are skipped."""
        try:
            with self.db.session() as session:
                ids = [r.decision_id for r in rewards]
                done = set(session.scalars(select(RoutingRewardRecord.decision_id)
                                           .where(RoutingRewardRecord.decision_id.in_(ids))))  # fmt: skip
                fresh = [r for r in rewards if r.decision_id not in done]
                for reward in fresh:
                    session.add(RoutingRewardRecord(
                        decision_id=reward.decision_id, incident_id=reward.incident_id, run_id=reward.run_id,
                        arm=reward.arm.value, terminal=reward.terminal, reward=reward.reward,
                        features_json=json.dumps(reward.features), components_json=json.dumps(reward.components),
                        synthetic=reward.synthetic, created_at=reward.created_at))  # fmt: skip
                self._store_arms(session, arms)
                return fresh
        except SQLAlchemyError as exc:
            raise _error("credit_routing_rewards", exc) from None

    def all(self) -> list[RoutingReward]:
        """Every stored reward, oldest first."""
        try:
            with self.db.session() as session:
                return [self._to_reward(r) for r in session.scalars(select(RoutingRewardRecord).order_by(RoutingRewardRecord.id))]
        except SQLAlchemyError as exc:
            raise _error("list_all_routing_rewards", exc) from None

    def provisional(self) -> list[RoutingReward]:
        """Rewards logged before L_max was known (``components["provisional"]``), oldest first."""
        try:
            with self.db.session() as session:
                query = (select(RoutingRewardRecord).where(RoutingRewardRecord.components_json.contains('"provisional": true'))
                         .order_by(RoutingRewardRecord.id))  # fmt: skip
                return [self._to_reward(r) for r in session.scalars(query)]
        except SQLAlchemyError as exc:
            raise _error("list_provisional_routing_rewards", exc) from None

    def finalize(self, rewards: list[RoutingReward], arms: dict[RoutingAction, ArmState]) -> None:
        """Overwrite the value and components of already stored ``rewards`` and store ``arms``, in one transaction."""
        try:
            with self.db.session() as session:
                for reward in rewards:
                    record = session.scalars(select(RoutingRewardRecord)
                                             .where(RoutingRewardRecord.decision_id == reward.decision_id)).one()  # fmt: skip
                    record.reward, record.components_json = reward.reward, json.dumps(reward.components)
                self._store_arms(session, arms)
        except SQLAlchemyError as exc:
            raise _error("finalize_routing_rewards", exc) from None

    @staticmethod
    def _store_arms(session, arms: dict[RoutingAction, ArmState]) -> None:
        now = _utcnow()
        for arm, state in arms.items():
            record = session.get(BanditArmRecord, arm.value)
            if record is None:
                record = BanditArmRecord(arm=arm.value)
                session.add(record)
            record.a_json, record.b_json = json.dumps(state.a), json.dumps(state.b)
            record.pulls, record.updated_at = state.pulls, now

    @staticmethod
    def _to_reward(r: RoutingRewardRecord) -> RoutingReward:
        return RoutingReward(decision_id=r.decision_id, incident_id=r.incident_id, run_id=r.run_id,
                             arm=RoutingAction(r.arm), terminal=r.terminal, reward=r.reward,
                             features=json.loads(r.features_json), components=json.loads(r.components_json),
                             synthetic=r.synthetic, created_at=r.created_at)  # fmt: skip

    def credited(self, decision_ids: list[int]) -> set[int]:
        try:
            with self.db.session() as session:
                return set(session.scalars(select(RoutingRewardRecord.decision_id)
                                           .where(RoutingRewardRecord.decision_id.in_(decision_ids))))  # fmt: skip
        except SQLAlchemyError as exc:
            raise _error("credited_routing_decisions", exc) from None

    def rewarded_incidents(self) -> int:
        """Incidents whose terminal decision has been rewarded (drives the bootstrap period)."""
        try:
            with self.db.session() as session:
                query = select(func.count(func.distinct(RoutingRewardRecord.incident_id))).where(RoutingRewardRecord.terminal)
                return int(session.scalar(query) or 0)
        except SQLAlchemyError as exc:
            raise _error("count_rewarded_incidents", exc) from None

    def first_latencies(self, count: int) -> list[float]:
        """Latencies of the first ``count`` terminal rewards (L_max is their 90th percentile)."""
        try:
            with self.db.session() as session:
                query = (select(RoutingRewardRecord.components_json).where(RoutingRewardRecord.terminal)
                         .order_by(RoutingRewardRecord.id).limit(count))  # fmt: skip
                return [float(json.loads(c).get("latency_seconds", 0.0)) for c in session.scalars(query)]
        except SQLAlchemyError as exc:
            raise _error("first_routing_latencies", exc) from None

    def list(self, incident_id: Optional[str] = None, limit: int = 200) -> list[RoutingReward]:
        try:
            with self.db.session() as session:
                query = select(RoutingRewardRecord).order_by(RoutingRewardRecord.id.desc()).limit(limit)
                if incident_id is not None:
                    query = query.where(RoutingRewardRecord.incident_id == incident_id)
                return [self._to_reward(r) for r in session.scalars(query)]
        except SQLAlchemyError as exc:
            raise _error("list_routing_rewards", exc) from None
