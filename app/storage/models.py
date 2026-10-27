"""SQLAlchemy ORM tables.

Complex nested data (the normalised incident, the canonical analysis) is stored as JSON text
validated by Pydantic on the way in and out; frequently queried fields get their own columns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator):
    """Stores UTC datetimes; returns timezone-aware values (SQLite drops tzinfo otherwise)."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetimes are not allowed in the database")
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    def process_result_value(self, value: Optional[datetime], dialect) -> Optional[datetime]:
        return value.replace(tzinfo=timezone.utc) if value is not None else None


class Base(DeclarativeBase):
    pass


class IncidentRecord(Base):
    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    title: Mapped[Optional[str]] = mapped_column(String(300))
    severity: Mapped[Optional[str]] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), index=True)
    raw_description: Mapped[str] = mapped_column(Text)
    normalized_json: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(UTCDateTime)
    occurred_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)
    analysis_started_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)

    analyses: Mapped[list["AnalysisRecord"]] = relationship(
        back_populates="incident", cascade="all, delete-orphan", order_by="AnalysisRecord.id"
    )


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    extraction_status: Mapped[str] = mapped_column(String(16))
    provider: Mapped[str] = mapped_column(String(64))
    model_name: Mapped[str] = mapped_column(String(200))
    attempts: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[float] = mapped_column(Float)
    warnings_json: Mapped[str] = mapped_column(Text, default="[]")
    error_category: Mapped[Optional[str]] = mapped_column(String(64))
    error_message: Mapped[Optional[str]] = mapped_column(String(1000))
    canonical_json: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)

    incident: Mapped[IncidentRecord] = relationship(back_populates="analyses")


# --- security ---------------------------------------------------------------------------


class ApiKeyRecord(Base):
    """An API key. Only the SHA-256 of the key is stored; the key itself is shown once at creation."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    key_prefix: Mapped[str] = mapped_column(String(16))
    analyst_id: Mapped[str] = mapped_column(String(100), index=True)
    roles_json: Mapped[str] = mapped_column(Text)
    description: Mapped[Optional[str]] = mapped_column(String(300))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)


class AuditRecord(Base):
    """Append-only audit trail (UPDATE/DELETE are blocked by triggers on SQLite, see database.py)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    actor: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    key_id: Mapped[Optional[int]] = mapped_column(Integer)
    auth_mode: Mapped[str] = mapped_column(String(32))
    claimed_label: Mapped[Optional[str]] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(100), index=True)
    target_type: Mapped[Optional[str]] = mapped_column(String(50))
    target_id: Mapped[Optional[str]] = mapped_column(String(100), index=True)
    outcome: Mapped[str] = mapped_column(String(32))
    details_json: Mapped[str] = mapped_column(Text, default="{}")


# --- evidence enrichment ----------------------------------------------------------------


class EvidenceAdditionRecord(Base):
    """Evidence added after analysis: real tool output, analyst input, or (kind=simulated) synthetic data."""

    __tablename__ = "evidence_additions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(64))
    evidence_key: Mapped[str] = mapped_column(String(64))
    payload_json: Mapped[str] = mapped_column(Text)
    created_by: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class ToolInvocationRecord(Base):
    """One call to a (real or simulated) tool; no raw tool output is stored here."""

    __tablename__ = "tool_invocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    purpose: Mapped[str] = mapped_column(String(32))
    reference: Mapped[Optional[str]] = mapped_column(String(100))
    tool: Mapped[str] = mapped_column(String(32))
    operation: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    synthetic: Mapped[bool] = mapped_column(Boolean)
    error_category: Mapped[Optional[str]] = mapped_column(String(64))
    duration_ms: Mapped[float] = mapped_column(Float)
    actor: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


# --- playbook library ---------------------------------------------------------------------


class PlaybookRecord(Base):
    """A library entry. The embedding is stored so the vector index can be rebuilt at startup."""

    __tablename__ = "playbooks"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    intent_label: Mapped[str] = mapped_column(String(64), index=True)
    origin: Mapped[str] = mapped_column(String(16))
    playbook_json: Mapped[str] = mapped_column(Text)
    structural_hash: Mapped[str] = mapped_column(String(64), index=True)
    required_json: Mapped[str] = mapped_column(Text)
    techniques_json: Mapped[str] = mapped_column(Text, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    execution_count: Mapped[int] = mapped_column(Integer, default=0)
    deprioritized: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding_model: Mapped[str] = mapped_column(String(200))
    embedding_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)


# --- response plans and runs ----------------------------------------------------------------


class ResponsePlanRecord(Base):
    __tablename__ = "response_plans"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    plan_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)


class PlaybookRunRecord(Base):
    __tablename__ = "playbook_runs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("response_plans.id", ondelete="CASCADE"), index=True)
    incident_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32))
    run_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)


# --- routing decisions and background jobs ----------------------------------------------------


class RoutingDecisionRecord(Base):
    __tablename__ = "routing_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    incident_id: Mapped[str] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"), index=True)
    action: Mapped[str] = mapped_column(String(32))
    decision_json: Mapped[str] = mapped_column(Text)
    created_by: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class BanditArmRecord(Base):
    """The LinUCB statistics of one routing arm (A_a and b_a, paper Eq. 8)."""

    __tablename__ = "bandit_arms"

    arm: Mapped[str] = mapped_column(String(32), primary_key=True)
    a_json: Mapped[str] = mapped_column(Text)
    b_json: Mapped[str] = mapped_column(Text)
    pulls: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)


class RoutingRewardRecord(Base):
    """One reward credited to a routing decision (paper Eq. 9); the router's training log."""

    __tablename__ = "routing_rewards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    decision_id: Mapped[int] = mapped_column(Integer, index=True, unique=True)
    incident_id: Mapped[str] = mapped_column(String(64), index=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    arm: Mapped[str] = mapped_column(String(32))
    terminal: Mapped[bool] = mapped_column(Boolean)
    reward: Mapped[float] = mapped_column(Float)
    features_json: Mapped[str] = mapped_column(Text)
    components_json: Mapped[str] = mapped_column(Text, default="{}")
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    incident_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    progress: Mapped[Optional[str]] = mapped_column(String(300))
    result_json: Mapped[Optional[str]] = mapped_column(Text)
    error_json: Mapped[Optional[str]] = mapped_column(Text)
    created_by: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime)
