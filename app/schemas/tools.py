"""Contracts for security-tool operations, requests and results."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.enums import EvidenceCategory, ToolStatus


class OperationSpec(BaseModel):
    """One operation a tool offers. Shared by the real adapter and its simulator."""

    model_config = ConfigDict(frozen=True)

    tool: str
    name: str
    description: str
    accepts: tuple[str, ...]  # artifact types any one of which can be the operation's target
    produces: tuple[str, ...]  # artifact types in the result's outputs
    requires: tuple[str, ...] = ()  # further input artifacts that must all be supplied as context
    categories: tuple[EvidenceCategory, ...] = ()  # evidence categories the result is about
    destructive: bool = False  # changes the environment (isolation, blocking, ...)
    params: dict[str, tuple[Any, ...]] = Field(default_factory=dict)  # name -> allowed values (first = default)

    @property
    def key(self) -> str:
        return f"{self.tool}.{self.name}"

    @property
    def enrichment(self) -> bool:
        """Read-only operations can be used to collect evidence."""
        return not self.destructive and bool(self.categories)

    def resolve_params(self, params: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Defaults for omitted params; unknown names or disallowed values are rejected."""
        params = params or {}
        unknown = set(params) - set(self.params)
        if unknown:
            raise ValueError(f"unknown parameter(s) for {self.key}: {', '.join(sorted(unknown))}")
        resolved = {}
        for name, allowed in self.params.items():
            value = params.get(name, allowed[0])
            if value not in allowed:
                raise ValueError(f"parameter '{name}' of {self.key} must be one of {list(allowed)}")
            resolved[name] = value
        return resolved


class ToolRequest(BaseModel):
    operation: str
    target_type: str
    target: str
    context: dict[str, str] = Field(default_factory=dict)  # values for OperationSpec.requires
    params: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    """A fact a tool reported about one evidence category. Bounded, no raw tool dumps."""

    category: EvidenceCategory
    description: str = Field(min_length=1, max_length=1000)


class ToolResult(BaseModel):
    tool: str
    operation: str
    target_type: str
    status: ToolStatus
    synthetic: bool  # True for simulated results -- they never count as real observations
    outputs: dict[str, Any] = Field(default_factory=dict)
    observations: list[ToolObservation] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    error_category: Optional[str] = None
    message: Optional[str] = Field(default=None, max_length=500)
    duration_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _consistency(self) -> "ToolResult":
        if self.status == ToolStatus.SUCCESS and not (self.outputs or self.observations or self.findings):
            raise ValueError("a successful tool result must carry outputs, observations or findings")
        if self.status != ToolStatus.SUCCESS and self.observations:
            raise ValueError("only successful tool results may carry observations")
        if self.synthetic and any(not f.startswith("SYNTHETIC") for f in self.findings):
            raise ValueError("findings of simulated tools must be labelled SYNTHETIC")
        return self


class ToolInfo(BaseModel):
    name: str
    description: str
    configured: bool  # a real adapter is configured
    effective: str  # what calls go to right now: "simulated", "real" or "unavailable"
    operations: list[OperationSpec]
