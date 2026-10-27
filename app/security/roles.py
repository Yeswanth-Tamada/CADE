"""Roles and the permissions they grant. Checked server-side on every protected route.

Separation of duties is deliberate: approving a destructive action and executing it are
different permissions, and live (non-simulated) execution needs its own role.
"""

from __future__ import annotations

from enum import Enum


class Permission(str, Enum):
    READ = "read"  # incidents, analyses, plans, runs, library, vocabulary
    SUBMIT_INCIDENT = "submit_incident"
    ANALYZE_INCIDENT = "analyze_incident"
    ENRICH_EVIDENCE = "enrich_evidence"  # run enrichment tools, add analyst-supplied evidence
    RESPOND = "respond"  # routing, synthesis, building response plans
    SUBMIT_FEEDBACK = "submit_feedback"
    APPROVE_ACTIONS = "approve_actions"
    EXECUTE_SIMULATED = "execute_simulated"
    EXECUTE_LIVE = "execute_live"
    MANAGE_LIBRARY = "manage_library"
    READ_AUDIT = "read_audit"


class Role(str, Enum):
    VIEWER = "viewer"
    ANALYST = "analyst"
    APPROVER = "approver"
    EXECUTOR = "executor"
    LIVE_EXECUTOR = "live_executor"
    LIBRARY_ADMIN = "library_admin"
    ADMIN = "admin"


ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: frozenset({Permission.READ}),
    Role.ANALYST: frozenset(
        {
            Permission.READ,
            Permission.SUBMIT_INCIDENT,
            Permission.ANALYZE_INCIDENT,
            Permission.ENRICH_EVIDENCE,
            Permission.RESPOND,
            Permission.SUBMIT_FEEDBACK,
        }
    ),
    Role.APPROVER: frozenset({Permission.READ, Permission.APPROVE_ACTIONS}),
    Role.EXECUTOR: frozenset({Permission.READ, Permission.EXECUTE_SIMULATED}),
    Role.LIVE_EXECUTOR: frozenset({Permission.READ, Permission.EXECUTE_SIMULATED, Permission.EXECUTE_LIVE}),
    Role.LIBRARY_ADMIN: frozenset({Permission.READ, Permission.MANAGE_LIBRARY}),
    Role.ADMIN: frozenset(Permission),
}

# With authentication disabled (local development only) nobody can be identified, so the
# operations whose value depends on knowing *who* acted are refused: fail closed.
AUTH_DISABLED_PERMISSIONS: frozenset[Permission] = frozenset(Permission) - {
    Permission.APPROVE_ACTIONS,
    Permission.EXECUTE_LIVE,
}


def permissions_for(roles: frozenset[Role]) -> frozenset[Permission]:
    granted: set[Permission] = set()
    for role in roles:
        granted |= ROLE_PERMISSIONS[role]
    return frozenset(granted)
