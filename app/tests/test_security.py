"""Authentication, server-side authorization and the append-only audit log (milestone M1)."""

from __future__ import annotations

import re

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from app.core.config import get_settings
from app.core.errors import ConfigurationError
from app.main import create_app
from app.security import keys as keys_cli
from app.security.keys import hash_key
from app.security.roles import AUTH_DISABLED_PERMISSIONS, ROLE_PERMISSIONS, Permission, Role
from app.tests.conftest import SYNTHETIC_DESCRIPTION, auth, good_output, issue_key

PUBLIC_PATHS = {"/health"}


def _with(settings, **changes):
    """A validated copy of ``settings`` (``model_copy(update=...)`` would skip validation)."""
    return type(settings).model_validate({**settings.model_dump(), **changes})


def _protected_routes(app) -> list[tuple[str, str]]:
    """Every documented API operation except the public ones (read from the OpenAPI schema)."""
    routes = []
    for path, operations in app.openapi()["paths"].items():
        if path in PUBLIC_PATHS:
            continue
        concrete = re.sub(r"\{[^}]+\}", "INC-TEST", path)
        routes.extend((method.upper(), concrete) for method in operations)
    return routes


def test_every_route_except_health_requires_an_api_key(make_client):
    client = make_client(roles=())
    routes = _protected_routes(client.app)
    assert routes, "no protected routes found"
    for method, path in routes:
        response = client.request(method, path)
        assert response.status_code == 401, f"{method} {path} -> {response.status_code}"
        assert response.json()["error"]["category"] == "unauthenticated"


@pytest.mark.parametrize(
    "header",
    ["Bearer", "Bearer ", "Basic dXNlcjpwYXNz", "Bearer cade_not-a-real-key", "cade_raw-token-without-scheme"],
)
def test_malformed_or_unknown_credentials_are_rejected(make_client, header):
    client = make_client(roles=())
    response = client.get("/incidents/INC-TEST", headers={"Authorization": header})
    assert response.status_code == 401


def test_claimed_analyst_header_alone_is_not_authentication(make_client):
    client = make_client(roles=())
    response = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION},
                           headers={"X-CADE-Analyst": "admin"})  # fmt: skip
    assert response.status_code == 401


def test_revoked_key_is_rejected(make_client):
    client = make_client(roles=())
    key = issue_key(client, "analyst")
    assert client.get("/incidents/INC-NONE", headers=auth(key)).status_code == 404  # authenticated
    from app.storage.security_repository import ApiKeyRepository

    repository = ApiKeyRepository(client.app.state.database)
    (info,) = [k for k in repository.list() if k.analyst_id == "analyst-test"]
    repository.revoke(info.key_id)
    assert client.get("/incidents/INC-NONE", headers=auth(key)).status_code == 401


@pytest.mark.parametrize(
    "role, method, path, body",
    [
        ("viewer", "POST", "/incidents", {"description": SYNTHETIC_DESCRIPTION}),
        ("approver", "POST", "/incidents", {"description": SYNTHETIC_DESCRIPTION}),
        ("executor", "POST", "/incidents/INC-TEST/analyze", None),
        ("analyst", "GET", "/audit", None),
        ("library_admin", "POST", "/incidents/INC-TEST/analyze", None),
    ],
)
def test_wrong_role_is_forbidden_server_side(make_client, role, method, path, body):
    client = make_client(roles=(role,))
    response = client.request(method, path, json=body)
    assert response.status_code == 403
    assert response.json()["error"]["category"] == "forbidden"


def test_forbidden_attempt_is_audited_with_real_identity(make_client):
    client = make_client(roles=("viewer",))
    response = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION},
                           headers={"X-CADE-Analyst": "someone-else"})  # fmt: skip
    assert response.status_code == 403
    (event,) = client.app.state.audit.repository.list(action="authorization.denied")
    assert event.actor == "analyst-test"  # identity from the key...
    assert event.claimed_label == "someone-else"  # ...the header is only a recorded label
    assert event.outcome == "denied"
    assert event.details == {"permission": "submit_incident"}


def test_analyst_workflow_is_audited(make_client):
    client = make_client([good_output()], roles=("analyst",))
    incident_id = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION}).json()["incident_id"]
    assert client.post(f"/incidents/{incident_id}/analyze").status_code == 200

    events = client.app.state.audit.repository.list(target_id=incident_id)
    assert [(e.action, e.outcome) for e in reversed(events)] == [
        ("incident.create", "success"),
        ("incident.analyze", "success"),
    ]
    assert all(e.actor == "analyst-test" and e.auth_mode == "api_key" and e.key_id for e in events)
    # Audit details never contain incident text.
    assert SYNTHETIC_DESCRIPTION[:30] not in str([e.details for e in events])


def test_audit_endpoint_requires_read_audit_and_lists_events(make_client):
    client = make_client(roles=("admin",))
    client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION})
    events = client.get("/audit").json()
    assert events[0]["action"] == "incident.create"


def test_audit_log_is_append_only_in_the_database(make_client):
    client = make_client()
    client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION})
    engine = client.app.state.database.engine
    for statement in ("UPDATE audit_log SET outcome = 'tampered'", "DELETE FROM audit_log"):
        with pytest.raises(DatabaseError, match="append-only"):
            with engine.begin() as conn:
                conn.execute(text(statement))


def test_only_key_hashes_are_stored(make_client):
    client = make_client(roles=())
    key = issue_key(client, "viewer")
    with client.app.state.database.engine.connect() as conn:
        rows = conn.execute(text("SELECT key_hash, key_prefix FROM api_keys")).all()
    assert [r.key_hash for r in rows] == [hash_key(key)]
    assert key not in str(rows) and len(rows[0].key_prefix) < len(key)


def test_health_is_public_but_minimal_without_a_key(make_client):
    anonymous = make_client(roles=()).get("/health").json()
    assert "dependencies" not in anonymous and "environment" not in anonymous["api"]
    detailed = make_client(roles=("viewer",)).get("/health").json()
    assert "dependencies" in detailed


def test_auth_cannot_be_disabled_in_production(settings):
    production = _with(settings, app_env="production", auth_enabled=False)
    with pytest.raises(ConfigurationError):
        create_app(production)


def test_docs_are_not_served_in_production(make_client, settings):
    client = make_client(app_settings=_with(settings, app_env="production"), roles=())
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_auth_disabled_mode_never_allows_approval_or_live_execution(make_client, settings):
    assert Permission.APPROVE_ACTIONS not in AUTH_DISABLED_PERMISSIONS
    assert Permission.EXECUTE_LIVE not in AUTH_DISABLED_PERMISSIONS
    client = make_client([good_output()], app_settings=_with(settings, auth_enabled=False), roles=())
    response = client.post("/incidents", json={"description": SYNTHETIC_DESCRIPTION})
    assert response.status_code == 201
    (event,) = client.app.state.audit.repository.list(action="incident.create")
    assert event.auth_mode == "auth_disabled"


def test_separation_of_duties_in_role_definitions():
    assert Permission.APPROVE_ACTIONS not in ROLE_PERMISSIONS[Role.ANALYST]
    assert Permission.APPROVE_ACTIONS not in ROLE_PERMISSIONS[Role.EXECUTOR]
    assert Permission.EXECUTE_SIMULATED not in ROLE_PERMISSIONS[Role.APPROVER]
    assert Permission.EXECUTE_LIVE not in ROLE_PERMISSIONS[Role.EXECUTOR]
    assert ROLE_PERMISSIONS[Role.ADMIN] == frozenset(Permission)


def test_key_cli_create_list_revoke(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CADE_DATABASE_URL", f"sqlite:///{(tmp_path / 'keys.db').as_posix()}")
    get_settings.cache_clear()
    try:
        assert keys_cli.main(["create", "--analyst", "alice", "--roles", "analyst,approver"]) == 0
        created = capsys.readouterr().out
        key = created.split("store it securely): ")[1].strip()
        assert key.startswith("cade_")

        assert keys_cli.main(["list"]) == 0
        listed = capsys.readouterr().out
        assert "alice" in listed and "analyst,approver" in listed and key not in listed

        assert keys_cli.main(["revoke", "--key-id", "1"]) == 0
        assert keys_cli.main(["revoke", "--key-id", "99"]) == 1
        assert "revoked" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()


def test_key_cli_rejects_unknown_roles(capsys):
    with pytest.raises(SystemExit):
        keys_cli.main(["create", "--analyst", "bob", "--roles", "superuser"])
