"""API key generation, hashing and a small management CLI.

Keys are random 256-bit tokens. Only their SHA-256 hash is stored, so a database leak does
not reveal usable keys; the key itself is printed once, at creation.

    python -m app.security.keys create --analyst alice --roles analyst,approver --description "SOC tier 2"
    python -m app.security.keys list
    python -m app.security.keys revoke --key-id 3
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
from typing import Optional

from app.schemas.security import ApiKeyInfo
from app.security.roles import Role
from app.storage.security_repository import ApiKeyRepository

KEY_PREFIX = "cade_"
_ANALYST_ID_MAX = 100


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def issue_key(
    repository: ApiKeyRepository, analyst_id: str, roles: list[Role], description: Optional[str] = None
) -> tuple[str, ApiKeyInfo]:
    """Create a key and return ``(key, info)``. The plaintext key is not stored anywhere."""
    analyst_id = analyst_id.strip()
    if not analyst_id or len(analyst_id) > _ANALYST_ID_MAX or not analyst_id.isprintable():
        raise ValueError(f"analyst id must be 1-{_ANALYST_ID_MAX} printable characters")
    if not roles:
        raise ValueError("at least one role is required")
    key = generate_key()
    info = repository.create(
        key_hash=hash_key(key), key_prefix=key[:12], analyst_id=analyst_id, roles=list(roles), description=description
    )
    return key, info


def parse_roles(value: str) -> list[Role]:
    try:
        return sorted({Role(part.strip()) for part in value.split(",") if part.strip()}, key=lambda r: r.value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"roles must be a comma-separated subset of: {', '.join(r.value for r in Role)}")


def main(argv: Optional[list[str]] = None) -> int:
    from app.core.config import get_settings
    from app.storage.database import Database

    parser = argparse.ArgumentParser(prog="python -m app.security.keys", description="Manage CADE API keys")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="issue a new API key (printed once)")
    create.add_argument("--analyst", required=True, help="identity recorded in the audit log")
    create.add_argument("--roles", required=True, type=parse_roles, help="comma-separated roles")
    create.add_argument("--description")
    commands.add_parser("list", help="list keys (never shows the keys themselves)")
    revoke = commands.add_parser("revoke", help="revoke a key")
    revoke.add_argument("--key-id", type=int, required=True)
    args = parser.parse_args(argv)

    database = Database(get_settings().database_url)
    database.create_schema()
    repository = ApiKeyRepository(database)
    try:
        if args.command == "create":
            key, info = issue_key(repository, args.analyst, args.roles, args.description)
            print(f"key_id={info.key_id} analyst={info.analyst_id} roles={','.join(r.value for r in info.roles)}")
            print(f"API key (shown once, store it securely): {key}")
        elif args.command == "list":
            for info in repository.list():
                state = "active" if info.active else f"revoked {info.revoked_at.isoformat()}"
                print(f"{info.key_id}\t{info.key_prefix}...\t{info.analyst_id}\t"
                      f"{','.join(r.value for r in info.roles)}\t{state}")  # fmt: skip
        else:
            info = repository.revoke(args.key_id)
            if info is None:
                print(f"no key with id {args.key_id}")
                return 1
            print(f"revoked key_id={info.key_id} ({info.analyst_id})")
    finally:
        database.dispose()
    return 0


if __name__ == "__main__":
    sys.exit(main())
