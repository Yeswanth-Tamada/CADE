"""Database engine/session management.

Initialise the schema explicitly with::

    python -m app.storage.database

The API also creates missing tables at startup (``Database.create_schema``). Phase 1 has no
migrations; add Alembic when the schema starts changing between releases.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.storage.models import Base


class Database:
    def __init__(self, url: str) -> None:
        parsed = make_url(url)
        connect_args: dict = {}
        if parsed.get_backend_name() == "sqlite":
            # FastAPI may run requests in worker threads; each session is used by one thread at a time.
            connect_args["check_same_thread"] = False
            if parsed.database and parsed.database != ":memory:":
                Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(url, connect_args=connect_args)
        if parsed.get_backend_name() == "sqlite":
            event.listen(self.engine, "connect", _enable_sqlite_foreign_keys)
        self._session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)
        if self.engine.dialect.name == "sqlite":
            # Make the audit trail append-only at the database level, not just by convention.
            with self.engine.begin() as conn:
                for operation in ("UPDATE", "DELETE"):
                    conn.execute(
                        text(
                            f"CREATE TRIGGER IF NOT EXISTS audit_log_no_{operation.lower()} "
                            f"BEFORE {operation} ON audit_log "
                            "BEGIN SELECT RAISE(ABORT, 'audit_log is append-only'); END"
                        )
                    )

    @contextmanager
    def session(self) -> Iterator[Session]:
        """A transactional scope: commit on success, roll back on any error."""
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    def dispose(self) -> None:
        self.engine.dispose()


def _enable_sqlite_foreign_keys(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


if __name__ == "__main__":
    from app.core.config import get_settings

    settings = get_settings()
    Database(settings.database_url).create_schema()
    print(f"Database schema initialised at {settings.database_url}")
