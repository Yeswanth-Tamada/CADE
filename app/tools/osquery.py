"""osquery adapter (``osqueryi``) for the machine CADE runs on.

``osqueryi`` only sees the local host, so requests for any other host are refused (reported as
unavailable) instead of silently answering about the wrong machine. Fleet-wide querying would
need an osquery fleet manager and is not implemented. All SQL is fixed text: no request value is
ever interpolated into a query.
"""

from __future__ import annotations

import json
import platform
import socket
from typing import Any, Optional

from app.core.errors import ToolError, ToolUnavailableError
from app.schemas.enums import ToolStatus
from app.schemas.tools import OperationSpec, ToolObservation, ToolResult
from app.tools import artifacts as A
from app.tools.base import CommandRunner, CommandTool

_WINDOWS = platform.system() == "Windows"
_QUERIES: dict[str, tuple[str, str]] = {
    "processes": (A.PROCESS_LIST, "SELECT pid, name, path, cmdline, parent FROM processes LIMIT 200;"),
    "scheduled_tasks": (
        A.SCHEDULED_TASKS,
        "SELECT name, action, path, enabled, last_run_time FROM scheduled_tasks LIMIT 200;"
        if _WINDOWS
        else "SELECT command, path, minute, hour FROM crontab LIMIT 200;",
    ),
    "listening_ports": (
        A.LISTENING_PORTS,
        "SELECT lp.port, lp.protocol, lp.address, p.name FROM listening_ports lp "
        "LEFT JOIN processes p USING (pid) LIMIT 200;",
    ),
    "startup_items": (A.STARTUP_ITEMS, "SELECT name, path, source, status FROM startup_items LIMIT 200;"),
    "local_users": (A.USER_LIST, "SELECT username, uid, description FROM users LIMIT 200;"),
}


class OsqueryTool(CommandTool):
    name = "osquery"

    def __init__(self, *, binary: str = "osqueryi", timeout_seconds: float = 30.0,
                 runner: Optional[CommandRunner] = None, local_hostnames: Optional[set[str]] = None) -> None:  # fmt: skip
        super().__init__(timeout_seconds=timeout_seconds, runner=runner)
        self.binary = binary
        names = local_hostnames or {socket.gethostname(), socket.getfqdn()}
        self.local_hostnames = {n.lower() for n in names} | {n.lower().split(".")[0] for n in names}

    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        if target.lower() not in self.local_hostnames:
            raise ToolUnavailableError("osqueryi can only query the host CADE runs on, not the requested host")
        artifact, sql = _QUERIES[spec.name]
        result = await self.execute([self.binary, "--json", sql])
        if result.returncode != 0:
            raise ToolError(f"osqueryi exited with status {result.returncode}")
        try:
            rows = json.loads(result.stdout or "[]")
        except ValueError:
            raise ToolError("osqueryi returned output that is not JSON") from None
        if not isinstance(rows, list):
            raise ToolError("osqueryi returned an unexpected JSON value")
        if not rows:
            return self.result(spec, target_type, status=ToolStatus.NO_DATA, message="the query returned no rows")
        summary = f"osquery returned {len(rows)} {spec.name.replace('_', ' ')} rows from the local host"
        return self.result(spec, target_type, outputs={artifact: rows[:200]},
                           observations=[ToolObservation(category=c, description=summary) for c in spec.categories],
                           findings=[summary])  # fmt: skip

    async def health(self) -> dict[str, Any]:
        return {"status": "configured", "binary": self.binary, "scope": "local host only"}
