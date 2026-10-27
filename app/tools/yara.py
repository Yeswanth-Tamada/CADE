"""YARA command-line adapter.

Scans a single file with a configured rules file. To stop the API from being used to read
arbitrary files, the target must resolve (symlinks included) to an existing regular file inside
one of the configured scan roots. The scan never modifies the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from app.core.errors import ToolError, ToolInputError
from app.schemas.tools import OperationSpec, ToolObservation, ToolResult
from app.tools import artifacts as A
from app.tools.base import CommandRunner, CommandTool


class YaraTool(CommandTool):
    name = "yara"

    def __init__(self, *, rules_path: Path, scan_roots: list[Path], binary: str = "yara",
                 timeout_seconds: float = 60.0, runner: Optional[CommandRunner] = None) -> None:  # fmt: skip
        super().__init__(timeout_seconds=timeout_seconds, runner=runner)
        self.binary = binary
        self.rules_path = rules_path
        self.scan_roots = [root.resolve() for root in scan_roots]

    def _allowed_path(self, target: str) -> Path:
        try:
            path = Path(target).resolve(strict=True)
        except (OSError, RuntimeError):
            raise ToolInputError("the file to scan does not exist") from None
        if not path.is_file():
            raise ToolInputError("the path to scan is not a regular file")
        if not any(path.is_relative_to(root) for root in self.scan_roots):
            raise ToolInputError("the file is outside the configured YARA scan roots")
        return path

    async def run(
        self, spec: OperationSpec, target_type: str, target: str, context: dict[str, str], params: dict[str, Any]
    ) -> ToolResult:
        path = self._allowed_path(target)
        result = await self.execute([self.binary, "--", str(self.rules_path), str(path)])
        if result.returncode != 0:
            raise ToolError(f"yara exited with status {result.returncode}")
        matches = sorted({line.split()[0] for line in result.stdout.splitlines() if line.strip()})
        summary = (f"YARA: {len(matches)} rule(s) matched the file ({', '.join(matches[:10])})" if matches
                   else "YARA: no rule matched the file")  # fmt: skip
        return self.result(spec, target_type, outputs={A.YARA_MATCHES: matches},
                           observations=[ToolObservation(category=c, description=summary) for c in spec.categories],
                           findings=[summary])  # fmt: skip

    async def health(self) -> dict[str, Any]:
        return {"status": "configured" if self.rules_path.exists() else "misconfigured",
                "scan_roots": len(self.scan_roots)}  # fmt: skip
