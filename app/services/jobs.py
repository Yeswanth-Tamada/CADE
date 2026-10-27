"""In-process background jobs for long operations (synthesis makes many LLM calls).

Job state is persisted, so clients can poll ``GET /jobs/{id}`` and jobs left unfinished by a
previous process are marked ``interrupted`` at startup. This is deliberately simple (no external
queue): one process, asyncio tasks, at most one active job of a kind per incident.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, Optional

from app.core.errors import CADEError, IncidentConflictError
from app.core.logging import get_logger, log_event
from app.core.threads import run_sync
from app.schemas.jobs import Job, JobStatus
from app.storage.routing_repository import JobRepository

logger = get_logger("jobs")

Progress = Callable[[str], Awaitable[None]]


class JobManager:
    def __init__(self, repository: JobRepository) -> None:
        self.repository = repository
        self._tasks: set[asyncio.Task] = set()

    def recover(self) -> int:
        return self.repository.interrupt_unfinished()

    async def start(self, kind: str, incident_id: Optional[str], created_by: Optional[str],
                    work: Callable[[Progress], Awaitable[dict[str, Any]]]) -> Job:  # fmt: skip
        if incident_id and await run_sync(self.repository.active_for, incident_id, kind):
            raise IncidentConflictError(f"a '{kind}' job is already running for this incident", incident_id=incident_id)
        job = await run_sync(self.repository.create, f"JOB-{uuid.uuid4().hex[:16].upper()}", kind, incident_id, created_by)
        task = asyncio.create_task(self._run(job.job_id, kind, incident_id, work))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def get(self, job_id: str) -> Job:
        return await run_sync(self.repository.get, job_id)

    async def _run(self, job_id: str, kind: str, incident_id: Optional[str], work) -> None:
        async def progress(message: str) -> None:
            await run_sync(self.repository.update, job_id, progress=message)

        await run_sync(self.repository.update, job_id, status=JobStatus.RUNNING, progress="started")
        try:
            result = await work(progress)
        except Exception as exc:  # noqa: BLE001 - every failure must end the job
            error = ({"category": exc.category, "message": exc.message} if isinstance(exc, CADEError)
                     else {"category": "internal_error", "message": "unexpected error"})  # fmt: skip
            log_event(logger, "job failed", level=logging.ERROR if not isinstance(exc, CADEError) else logging.WARNING,
                      incident_id=incident_id, operation=kind, outcome="failure", error_category=error["category"],
                      exception_type=type(exc).__name__)  # fmt: skip
            await run_sync(self.repository.update, job_id, status=JobStatus.FAILED, error=error, progress="failed")
            return
        await run_sync(self.repository.update, job_id, status=JobStatus.SUCCEEDED, result=result, progress="done")

    async def drain(self) -> None:
        """Wait for running jobs (used by tests and on shutdown)."""
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
