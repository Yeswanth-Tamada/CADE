"""Run blocking calls (database, subprocesses, CPU-bound search) off the event loop."""

from __future__ import annotations

from functools import partial

from anyio import to_thread


async def run_sync(func, *args, **kwargs):
    return await to_thread.run_sync(partial(func, *args, **kwargs))
