"""Exception handlers that turn errors into sanitised JSON responses.

Every error response has the shape::

    {"error": {"category": "...", "message": "...", "incident_id": "...", "details": [...]}}

Stack traces and internal exception messages are never returned to clients.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import CADEError
from app.core.logging import get_logger, log_event

logger = get_logger("api")


def error_response(status: int, category: str, message: str, **extra) -> JSONResponse:
    body = {"category": category, "message": message}
    body.update({k: v for k, v in extra.items() if v})
    return JSONResponse(status_code=status, content={"error": body})


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(CADEError)
    async def handle_cade_error(request: Request, exc: CADEError) -> JSONResponse:
        log_event(
            logger,
            "request failed",
            level=logging.WARNING,
            incident_id=exc.incident_id,
            operation=f"{request.method} {request.url.path}",
            outcome="failure",
            error_category=exc.category,
            http_status=exc.http_status,
        )
        return error_response(
            exc.http_status, exc.category, exc.message, incident_id=exc.incident_id, details=exc.details
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")} for e in exc.errors()]
        return error_response(422, "validation_error", "request validation failed", details=details)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(exc.status_code, "http_error", str(exc.detail))

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        log_event(
            logger,
            "unhandled error",
            level=logging.ERROR,
            operation=f"{request.method} {request.url.path}",
            outcome="failure",
            error_category="internal_error",
            exception_type=type(exc).__name__,
        )
        return error_response(500, "internal_error", "internal server error")
