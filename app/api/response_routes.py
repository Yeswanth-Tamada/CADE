"""Routing, response plans, approvals, verification, execution, runs and jobs.

Thin wrappers around ``ResponseService`` / ``PlanService``. Long-running operations (``respond``,
``synthesize``) run as background jobs unless ``wait=true``.
"""

from __future__ import annotations

from fastapi import APIRouter, Path, Query, Request
from fastapi.responses import JSONResponse

from app.api.routes import IncidentIdPath
from app.schemas.jobs import IncidentResponse, Job, ResponseOutcome
from app.schemas.response import ApprovalRequest, FeedbackRequest, PlaybookRun, ResponsePlan, VerificationRequest
from app.schemas.routing import StoredRoutingDecision
from app.security.auth import Principal, require
from app.security.roles import Permission
from app.services.plan_service import PlanService
from app.services.response_service import ResponseService

router = APIRouter(tags=["response"])

PlanIdPath = Path(pattern=r"^PLAN-[0-9A-F]{16}$")
RunIdPath = Path(pattern=r"^RUN-[0-9A-F]{16}$")
JobIdPath = Path(pattern=r"^JOB-[0-9A-F]{16}$")

_RESPONSES = {
    401: {"description": "Missing, invalid or revoked API key"},
    403: {"description": "The API key's roles do not allow this operation"},
    404: {"description": "Not found"},
    409: {"description": "Conflict (no usable analysis, already executed, job running, ...)"},
}


def _responses(request: Request) -> ResponseService:
    return request.app.state.response_service


def _plans(request: Request) -> PlanService:
    return request.app.state.plan_service


@router.post("/incidents/{incident_id}/route", response_model=StoredRoutingDecision, responses=_RESPONSES)
async def route_incident(request: Request, incident_id: str = IncidentIdPath,
                         principal: Principal = require(Permission.RESPOND)) -> StoredRoutingDecision:  # fmt: skip
    """Compute (and record) the routing decision only: reuse, request evidence or synthesize (Eq. 1-5)."""
    return await _responses(request).route(incident_id, principal)


async def _start(request: Request, incident_id: str, principal: Principal, *, wait: bool, auto_acquire: bool,
                 force: bool):  # fmt: skip
    service = _responses(request)
    if wait:
        return await service.respond(incident_id, principal, auto_acquire=auto_acquire, force_synthesis=force)
    await request.app.state.evidence_service.usable_analysis(incident_id)  # fail fast (404/409) before queueing

    async def work(progress):
        outcome = await service.respond(incident_id, principal, auto_acquire=auto_acquire, force_synthesis=force,
                                        progress=progress)  # fmt: skip
        return outcome.model_dump(mode="json")

    job = await request.app.state.jobs.start("respond", incident_id, principal.analyst_id, work)
    return JSONResponse(status_code=202, content=job.model_dump(mode="json"))


@router.post("/incidents/{incident_id}/respond", response_model=ResponseOutcome,
             responses={**_RESPONSES, 202: {"model": Job, "description": "Job started (poll GET /jobs/{job_id})"}})  # fmt: skip
async def respond(
    request: Request,
    incident_id: str = IncidentIdPath,
    wait: bool = Query(default=False, description="run synchronously instead of as a background job"),
    auto_acquire: bool = Query(default=True, description="request missing evidence from the tools automatically"),
    principal: Principal = require(Permission.RESPOND),
):
    """Route the incident and act: reuse a library playbook, request evidence, or synthesize one with CADE."""
    return await _start(request, incident_id, principal, wait=wait, auto_acquire=auto_acquire, force=False)


@router.post("/incidents/{incident_id}/synthesize", response_model=ResponseOutcome,
             responses={**_RESPONSES, 202: {"model": Job, "description": "Job started"}})  # fmt: skip
async def synthesize(request: Request, incident_id: str = IncidentIdPath, wait: bool = Query(default=False),
                     principal: Principal = require(Permission.RESPOND)):  # fmt: skip
    """Force synthesis (the routing decision is still computed and recorded)."""
    return await _start(request, incident_id, principal, wait=wait, auto_acquire=True, force=True)


@router.get("/incidents/{incident_id}/response", response_model=IncidentResponse, responses=_RESPONSES)
async def incident_response(request: Request, incident_id: str = IncidentIdPath,
                            principal: Principal = require(Permission.READ)) -> IncidentResponse:  # fmt: skip
    """All routing decisions, plans and runs recorded for the incident."""
    return await _responses(request).history(incident_id)


@router.get("/jobs/{job_id}", response_model=Job, responses=_RESPONSES)
async def get_job(request: Request, job_id: str = JobIdPath, principal: Principal = require(Permission.READ)) -> Job:
    return await request.app.state.jobs.get(job_id)


@router.get("/response-plans/{plan_id}", response_model=ResponsePlan, responses=_RESPONSES)
async def get_plan(request: Request, plan_id: str = PlanIdPath, principal: Principal = require(Permission.READ)) -> ResponsePlan:
    return await _plans(request).get_plan(plan_id)


@router.post("/response-plans/{plan_id}/approve", response_model=ResponsePlan, responses=_RESPONSES)
async def approve(request: Request, body: ApprovalRequest, plan_id: str = PlanIdPath,
                  principal: Principal = require(Permission.APPROVE_ACTIONS)) -> ResponsePlan:  # fmt: skip
    """Approve destructive steps of this exact plan (recorded with approver, time and plan hash)."""
    return await _plans(request).approve(plan_id, body, principal)


@router.post("/response-plans/{plan_id}/verifications", response_model=ResponsePlan, responses=_RESPONSES)
async def verify(request: Request, body: VerificationRequest, plan_id: str = PlanIdPath,
                 principal: Principal = require(Permission.APPROVE_ACTIONS)) -> ResponsePlan:  # fmt: skip
    """Record an analyst's verdict for a verification step, naming the confirmed target(s)."""
    return await _plans(request).record_verification(plan_id, body, principal)


@router.post("/response-plans/{plan_id}/execute", response_model=PlaybookRun, responses=_RESPONSES)
async def execute(request: Request, plan_id: str = PlanIdPath,
                  principal: Principal = require(Permission.EXECUTE_SIMULATED)) -> PlaybookRun:  # fmt: skip
    """Execute the plan in the configured mode (simulate by default; live needs the execute_live role)."""
    return await _plans(request).execute(plan_id, principal)


@router.get("/playbook-runs/{run_id}", response_model=PlaybookRun, responses=_RESPONSES)
async def get_run(request: Request, run_id: str = RunIdPath, principal: Principal = require(Permission.READ)) -> PlaybookRun:
    return await _plans(request).get_run(run_id)


@router.post("/playbook-runs/{run_id}/feedback", response_model=PlaybookRun, responses=_RESPONSES)
async def feedback(request: Request, body: FeedbackRequest, run_id: str = RunIdPath,
                   principal: Principal = require(Permission.SUBMIT_FEEDBACK)) -> PlaybookRun:  # fmt: skip
    """Report whether the run resolved the incident and whether its findings were retained (library write-back)."""
    return await _plans(request).feedback(run_id, body, principal)
