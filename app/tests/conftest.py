"""Shared test fixtures.

No test needs a running Ollama server or internet access: the LLM is replaced by
``MockLLMClient`` and Ollama HTTP calls by ``httpx.MockTransport``. All incidents are SYNTHETIC.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from app.core.config import PROJECT_ROOT, LLMProvider, Settings
from app.intent.attack_mapper import AttackDataset, AttackMapper
from app.intent.llm_client import LLMClient
from app.intent.mock_llm_client import MockItem, MockLLMClient
from app.main import create_app
from app.schemas.incident import IncidentSubmission, NormalizedIncident
from app.schemas.taxonomy import IntentTaxonomy
from app.storage.database import Database
from app.storage.repository import IncidentRepository

FIXTURES = Path(__file__).parent / "fixtures"
EXAMPLES = PROJECT_ROOT / "examples" / "synthetic_incidents"

# A SYNTHETIC incident used across the extraction and API tests.
SYNTHETIC_DESCRIPTION = (
    "SYNTHETIC: EDR alert on host WS-FIN-042. User jdoe opened Invoice_8841.docm; "
    "WINWORD.EXE spawned powershell.exe -enc SQBFAFgA. "
    "The host then connected to 203.0.113.45 on port 443."
)

GOOD_LLM_OUTPUT: dict[str, Any] = {
    "intent_label": "suspicious_process_execution",
    "intent_confidence": 0.8,
    "summary": "An Office document on WS-FIN-042 launched encoded PowerShell, followed by an outbound connection.",
    "entities": {
        "hosts": ["WS-FIN-042"],
        "users": ["jdoe"],
        "ips": ["203.0.113.45"],
        "domains": [],
        "file_hashes": [],
        "processes": ["WINWORD.EXE", "powershell.exe"],
    },
    "observed_evidence": [
        {
            "category": "process_telemetry",
            "description": "Word spawned PowerShell with an encoded command",
            "excerpt": "WINWORD.EXE spawned powershell.exe -enc SQBFAFgA",
        },
        {
            "category": "network_traffic",
            "description": "Outbound connection after execution",
            "excerpt": "connected to 203.0.113.45 on port 443",
        },
    ],
    "unconfirmed_evidence": [
        {
            "category": "command_line",
            "description": "The encoded command may download a second stage",
            "rationale": "Common pattern for macro droppers; not confirmed",
        }
    ],
    "missing_evidence": [
        {
            "category": "process_telemetry",
            "description": "Full process tree with decoded command line",
            "critical": True,
            "rationale": "Needed to know what PowerShell executed",
        },
        {"category": "dns", "description": "DNS queries from the host", "critical": False, "rationale": "Context"},
    ],
    "attack_techniques": [
        {"technique_id": "T1059.001", "confidence": 0.7, "justification": "Encoded PowerShell execution"},
        {"technique_id": "T1566.001", "confidence": 0.4, "justification": "Possible malicious attachment"},
    ],
    "uncertainties": ["Whether the PowerShell payload executed successfully is unknown"],
    "conflicts": [],
}


def good_output(**overrides: Any) -> dict[str, Any]:
    data = copy.deepcopy(GOOD_LLM_OUTPUT)
    data.update(overrides)
    return data


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        # a developer's .env (real tool URLs and keys) must never reach the tests
        _env_file=None,
        app_env="test",
        log_level="WARNING",
        database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        llm_provider=LLMProvider.MOCK,
        attack_dataset_path=tmp_path / "no-attack-dataset.json",
        llm_max_attempts=2,
        # scripted mock responses answer one analysis call; two-stage analysis has its own tests
        intent_two_stage=False,
        # offline and fast: no ML packages, small search, reproducible
        embedder="hashing",
        vector_index="numpy",
        os_judge_mode="heuristic",
        generation_pool_size=4,
        population_size=8,
        max_generations=3,
        random_seed=7,
    )


@pytest.fixture(scope="session")
def taxonomy() -> IntentTaxonomy:
    return IntentTaxonomy.load(PROJECT_ROOT / "app" / "data" / "intent_taxonomy.json")


@pytest.fixture(scope="session")
def attack_mapper() -> AttackMapper:
    return AttackMapper(AttackDataset.from_stix_file(FIXTURES / "attack_fixture.json"))


@pytest.fixture
def unverified_mapper(tmp_path: Path) -> AttackMapper:
    return AttackMapper.from_path(tmp_path / "missing.json")


def issue_key(client: TestClient, *roles: str, analyst: str = "analyst-test") -> str:
    """Create an API key in the client's database and return the plaintext key."""
    from app.security.keys import issue_key as _issue
    from app.security.roles import Role
    from app.storage.security_repository import ApiKeyRepository

    key, _ = _issue(ApiKeyRepository(client.app.state.database), analyst, [Role(r) for r in roles])
    return key


def auth(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def with_settings(settings: Settings, **changes: Any) -> Settings:
    """A validated copy of ``settings`` (``model_copy(update=...)`` would skip validation)."""
    return type(settings).model_validate({**settings.model_dump(), **changes})


@pytest.fixture
def make_client(settings: Settings, attack_mapper: AttackMapper) -> Iterator[Callable[..., TestClient]]:
    """Factory: ``make_client(responses=[...])`` -> a TestClient whose LLM replays ``responses``.

    The client is authenticated with an API key holding ``roles`` (admin by default);
    pass ``roles=()`` for an anonymous client.
    """
    clients: list[TestClient] = []

    def factory(
        responses: Optional[list[MockItem]] = None,
        *,
        llm_client: Optional[LLMClient] = None,
        mapper: Optional[AttackMapper] = attack_mapper,
        roles: tuple[str, ...] = ("admin",),
        app_settings: Optional[Settings] = None,
        real_tools: Optional[dict] = None,
    ) -> TestClient:
        app = create_app(
            app_settings or settings,
            llm_client=llm_client or MockLLMClient(responses),
            attack_mapper=mapper,
            real_tools=real_tools if real_tools is not None else {},
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        if roles:
            client.headers.update(auth(issue_key(client, *roles)))
        return client

    yield factory
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def repository(settings: Settings) -> Iterator[IncidentRepository]:
    db = Database(settings.database_url)
    db.create_schema()
    yield IncidentRepository(db)
    db.dispose()


def make_incident(description: str = SYNTHETIC_DESCRIPTION, **fields: Any) -> NormalizedIncident:
    from app.ingestion.normalizer import normalize_submission

    submission = IncidentSubmission(description=description, **fields)
    return normalize_submission(
        submission, max_description_chars=50_000, received_at=datetime(2026, 10, 19, 9, 0, tzinfo=timezone.utc)
    )


def load_example(name: str) -> dict[str, Any]:
    return json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
