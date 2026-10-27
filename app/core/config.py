"""Application configuration loaded from environment variables (and an optional .env file).

Every setting uses the ``CADE_`` prefix, e.g. ``CADE_OLLAMA_MODEL``. See ``.env.example``.
No secrets are required in Phase 1; never hardcode credentials here.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.enums import ExecutionMode, ToolFailurePolicy

# Project root = the directory that contains the ``app`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    MOCK = "mock"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CADE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CADE Incident Understanding Service"
    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    database_url: str = "sqlite:///./cade.db"

    # API keys are required on every route except /health. Disabling authentication is for local
    # development only: it is refused in production, and even then approvals and live execution
    # stay impossible because nobody can be identified.
    auth_enabled: bool = True

    llm_provider: LLMProvider = LLMProvider.OLLAMA
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2:3b"
    llm_request_timeout_seconds: float = Field(default=120.0, gt=0, le=1800)
    llm_health_timeout_seconds: float = Field(default=3.0, gt=0, le=60)
    llm_max_attempts: int = Field(default=2, ge=1, le=5)
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    attack_dataset_path: Path = Path("app/data/attack/enterprise-attack.json")
    # List dataset technique names and well-known technique artifacts (app/data/attack_artifact_hints.json)
    # found in the incident text as hints in the prompt (never as results).
    attack_prompt_hints: bool = True
    # Paper 968 Section 3: a focused intent call (activity -> ATT&CK tactic -> label), then the evidence
    # profile. False = one combined call (the label is chosen together with the evidence).
    intent_two_stage: bool = True

    # --- playbook synthesis (paper Sections 5, 6.3 / Table 3) ------------------------------
    # Three independently configured models: intent extraction (ollama_model), playbook
    # generation and operational-soundness judging. The judge should be from a different
    # model family than the generator (a startup warning is logged otherwise).
    generator_model: str = "qwen3:14b"
    judge_model: str = "gemma3:12b"
    # "llm": the judge model scores OS; "heuristic": OS_H only (always used with the mock provider)
    os_judge_mode: str = Field(default="llm", pattern=r"^(llm|heuristic)$")
    os_judge_samples: int = Field(default=3, ge=1, le=10)
    # m: candidates judged per generation (ranked by provisional fitness)
    os_judge_top_k: int = Field(default=5, ge=0, le=30)
    os_judge_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    action_vocabulary_path: Path = Path("app/data/action_vocabulary.json")
    intent_goals_path: Path = Path("app/data/intent_goals.json")
    node_cap: int = Field(default=20, ge=4, le=50)
    # B: base pool sampled from the generator
    generation_pool_size: int = Field(default=20, ge=1, le=40)
    generation_temperatures: list[float] = Field(default_factory=lambda: [0.8], min_length=1)
    generation_examples: int = Field(default=2, ge=0, le=5)
    # Adaptive K: seeds = clusters of the pool (average linkage, Jaccard distance, cut at tau).
    # K is the number of clusters; k_min / k_max are optional guards (None = unclamped, as in the paper).
    seed_k_min: int = Field(default=1, ge=1, le=40)
    seed_k_max: Optional[int] = Field(default=None, ge=1, le=40)
    seed_cluster_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # --- CADE evolution (paper Section 5.3 / Table 3: 30 candidates, <= 20 generations, 2 elites, 20-node cap)
    population_size: int = Field(default=30, ge=4, le=200)
    max_generations: int = Field(default=20, ge=1, le=200)
    elite_count: int = Field(default=2, ge=0, le=10)
    crossover_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    # Fraction of add-node mutations that target a coverage gap in U(p) (the rest pick a random target)
    directed_mutation_probability: float = Field(default=0.7, ge=0.0, le=1.0)
    # Gap-directed additions ask the generator model which action closes the gap (falls back to a random closer)
    directed_mutation_llm: bool = True
    # Invalid mutants are retried this many times; then the parent is kept
    mutation_retries: int = Field(default=3, ge=1, le=10)
    # Generational replacement (paper): elites + offspring. False = (mu + lambda) survival (ablation)
    generational_replacement: bool = True
    # Stop at full coverage (IC = EA = 1) once the best fitness reaches this; 0 = full coverage alone (paper)
    fitness_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    # Stop when the best fitness gained less than plateau_epsilon over plateau_generations generations
    plateau_generations: int = Field(default=3, ge=1, le=50)
    plateau_epsilon: float = Field(default=0.01, ge=0.0, le=0.5)
    # The four places coverage steers the search (all on = full CADE; switches exist for ablation)
    coverage_init: bool = True
    coverage_variation: bool = True
    coverage_selection: bool = True
    coverage_termination: bool = True
    random_seed: Optional[int] = None

    # --- playbook library (paper Section 3) ------------------------------------------------
    # "sentence_transformers" (paper; optional package, model downloaded on first use) or "hashing"
    embedder: str = Field(default="sentence_transformers", pattern=r"^(sentence_transformers|hashing)$")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "faiss" (paper; optional faiss-cpu package) or "numpy"; falls back to numpy if faiss is missing
    vector_index: str = Field(default="faiss", pattern=r"^(faiss|numpy)$")
    library_seed_path: Path = Path("app/data/seed_playbooks.json")
    library_seed_on_startup: bool = True
    library_near_duplicate_cosine: float = Field(default=0.95, ge=0.0, le=1.0)
    library_near_duplicate_jaccard: float = Field(default=0.8, ge=0.0, le=1.0)
    library_deprioritize_after: int = Field(default=5, ge=1)
    library_deprioritize_below: float = Field(default=0.35, ge=0.0, le=1.0)
    library_deprioritized_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    # Seed/imported playbooks are curated, so they may be reused before they have a track record.
    library_allow_curated_reuse: bool = True

    # --- routing (paper Section 4 / Table 3: w = 0.7, k = 3, top N = 10) ---------------------
    routing_w: float = Field(default=0.7, ge=0.0, le=1.0)
    routing_k: float = Field(default=3.0, gt=0.0)
    routing_top_n: int = Field(default=10, ge=1, le=100)
    # A missing critical category counts as obtainable for Cov+ when its availability is >= tau
    routing_tau: float = Field(default=0.5, ge=0.0, le=1.0)
    # The paper caps evidence requests at one per incident
    max_evidence_rounds: int = Field(default=1, ge=0, le=5)
    # 1.0 = the paper's Eq. 2 exactly. > 1 sharpens the similarity distribution before the entropy
    # is taken (opt-in implementation option; see app/routing/utility.normalized_entropy).
    routing_similarity_sharpness: float = Field(default=1.0, ge=1.0, le=20.0)
    intent_taxonomy_path: Path = Path("app/data/intent_taxonomy.json")

    # --- learned routing (paper Section 4.2-4.3: disjoint LinUCB, d = 9, alpha = 1.0) ----------
    # "linucb": the bandit chooses once bandit_bootstrap_incidents rewards were observed (the rule
    # of Eq. 7 chooses before that); "rule": always the hand-designed rule (router ablation).
    routing_policy: str = Field(default="linucb", pattern=r"^(linucb|rule)$")
    bandit_alpha: float = Field(default=1.0, ge=0.0, le=10.0)
    bandit_ridge_lambda: float = Field(default=1.0, gt=0.0, le=100.0)
    bandit_bootstrap_incidents: int = Field(default=20, ge=0, le=10_000)
    # Library size enters the context as n / (n + scale)
    bandit_library_size_scale: float = Field(default=50.0, gt=0.0)
    # Simulated runs produce synthetic telemetry; by default they do not train the router.
    bandit_learn_from_simulated: bool = False
    # Reward (Eq. 9): r = 1_ret * [lc * IC_exec + ll * (1 - latency / L_max)] - lq * c_q(a)
    reward_lambda_coverage: float = Field(default=0.6, ge=0.0, le=1.0)
    reward_lambda_latency: float = Field(default=0.3, ge=0.0, le=1.0)
    reward_lambda_request: float = Field(default=0.1, ge=0.0, le=1.0)
    reward_request_cost: float = Field(default=1.0, ge=0.0)
    # L_max in seconds. None = the 90th-percentile latency of the bootstrap incidents (paper);
    # until those exist, reward_latency_fallback_seconds is used.
    reward_latency_max_seconds: Optional[float] = Field(default=None, gt=0.0)
    reward_latency_fallback_seconds: float = Field(default=600.0, gt=0.0)

    # Input limits protect the service and the LLM context window.
    max_description_chars: int = Field(default=50_000, ge=100)

    # --- security tools (Phase 2) -------------------------------------------------------
    # "simulate" (default): every tool call is answered by a deterministic simulator and labelled
    # SYNTHETIC. "live": calls go to the real tools configured below; nothing is simulated.
    execution_mode: ExecutionMode = ExecutionMode.SIMULATE
    # When a real tool fails or is not configured: "manual_task" or "error". Never simulation.
    tool_failure_policy: ToolFailurePolicy = ToolFailurePolicy.MANUAL_TASK
    tool_timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    tool_verify_tls: bool = True
    evidence_availability_path: Path = Path("app/data/evidence_availability.json")
    enrichment_max_calls: int = Field(default=20, ge=1, le=200)
    enrichment_max_values_per_type: int = Field(default=3, ge=1, le=20)

    virustotal_api_key: Optional[SecretStr] = None
    virustotal_base_url: str = "https://www.virustotal.com/api/v3"
    virustotal_requests_per_minute: float = Field(default=4, gt=0, le=1000)

    wazuh_api_url: Optional[str] = None
    wazuh_username: Optional[str] = None
    wazuh_password: Optional[SecretStr] = None
    wazuh_indexer_url: Optional[str] = None
    wazuh_indexer_username: Optional[str] = None
    wazuh_indexer_password: Optional[SecretStr] = None
    wazuh_isolation_command: Optional[str] = None  # Wazuh ships no isolation command; name yours here
    wazuh_block_ip_command: str = "!firewall-drop"

    elastic_url: Optional[str] = None
    elastic_api_key: Optional[SecretStr] = None
    elastic_index_pattern: str = "logs-*"
    kibana_url: Optional[str] = None

    osquery_binary: Optional[str] = None  # e.g. "osqueryi"; queries the CADE host only
    yara_binary: Optional[str] = None  # e.g. "yara"
    yara_rules_path: Optional[Path] = None
    yara_scan_roots: list[Path] = Field(default_factory=list)  # files outside these are never scanned
    sigma_rules_dir: Optional[Path] = None

    # --- controlled execution (safety gates; see app/execution/executor.py) ---------------------
    # An extra gate for destructive actions against REAL tools. It never replaces per-node analyst
    # approval or verification of the target; both are always required.
    allow_destructive_live: bool = False
    # The analyst who approved a destructive step may not also be the one executing it.
    require_separate_approver: bool = True
    max_destructive_targets: int = Field(default=1, ge=1, le=10)
    max_targets_per_node: int = Field(default=3, ge=1, le=20)
    # Simulated runs only produce synthetic findings; by default they do not train the library.
    library_learn_from_simulated: bool = False

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    def resolve_path(self, path: Path) -> Path:
        """Resolve a relative data path against the project root (not the current directory)."""
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
