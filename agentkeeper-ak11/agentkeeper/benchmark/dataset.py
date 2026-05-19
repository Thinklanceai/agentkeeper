"""Reference dataset for AgentKeeper benchmarks."""

from __future__ import annotations

from ..cso.types import CognitiveStateObject

CRITICAL_DATA: list[str] = [
    "project budget: 50000 EUR",
    "project deadline: March 1 2025",
    "client name: Acme Corporation",
    "primary contact: Jean Dupont jean@acme.com",
    "tech stack: Python FastAPI React PostgreSQL",
    "current sprint: Sprint 4 ends January 15",
    "blocker: API rate limiting on OpenAI tier 1",
    "decision: use Anthropic Claude for production",
    "team size: 3 engineers 1 designer",
    "deployment target: AWS eu-west-1",
    "database: PostgreSQL 15 with pgvector extension",
    "authentication: JWT tokens 24h expiry",
    "staging URL: https://staging.acme-project.com",
    "production URL: https://app.acme.com",
    "repository: github.com/acme/main-project private",
    "last release: v1.4.2 deployed December 20",
    "next milestone: beta launch February 1",
    "compliance requirement: GDPR data residency EU",
    "SLA target: 99.9 percent uptime",
    "monitoring: Datadog APM enabled",
]


NON_CRITICAL_TEMPLATES: list[str] = [
    "meeting notes day {i}: discussed sprint progress",
    "ticket AK-{i}: bug in user registration flow resolved",
    "dependency update {i}: bumped lodash to 4.17.{i}",
    "code review {i}: approved PR by engineer {i}",
    "slack message {i}: asked about deployment window",
    "log entry {i}: increased memory usage detected at noon",
    "research note {i}: evaluated competitor product feature",
    "todo {i}: refactor authentication middleware",
    "design feedback {i}: update button colors per brand guide",
    "infrastructure note {i}: reserved additional EC2 capacity",
]


def generate_test_facts(
    n_total: int = 100,
    n_critical: int = 20,
    agent_id: str = "benchmark-agent-001",
) -> CognitiveStateObject:
    """Generate a reproducible test CSO for benchmarking."""
    cso = CognitiveStateObject.create(agent_id=agent_id)

    for content in CRITICAL_DATA[:n_critical]:
        cso.add_fact(content, critical=True)

    for i in range(n_total - n_critical):
        template = NON_CRITICAL_TEMPLATES[i % len(NON_CRITICAL_TEMPLATES)]
        cso.add_fact(template.format(i=i + 1), critical=False)

    return cso
