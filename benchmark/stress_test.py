"""Honest stress test for AgentKeeper 1.1.1 (installed from PyPI).

Goal: find where it breaks, not prove it works. We push hard and
measure wall-clock time, memory, identity survival, and data integrity
at each phase. Nothing is mocked away — this is the real compression
pipeline, the real semantic recaller (mock embeddings for determinism
and speed, but the real indexing path), the real SQLite persistence.

Phases:
  1. Bulk insert 10k facts
  2. 500 compression cycles, assert identity survives every one
  3. Semantic recall latency at scale
  4. Save / load round-trip at 10k facts (integrity)
  5. Graph stress: 5k triples, traversal latency
"""

from __future__ import annotations

import os
import random
import time
import tracemalloc

os.environ["AGENTKEEPER_VECTOR_INDEX"] = "in_memory"
os.environ["AGENTKEEPER_EMBEDDING_PROVIDER"] = "mock"
os.environ["AGENTKEEPER_DB"] = "/tmp/ak_stress.db"

import agentkeeper

# Clean slate
if os.path.exists("/tmp/ak_stress.db"):
    os.remove("/tmp/ak_stress.db")

SUBJECTS = ["Acme", "Globex", "Initech", "Umbrella", "Stark", "Wayne", "Cyberdyne"]
PREDICATES = ["owns", "partners_with", "competes_with", "located_in", "funds"]
TOPICS = ["budget", "deadline", "client", "contract", "meeting", "risk", "hire"]


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main() -> None:
    results: dict[str, str] = {}

    # ----- setup -----
    agent = agentkeeper.create(agent_id="stress", provider="mock")
    agent.set_identity(
        name="Aria",
        role="stress-test subject",
        principles=["never lose identity", "survive compression"],
        constraints=["EU data residency only"],
    )
    assert agent.identity.name == "Aria"

    # ----- Phase 1: bulk insert 10k facts -----
    section("PHASE 1 — insert 10,000 facts")
    tracemalloc.start()
    t0 = time.perf_counter()
    for i in range(10_000):
        topic = random.choice(TOPICS)
        agent.fact(
            f"{topic} note {i}: value {random.randint(1, 100_000)}",
            importance=random.uniform(0.1, 0.9),
        )
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    insert_time = t1 - t0
    print(f"  inserted 10,000 facts in {insert_time:.2f}s "
          f"({10_000 / insert_time:.0f} facts/s)")
    print(f"  peak memory during insert: {peak / 1_048_576:.1f} MB")
    print(f"  agent.facts count: {len(agent.facts)}")
    results["insert_10k"] = f"{insert_time:.2f}s, {peak / 1_048_576:.1f}MB peak"
    assert len(agent.facts) == 10_000, "fact count mismatch!"

    # ----- Phase 2: 500 compression cycles -----
    section("PHASE 2 — 500 compression cycles, identity must survive each")
    t0 = time.perf_counter()
    identity_failures = 0
    first_compress_time = None
    for cycle in range(500):
        c0 = time.perf_counter()
        agent.compress()
        c1 = time.perf_counter()
        if first_compress_time is None:
            first_compress_time = c1 - c0
        # Identity assertions every cycle
        if agent.identity.name != "Aria":
            identity_failures += 1
        if "never lose identity" not in agent.identity.principles:
            identity_failures += 1
        # Re-inject some facts to keep the pipeline working
        if cycle % 50 == 0:
            for _ in range(20):
                agent.fact(f"refill {cycle}-{_}", importance=random.uniform(0.2, 0.7))
            print(f"  cycle {cycle:3d}: {len(agent.facts):5d} facts, "
                  f"identity={'OK' if agent.identity.name == 'Aria' else 'FAIL'}")
    t1 = time.perf_counter()
    compress_total = t1 - t0
    print(f"\n  500 compressions in {compress_total:.2f}s "
          f"({compress_total / 500 * 1000:.1f}ms/cycle avg)")
    print(f"  first compression: {first_compress_time * 1000:.1f}ms")
    print(f"  identity failures across 500 cycles: {identity_failures}")
    results["compress_500"] = (
        f"{compress_total:.2f}s, {identity_failures} identity failures"
    )
    assert identity_failures == 0, "IDENTITY DRIFT DETECTED!"

    # ----- Phase 3: semantic recall at scale -----
    section("PHASE 3 — semantic recall latency")
    # Refill to a known size first
    for i in range(2000):
        agent.fact(f"budget allocation {i}: {random.randint(1, 99)}k EUR",
                   importance=random.uniform(0.3, 0.8))
    recall_times = []
    for _ in range(20):
        r0 = time.perf_counter()
        hits = agent.recall("how much budget money allocated", top_k=5)
        r1 = time.perf_counter()
        recall_times.append(r1 - r0)
    avg_recall = sum(recall_times) / len(recall_times)
    print(f"  facts in store: {len(agent.facts)}")
    print(f"  avg recall latency (top_k=5): {avg_recall * 1000:.1f}ms")
    print(f"  max recall latency: {max(recall_times) * 1000:.1f}ms")
    print(f"  sample hit: {hits[0][1]:.3f}  {hits[0][0].content[:50]}" if hits else "  no hits")
    results["recall"] = f"{avg_recall * 1000:.1f}ms avg"

    # ----- Phase 4: save / load round trip -----
    section("PHASE 4 — save / load integrity")
    facts_before = len(agent.facts)
    identity_before = agent.identity.name
    s0 = time.perf_counter()
    agent.save()
    s1 = time.perf_counter()
    db_size = os.path.getsize("/tmp/ak_stress.db") / 1_048_576
    print(f"  save time: {s1 - s0:.2f}s")
    print(f"  db file size: {db_size:.1f} MB")

    l0 = time.perf_counter()
    reloaded = agentkeeper.load("stress", provider="mock")
    l1 = time.perf_counter()
    print(f"  load time: {l1 - l0:.2f}s")
    print(f"  facts before: {facts_before}, after reload: {len(reloaded.facts)}")
    print(f"  identity before: {identity_before}, after: {reloaded.identity.name}")
    integrity_ok = (
        len(reloaded.facts) == facts_before
        and reloaded.identity.name == identity_before
    )
    print(f"  integrity: {'OK' if integrity_ok else 'BROKEN'}")
    results["save_load"] = (
        f"save {s1 - s0:.2f}s, load {l1 - l0:.2f}s, "
        f"{'intact' if integrity_ok else 'BROKEN'}"
    )
    assert integrity_ok, "DATA INTEGRITY BROKEN ON RELOAD!"

    # ----- Phase 5: graph stress -----
    section("PHASE 5 — graph: 5,000 triples + traversal")
    g0 = time.perf_counter()
    for i in range(5000):
        s = random.choice(SUBJECTS)
        p = random.choice(PREDICATES)
        o = random.choice(SUBJECTS)
        if s != o:
            agent.link(s, p, o, confidence=random.uniform(0.5, 1.0))
    g1 = time.perf_counter()
    print(f"  inserted {len(agent.triples)} triples in {g1 - g0:.2f}s")
    f0 = time.perf_counter()
    related = agent.find_related("Acme", max_hops=2, direction="both")
    f1 = time.perf_counter()
    print(f"  find_related('Acme', 2 hops): {len(related)} entities "
          f"in {(f1 - f0) * 1000:.1f}ms")
    results["graph_5k"] = f"{g1 - g0:.2f}s insert, {(f1 - f0) * 1000:.1f}ms traversal"

    # ----- summary -----
    section("SUMMARY")
    for k, v in results.items():
        print(f"  {k:16s}: {v}")
    print("\n  ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
