#!/usr/bin/env python3
"""Process 1: a coding agent does part of a task, checkpoints, then dies.

Killed with SIGKILL so nothing can be flushed from memory on the way out
— the only thing that survives is what was already written to the
checkpoint store on disk.
"""

from __future__ import annotations

import os
import signal
import time

import agentkeeper

GREEN = "\033[32m"
DIM = "\033[2m"
RESET = "\033[0m"

# Step delay. Default fast; set CRASH_DEMO_DELAY=1.0 for screen captures.
_DELAY = float(os.environ.get("CRASH_DEMO_DELAY", "0.2"))


def main() -> None:
    agent = agentkeeper.create(agent_id="dev-agent", provider="mock")
    agent.set_identity(
        name="Dev",
        role="autonomous coding agent",
        principles=[
            "never push to main without a green test run",
            "keep diffs small and reversible",
        ],
        constraints=["work only inside the repo sandbox"],
    )

    print(f"{DIM}  t+0.0s  agent booted, identity set{RESET}")
    agent.fact("repo: thinklanceai/auth-service", importance=0.8)
    agent.fact("goal: migrate JWT middleware to RS256", importance=0.95)
    time.sleep(_DELAY)
    print(f"{DIM}  t+0.2s  analysed codebase, opened auth/middleware.py{RESET}")
    agent.fact("decision: rotate signing keys via JWKS endpoint", importance=0.9)
    agent.event("started editing auth/middleware.py", when="2026-05-24")
    time.sleep(_DELAY)
    print(f"{DIM}  t+0.4s  wrote 1st half of the migration{RESET}")

    execution_state = {
        "current_file": "auth/middleware.py",
        "pending_task": "finish RS256 migration: swap verify() to JWKS",
        "completed_steps": [
            "added pyjwt[crypto] dependency",
            "wrote RS256 keypair loader",
        ],
        "open_todos": [
            "replace HS256 verify() call",
            "update 3 failing tests in test_auth.py",
            "remove legacy SECRET_KEY env var",
        ],
        "last_error": "test_auth.py::test_valid_token — AssertionError (still HS256)",
        "cursor_line": 142,
    }

    snap = agent.checkpoint(
        label="mid-migration before crash",
        execution_state=execution_state,
    )
    agent.save()

    print(
        f"{GREEN}  t+0.5s  ✔ checkpoint written: {snap.snapshot_id}{RESET}\n"
        f"{DIM}          hash={snap.content_hash[:16]}…  "
        f"facts={snap.meta.fact_count}  exec_state=yes{RESET}"
    )
    print(f"{DIM}  t+0.6s  … still working when the process is killed …{RESET}")
    time.sleep(_DELAY * 2)

    # Hard kill: no atexit, no flush, no graceful save. Whatever wasn't
    # already on disk is lost forever.
    os.kill(os.getpid(), signal.SIGKILL)


if __name__ == "__main__":
    main()
