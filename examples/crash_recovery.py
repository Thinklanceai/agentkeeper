#!/usr/bin/env python3
"""Crash recovery demo — a coding agent that survives a process death.

Runs entirely on provider="mock". No API key, no network.

It launches TWO separate Python processes:

  1. `_work.py`  — an agent does part of a coding task, checkpoints its
                   cognitive + execution state, then the process is KILLED
                   mid-task (SIGKILL). Nothing is kept in memory.
  2. `_resume.py`— a brand-new process starts, finds the last checkpoint
                   on disk, restores it, and continues exactly where the
                   dead process stopped.

The point: continuity comes from the on-disk checkpoint, not from shared
memory. Process 2 never spoke to process 1. Honest framing — AgentKeeper
restores *state*, not the model's next decision.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CYAN = "\033[36m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"


def banner(text: str, colour: str = CYAN) -> None:
    line = "─" * 64
    print(f"\n{colour}{line}{RESET}")
    print(f"{colour}{BOLD}  {text}{RESET}")
    print(f"{colour}{line}{RESET}")


def main() -> None:
    workdir = tempfile.mkdtemp(prefix="agentkeeper_crashdemo_")
    env = dict(os.environ)
    env["AGENTKEEPER_CHECKPOINT_DIR"] = str(Path(workdir) / "checkpoints")
    env["AGENTKEEPER_DB"] = str(Path(workdir) / "agent.db")
    env["PYTHONUNBUFFERED"] = "1"

    banner("PROCESS 1 — agent starts a coding task, then is KILLED", CYAN)
    p1 = subprocess.run(
        [sys.executable, str(HERE / "_crash_work.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    print(p1.stdout, end="")
    if p1.stderr.strip():
        print(f"{DIM}{p1.stderr.strip()}{RESET}")
    # The work process kills itself with SIGKILL (-9) after checkpointing.
    # On POSIX, returncode is -9. We treat anything else as a setup error.
    if p1.returncode not in (-9, 137):
        print(f"{RED}Setup error: work process exited {p1.returncode}{RESET}")
        sys.exit(1)
    print(f"\n{RED}{BOLD}  ☠  Process 1 was killed (SIGKILL). Its memory is gone.{RESET}")

    banner("PROCESS 2 — a NEW process resumes from the checkpoint", GREEN)
    p2 = subprocess.run(
        [sys.executable, str(HERE / "_crash_resume.py")],
        env=env,
        capture_output=True,
        text=True,
    )
    print(p2.stdout, end="")
    if p2.stderr.strip():
        print(f"{DIM}{p2.stderr.strip()}{RESET}")
    if p2.returncode != 0:
        print(f"{RED}Resume process failed ({p2.returncode}).{RESET}")
        sys.exit(1)

    banner("WHAT JUST HAPPENED", DIM)
    print(
        f"{DIM}  Process 2 shared no memory with Process 1. Continuity came\n"
        f"  entirely from the on-disk checkpoint: identity, facts, and the\n"
        f"  opaque execution_state (current file, task, todos).\n\n"
        f"  AgentKeeper restored the agent's *state*. What it does next is\n"
        f"  up to the model — that part was never claimed to be deterministic.{RESET}\n"
    )


if __name__ == "__main__":
    main()
