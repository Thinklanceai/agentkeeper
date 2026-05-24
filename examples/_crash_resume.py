#!/usr/bin/env python3
"""Process 2: a fresh process resumes the dead agent from its checkpoint.

This process never shared memory with process 1. It loads the agent's
last persisted state, finds the most recent checkpoint, restores it, and
prints exactly where work stopped.
"""

from __future__ import annotations

import agentkeeper

GREEN = "\033[32m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
RESET = "\033[0m"


def main() -> None:
    print(f"{DIM}  fresh process — no memory of the previous run{RESET}")

    checkpoints = agentkeeper.list_checkpoints("dev-agent")
    if not checkpoints:
        raise SystemExit("No checkpoint found — nothing to resume.")

    latest = checkpoints[-1]
    print(
        f"{DIM}  found {len(checkpoints)} checkpoint(s); "
        f"latest = {latest.snapshot_id}{RESET}"
    )

    agent = agentkeeper.load("dev-agent", provider="mock")
    agent.restore(latest.snapshot_id)

    snap = agentkeeper.load_checkpoint("dev-agent", latest.snapshot_id)
    if not snap.verify():
        raise SystemExit("Checkpoint integrity check FAILED.")
    print(f"{GREEN}  ✔ checkpoint integrity verified (hash matches){RESET}")

    ident = agent._cso.identity
    print(f"\n{CYAN}{BOLD}  Identity restored:{RESET}")
    print(f"    name        : {ident.name}")
    print(f"    role        : {ident.role}")
    print(f"    principles  : {len(ident.principles)} (protected, survived)")

    print(f"\n{CYAN}{BOLD}  Cognitive memory restored:{RESET}")
    for fact in agent._cso.memory_facts:
        print(f"    • {fact.content}")

    exec_state = snap.execution_state or {}
    print(f"\n{GREEN}{BOLD}  ▶ Resuming task: "
          f"{exec_state.get('pending_task', '(none)')}{RESET}")
    print(f"{DIM}    current file : {exec_state.get('current_file')}"
          f"  (line {exec_state.get('cursor_line')}){RESET}")
    print(f"{DIM}    last error   : {exec_state.get('last_error')}{RESET}")
    todos = exec_state.get("open_todos", [])
    print(f"{DIM}    open todos   :{RESET}")
    for todo in todos:
        print(f"{DIM}      ☐ {todo}{RESET}")

    print(f"\n{GREEN}{BOLD}  The 9-step migration picked up exactly where it died.{RESET}")


if __name__ == "__main__":
    main()
