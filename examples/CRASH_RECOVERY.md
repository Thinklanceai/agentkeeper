# Crash recovery demo

A coding agent survives a real process death and resumes from disk.

```bash
python examples/crash_recovery.py
```

No API key, no network — runs on `provider="mock"`.

## What it shows

The orchestrator launches **two genuinely separate processes**:

1. **`_crash_work.py`** — an agent starts a JWT→RS256 migration, writes a
   checkpoint (cognitive state + an opaque `execution_state`: current
   file, pending task, open todos, last error), then kills itself with
   `SIGKILL`. No graceful shutdown, no flush — whatever wasn't already on
   disk is gone.

2. **`_crash_resume.py`** — a brand-new process, with no shared memory,
   finds the latest checkpoint, verifies its hash, restores it, and
   prints exactly where work stopped.

Continuity comes entirely from the on-disk checkpoint, not from shared
memory — which is why two separate processes prove the point a single
script can't.

## Honest framing

AgentKeeper restores the agent's **state** — identity, facts, graph, and
the opaque execution payload you attached. It does **not** restore or
guarantee the model's next decision; reconstruction is deterministic,
behaviour is not. The `execution_state` is stored and returned verbatim:
AgentKeeper never interprets or runs it.
