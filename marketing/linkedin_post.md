# LinkedIn launch post

Single post, ~1200 characters. Attach the architecture SVG as image.

---

AgentKeeper 1.0 is out.

When I first shipped this project in February, I described it as "cross-model memory persistence." That was too narrow.

What it actually does — reconstruct an agent's full cognitive state across model switches, crashes, restarts, and constrained contexts — deserved a different name. v1.0 ships that name and the missing layers that make the claim true.

What's new:
• Memory hierarchy (working / episodic / semantic / archival)
• Persistent identity that survives every form of compression
• Semantic recall via local embeddings by default
• Cognitive compression: decay + consolidation + contradiction arbitration
• Cross-model translation: same state, rendered differently for Claude (XML), GPT-4 (sections), Gemini (narrative), and local models (minimal)
• Full async API
• Typed exceptions, structured logging, retry with backoff

267 tests. CI on Python 3.10/3.11/3.12. MIT license. Zero infrastructure required.

If you ship agents that need to survive across model changes, this is the layer you've been writing by hand.

→ https://github.com/Thinklanceai/agentkeeper
→ pip install agentkeeper

Built by ThinkLanceAI.

#AI #LLM #OpenSource #Python #AgentInfrastructure
