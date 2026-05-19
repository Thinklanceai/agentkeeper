# Email template — warm outreach

Use this for warm contacts (Travis Kirschbaum at Beatport, Alon Shulman
/ World Famous Group, AKQA contacts, agency leads).

Tone: technical, no hype, lead with what they care about. Customise the
opening line and the closing CTA per recipient.

Subject lines to A/B test:

- "Shipping v1.0 of the cognitive layer I mentioned"
- "Cognitive continuity infrastructure — v1.0 just shipped"
- "From cross-model memory to cognitive continuity (v1.0)"

---

Hi [first name],

Following up briefly on [the conversation we had / the demo I sent].
I just shipped v1.0 of AgentKeeper — the cognitive continuity layer
I was building when we last spoke.

The short version: it's the infrastructure that sits between an AI
agent and the LLM provider, owning the question of *what cognitive
state to reconstruct for each call.* It handles identity persistence,
memory hierarchy, semantic recall, cognitive compression, and
cross-model translation (same agent state, rendered differently for
Claude / GPT / Gemini / local models).

Three things I think might matter for [their company / their stack]:

1. **Vendor-agnostic by design.** No lock-in to OpenAI or Anthropic.
   Switch providers without rewriting the agent.
2. **Identity survives compression.** Principles and hard constraints
   are exempt from every form of memory compression. Critical for
   regulated workflows.
3. **Async-native.** Full asyncio API, fits cleanly into FastAPI,
   LangChain async, or any agent runtime.

Links:

- README: https://github.com/Thinklanceai/agentkeeper
- PyPI: `pip install agentkeeper`
- Built by ThinkLanceAI: https://thinklanceai.com

It's MIT-licensed open source. If you'd like a 20-minute walkthrough
of how it fits a specific workflow on your side, happy to set one up.

Best,
Tom
ThinkLanceAI
hello@thinklanceai.com
