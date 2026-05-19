# AgentKeeper v1.0 — Marketing Kit

This directory contains the materials prepared for the v1.0 launch.
Nothing here is published automatically. Tom reviews each piece before
sending.

## Contents

- `release_announcement.md` — long-form post for blog / LinkedIn / Substack.
- `launch_thread_x.md` — short-form thread for X (Twitter).
- `linkedin_post.md` — single LinkedIn post (1300-character format).
- `supporter_dms.md` — 7 personalised DMs for the March 2026 supporters
  (Shruti, Chidanand, Kayvon, Leonard, Martin, Mustapha, Bespoke AI).
- `release_notes_github.md` — copy/paste for the GitHub Release notes.
- `email_template.md` — outreach email for warm contacts
  (Travis Kirschbaum, Alon Shulman, AKQA, etc.).

## Release checklist

1. [ ] Merge all `sprint-ak*` branches into `main` via a single PR.
2. [ ] Run `pytest -q` and `ruff check agentkeeper tests` one final time.
3. [ ] Tag `v1.0.0` and push tags.
4. [ ] Build and upload to PyPI: `python -m build && twine upload dist/*`.
5. [ ] Create GitHub Release using `release_notes_github.md`.
6. [ ] Reply to the open GitHub issue with "Fixed in v1.0".
7. [ ] Set up GitHub notifications across all 4 OSS repos (Settings →
   Notifications → Watch all repositories owned by Thinklanceai).
8. [ ] Send 7 personalised DMs from `supporter_dms.md`.
9. [ ] Post `linkedin_post.md` on LinkedIn.
10. [ ] Post `launch_thread_x.md` on X.
11. [ ] Publish `release_announcement.md` on the ThinkLanceAI blog.
12. [ ] Update thinklanceai.com "What's shipping" section.

## Tone

- Confident but technical. No hype words ("revolutionary", "game-changing").
- Concrete claims, not abstract ones.
- Always credit the March 2026 supporters who saw it early.
- Acknowledge the repositioning openly when relevant.
