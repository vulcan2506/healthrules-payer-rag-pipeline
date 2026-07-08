---
title: HealthRules Payer Backend
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# HealthRules Payer — Backend API

FastAPI backend for the HealthRules Payer documentation-intelligence pipeline.
Serves `/api/health`, `/api/chat`, `/api/process`, and Knowledge Explorer
endpoints consumed by the Next.js frontend (deployed separately on Vercel —
see the main repo's `DEPLOY.md`).

## Required Space secrets (Settings → Variables and secrets)

- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `GROQ_API_KEY`
- `ALLOWED_ORIGINS` — comma-separated, e.g. `https://your-app.vercel.app,http://localhost:3000`

The pre-built corpus (`chroma_db/`, `index/`) ships baked into this Space, so
chat/Knowledge Explorer/Visualize work immediately without running Process
first.
