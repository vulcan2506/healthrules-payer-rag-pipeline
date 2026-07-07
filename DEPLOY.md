# Deploying frontend (Vercel) + backend (Render)

Split deploy: the frontend runs on Vercel (what it's built for); the backend
runs on Render as an always-on Docker service (Vercel's serverless functions
can't fit torch/chromadb/docling or hold this backend's in-memory state —
see `Dockerfile.backend`'s header comment for why).

These steps need your own Render + Vercel accounts (I can't create accounts
or log in on your behalf) — everything else (Dockerfile, render.yaml, env
var wiring, CORS) is already prepared in this repo.

## 1. Backend → Render

1. Push this repo to GitHub if you haven't already (`git push origin main`).
2. Render dashboard → **New** → **Blueprint** → connect the
   `healthrules-payer-rag-pipeline` repo. Render will read `render.yaml` at
   the repo root and provision:
   - `healthrules-payer-backend` — the Docker web service (`Dockerfile.backend`)
   - `healthrules-payer-cache` — a free managed Redis instance
3. Before/after the first deploy, set these in the web service's
   **Environment** tab (Render never reads `Stage 1/.env` — it's gitignored):
   - `ANTHROPIC_API_KEY` — your key
   - `OPENROUTER_API_KEY` — your key
   - `GROQ_API_KEY` — your key
   - `ANTHROPIC_MODEL` — already defaulted to `claude-sonnet-4-6` in `render.yaml`, override if needed
   - `REDIS_URL` — auto-wired from the Redis service by `render.yaml`, no action needed
4. First build will take a while (torch + docling + sentence-transformers).
   Once live, confirm health: `curl https://<your-service>.onrender.com/api/health`
   should return `{"status": "ok"}`.
5. Test chat end-to-end:
   ```
   curl -X POST https://<your-service>.onrender.com/api/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "How does claim adjudication work?", "mode": "concise", "session_id": "test-1"}'
   ```

**Known limitation, not new here:** `/api/process` (PDF upload + reprocess)
always reprocesses the *entire* corpus, not just the new file — same
behavior as local dev, just slower on Render's shared CPU. The pre-built
corpus (`chroma_db/`, `index/`) already ships baked into the image via
git-LFS, so chat/knowledge-browsing/visualize all work immediately without
running Process first.

## 2. Frontend → Vercel

1. Vercel dashboard → **Add New** → **Project** → import the same GitHub repo.
2. Set **Root Directory** to `frontend/` (the Next.js app doesn't live at
   the repo root).
3. Framework preset should auto-detect Next.js. Add one env var:
   - `NEXT_PUBLIC_API_BASE_URL` = `https://<your-service>.onrender.com`
     (the Render backend URL from step 1)
4. Deploy. Vercel gives you a `https://<your-app>.vercel.app` URL.

## 3. Close the loop — CORS

Go back to Render → the backend service's **Environment** tab → update:
```
ALLOWED_ORIGINS = https://<your-app>.vercel.app,http://localhost:3000
```
Save (triggers a redeploy). Without this, the deployed frontend's requests
will be blocked by CORS — `api_server.py` only allows origins listed here.

## 4. Verify end-to-end

Open the Vercel URL, send a chat message, confirm:
- A real answer comes back (not a CORS/network error in the browser console)
- Knowledge Explorer lists real files
- A follow-up like "explain that in more detail" gets a grounded answer, not
  a hallucinated unrelated one (the conversation-memory fix from this session)

## Notes

- `render.yaml` currently targets the **free** web-service plan (512MB RAM)
  to start with zero cost. This is genuinely tight for torch +
  sentence-transformers + docling loaded together — if the first real chat
  request OOM-crashes (check the service's Logs tab for a killed/restarted
  process), bump `plan: free` → `plan: standard` (~$25/mo, 2GB RAM) in
  `render.yaml` and push again. Free tier also sleeps after inactivity,
  causing a slow first request after idle.
- Local llama.cpp (fallback tier 4) obviously can't run on Render — the
  chain still degrades gracefully through Claude → OpenRouter → Groq;
  tier 4 simply won't come up if all three fail, surfaced as a clear error
  rather than a crash.
