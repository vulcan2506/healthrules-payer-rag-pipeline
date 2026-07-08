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

## 1b. Backend → Hugging Face Spaces (alternative to Render)

Same backend, different host — use this instead of step 1 if you'd rather
avoid Render's 512MB free-tier RAM ceiling. HF Spaces' free CPU Basic tier
gives 16GB RAM / 2 vCPU (vs Render's 512MB), at the cost of the Space
sleeping after inactivity on the free tier (same tradeoff Render's free plan
already has). `Dockerfile.hf` bundles a `redis-server` sidecar since HF
Spaces has no managed Redis add-on.

1. Hugging Face → **New Space** → SDK: **Docker** → note the `<username>/<space-name>`.
2. Space → **Settings** → **Access Tokens** (your account, not the Space) →
   create a token with **write** access if you don't already have one.
3. Locally:
   ```bash
   pip install -U huggingface_hub
   hf auth login   # paste the token
   python deploy_hf_space.py <username>/<space-name>
   ```
   This uploads exactly the files `git ls-files "Stage 1" retrieval_layer`
   returns (73 files, ~139MB — the same set already safe for GitHub: no
   secrets, no `venv/`, no `Stage 1/data/`), plus `Dockerfile.hf` renamed to
   `Dockerfile` and `README_hf.md` renamed to `README.md` (HF Spaces
   requires those exact filenames at the repo root; your GitHub-facing
   `README.md`/`Dockerfile.backend` are untouched).
4. Space → **Settings** → **Variables and secrets** → add `ANTHROPIC_API_KEY`,
   `OPENROUTER_API_KEY`, `GROQ_API_KEY`. Leave `ALLOWED_ORIGINS` for step 3 below.
5. Wait for the build (torch + docling + sentence-transformers — a few
   minutes). Confirm: `curl https://<username>-<space-name>.hf.space/api/health`
   → `{"status": "ok"}`.
6. Use `https://<username>-<space-name>.hf.space` as the backend URL in
   step 2 (Vercel) and step 3 (CORS) below, same as the Render URL.

Re-run `python deploy_hf_space.py <username>/<space-name>` any time backend
code changes — one command, one new commit.

**Known limitation:** this only closes the RAM gap, not the timeout risk —
`/api/chat` still returns one blocking (non-streamed) response, and a cold
first request that also cascades through the Claude→OpenRouter→Groq fallback
chain can run long behind HF's proxy. Streaming the response is the real fix
for that and hasn't been done yet.

## 2. Frontend → Vercel

1. Vercel dashboard → **Add New** → **Project** → import the same GitHub repo.
2. Set **Root Directory** to `frontend/` (the Next.js app doesn't live at
   the repo root).
3. Framework preset should auto-detect Next.js. Add one env var:
   - `NEXT_PUBLIC_API_BASE_URL` = your backend URL from step 1 (Render,
     `https://<your-service>.onrender.com`) or step 1b (HF Spaces,
     `https://<username>-<space-name>.hf.space`)
4. Deploy. Vercel gives you a `https://<your-app>.vercel.app` URL.

## 3. Close the loop — CORS

Go back to whichever backend host you used (Render's **Environment** tab, or
HF Space's **Variables and secrets**) → update:
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
