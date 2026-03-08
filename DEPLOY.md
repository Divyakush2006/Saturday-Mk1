# Saturday MK1 — Deployment Guide (Free Stack)

Deploy Saturday MK1 to production with a **100% free** stack.

| Layer | Service | Free Tier |
|---|---|---|
| Hosting | [Vercel](https://vercel.com) | 100GB bandwidth, 10s function timeout |
| Database | [Neon](https://neon.tech) | 0.5GB PostgreSQL, auto-suspend |
| LLM | [NVIDIA API](https://build.nvidia.com) | Existing API key |

---

## Step 1 — Create Neon Database

1. Go to [neon.tech](https://neon.tech) → Sign up (free)
2. Create a new project → Name it `saturday-mk1`
3. Copy the **connection string** — it looks like:
   ```
   postgresql://username:password@ep-xxxx.us-east-2.aws.neon.tech/neondb?sslmode=require
   ```
4. Keep this safe — you'll need it in Step 3.

---

## Step 2 — Push to GitHub

If you haven't already:

```bash
cd /path/to/Saturday
git init
git add .
git commit -m "Saturday MK1 — ready for deployment"
git remote add origin https://github.com/YOUR_USERNAME/saturday-mk1.git
git push -u origin main
```

> **Important**: `.env` is in `.gitignore` — your API keys will NOT be pushed.

---

## Step 3 — Deploy to Vercel

### Option A: Vercel Dashboard (easiest)

1. Go to [vercel.com](https://vercel.com) → Sign up / Log in
2. Click **"Add New Project"** → Import your GitHub repo
3. Vercel auto-detects `vercel.json` — no build settings needed
4. **Set Environment Variables** (critical):

   | Variable | Value |
   |---|---|
   | `DATABASE_URL` | Your Neon connection string from Step 1 |
   | `SATURDAY_API_KEY` | Your NVIDIA API key |
   | `SATURDAY_BASE_URL` | `https://integrate.api.nvidia.com/v1` |
   | `SATURDAY_MODEL` | `qwen/qwen3.5-122b-a10b` |
   | `SATURDAY_PROVIDER` | `openai` |
   | `SATURDAY_SERVER_KEY` | Any secret string for API auth (e.g. `sk-saturday-prod-xxxx`) |
   | `SATURDAY_JWT_SECRET` | A long random string (run `python -c "import secrets; print(secrets.token_hex(64))"`) |

5. Click **Deploy** → Wait ~60 seconds

### Option B: Vercel CLI

```bash
npm i -g vercel
cd /path/to/Saturday
vercel --prod
```

Then set env vars via:
```bash
vercel env add DATABASE_URL
vercel env add SATURDAY_API_KEY
# ... repeat for each variable
vercel --prod  # redeploy with env vars
```

---

## Step 4 — Verify Deployment

After deployment, test these URLs (replace `your-app.vercel.app` with your actual URL):

1. **Frontend**: `https://your-app.vercel.app/` → Should show Saturday chat UI
2. **Health**: `https://your-app.vercel.app/api/v1/health` → Should return `{"status": "healthy", ...}`
3. **API Docs**: `https://your-app.vercel.app/docs` → Interactive Swagger UI
4. **Sign up**: Use the auth modal in the chat UI to create an account
5. **Chat**: Send a message in "Fast" mode — should respond in 3-8 seconds

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `502 Bad Gateway` | Check Vercel function logs (Dashboard → Functions tab). Usually a missing env var. |
| `FUNCTION_INVOCATION_TIMEOUT` | The LLM call took >10s. Try simpler prompts or use Fast mode. |
| `ModuleNotFoundError` | A dependency is missing from `requirements.txt`. |
| Database connection fails | Verify `DATABASE_URL` starts with `postgresql://` and has `?sslmode=require`. |
| Auth not working | Ensure `SATURDAY_JWT_SECRET` is set in Vercel env vars. |
| Chat returns empty | Check `SATURDAY_API_KEY` and `SATURDAY_BASE_URL` are correct. |

---

## Architecture on Vercel

```
Browser → Vercel CDN → frontend/index.html (static)
                     → api/index.py (Python serverless function)
                        ↓
                     saturday_server.py (FastAPI)
                        ↓
                     brain/saturday_core.py → engines/*
                        ↓
                     NVIDIA API (Qwen 3.5) + Neon PostgreSQL
```

All API routes (`/api/v1/*`) are handled by a single Python serverless function.
Static files (`/frontend/*`, `/`) are served from Vercel's CDN edge network.
