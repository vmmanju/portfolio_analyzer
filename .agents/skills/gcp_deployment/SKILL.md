---
name: GCP Cloud Run Deployment
description: Comprehensive steps and commands to deploy a containerized Python/Streamlit application to Google Cloud Run, configure PostgreSQL connections (Neon), map a custom domain, and manage Streamlit secrets securely via GCP Secret Manager.
---

# Deploying to Google Cloud Run

This skill encapsulates the exact workflow required to deploy applications securely to Google Cloud Run based on our experiences deploying the AI Stock Engine.

## Prerequisites
- A `Dockerfile` defining the container image.
- A `requirements.txt` containing all dependencies. Be extremely careful that Streamlit version is high enough (>=1.42.0) if using `st.login()` OIDC features.
- A `.dockerignore` file excluding local environments, cache, and sensitive files like `.env` or `secrets.toml`.

## Deployment Workflow

### 1. Initial Cloud Run Deployment
To package the app and deploy it to Google Cloud Run:
gcloud run deploy [SERVICE-NAME] \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=web,DATABASE_URL=postgresql://[DB_USER]:[DB_PASSWORD]@[DB_HOST_URL]?sslmode=require" \
  --memory 8Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 1 \
  --port 8080 \
  --timeout 3600 \
  --project=[GCP_PROJECT_ID]
```

#### Important Flags from Experience:
- **`--memory 8Gi` & `--max-instances 1`**: If your app does heavy data science (like matrix multiplication), 1Gi will OOM. However, GCP strictly enforces Region-level Project quotas (e.g., `us-central1` allows max 40 GiB total). During a deployment rollout, Cloud Run spins up the *new* revision simultaneously with the *old* revision. Deploying 16GiB with `--max-instances 2` requests 64 GiB during the update and will hard-fail with `Quota violated: MemAllocPerProjectRegion`. Therefore, setting 8Gi with max-instances 1 allows the rollout to safely peak at 16GiB bounds.
*Note: We deploy to `us-central1` because custom domain mapping is fully supported there, unlike `asia-south1` which requires a load balancer.*

### 2. Custom Domain Mapping
Once deployed, to map a custom domain (e.g., `jooju.in` or `www.jooju.in`) natively to Cloud Run:
1. Go to the Cloud Run console -> **Manage Custom Domains**.
2. Click **Add Mapping** and map your service to your verified domain.
3. Add the provided DNS records (usually `A` and `AAAA` for the root, and `CNAME ghs.googlehosted.com.` for `www`) at your domain registrar.
4. Wait 30-60 mins for SSL certificates to provision.

### 3. Secure Secret Management (GCP Secret Manager)
If your app utilizes `st.secrets` (via `secrets.toml`), do NOT bake them into the Docker image. Cloud Run requires explicitly mounting the secrets.

1. Enable Secret Manager API:
```bash
gcloud services enable secretmanager.googleapis.com --project=[GCP_PROJECT_ID]
```

2. Create a new secret from a local file:
```bash
gcloud secrets create streamlit-secrets --data-file=.streamlit/secrets.toml --project=[GCP_PROJECT_ID]
```

3. If you ever update the file locally, push a new version:
```bash
gcloud secrets versions add streamlit-secrets --data-file=.streamlit/secrets.toml --project=[GCP_PROJECT_ID]
```

4. **Critical**: Redeploy Cloud Run while natively mounting the secret as a volume to the path Streamlit expects:
gcloud run deploy [SERVICE-NAME] \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars "ENVIRONMENT=web,DATABASE_URL=postgresql://[DB_USER]:[DB_PASSWORD]@[DB_HOST_URL]?sslmode=require" \
  --set-secrets="/app/.streamlit/secrets.toml=streamlit-secrets:latest" \
  --memory 8Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 1 \
  --port 8080 \
  --timeout 3600 \
  --project=[GCP_PROJECT_ID]
```

### 4. Source Upload Crash (UnicodeDecodeError)
The `gcloud` CLI (especially on Windows) sometimes crashes aggressively during source upload with a Python `UnicodeDecodeError: invalid start byte 0xff` when trying to parse random binary files (like `.docx`, `.xlsx`, `.pyc`).

To bypass this safely:
1. Do not use complex wildcard overrides like `!**/*.py` in `.gcloudignore`.
2. Disable gcloudignore entirely so it falls back to standard Docker behavior:
```bash
gcloud config set gcloudignore/enabled false
```
3. Instead, rely natively on `.dockerignore` to filter out builds and unnecessary cached states (`__pycache__`, `.pytest_cache`, `.git/`, `.agents/`).

### 5. Multiprocessing & PostgreSQL Disconnections on Cloud Run
When using Python `concurrent.futures.ProcessPoolExecutor` on Google Cloud Run (Linux):
- Linux uses `fork` to spool child processes.
- Inherited SQLAlchemy objects (specifically `engine` connection pools mapping to db sockets like Neon PostgreSQL) are physically duplicated. 
- If multiple child processes send data through the same duplicated network socket simultaneously, the database router gets overwhelmed and forcibly aborts: `server closed the connection unexpectedly / psycopg2.OperationalError`.

**Fix:**
Always purge inherited pools in the spawned worker process before making DB calls:
```python
def parallel_worker_task(args...):
    from app.database import engine
    engine.dispose(close=False) # Safely drop inherited socket, forcing new connection
    # ... handle DB query ...
```

### 4. Updating OAuth Credentials for Google Login
If the deployment uses Google OIDC (`st.login`), ensure your OAuth 2.0 Client ID has the exact domains allowed.
If moving to a custom domain:
- The backend `secrets.toml` `redirect_uri` MUST be exactly `https://[YOUR-DOMAIN]/oauth2callback` (e.g., `https://www.jooju.in/oauth2callback`).
- The **Authorized Redirect URIs** in Google Cloud Console MUST include that exact URI.
- The **Authorized JavaScript origins** in Google Cloud MUST include the root domain schemas (`https://jooju.in` and `https://www.jooju.in`).

## Related Resources — Lessons RAG

Before deploying or troubleshooting GCP issues, query the project's RAG knowledge base:

```bash
# From portfolio_analyzer root:
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "your question here"

# Useful examples for this skill:
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "memory quota cloud run deployment"
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "multiprocessing postgres connection child process"
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "gcloud windows unicode error upload"
.venv\Scripts\python .agents/skills/lessons_rag/scripts/rag_engine.py "oauth redirect uri custom domain"
```

See `.agents/skills/lessons_rag/SKILL.md` for full usage instructions.
