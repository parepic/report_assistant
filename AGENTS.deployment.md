# 🚀 Deployment Plan (GCP Single-VM + Docker Compose)

This deployment plan targets the **simplest path to “online + publicly accessible”** on Google Cloud Platform (GCP):

- One **Compute Engine VM** runs the entire stack via **Docker Compose**
- **Nginx** handles public HTTP(S) traffic and reverse-proxies to the UI/API
- **Postgres** and **Qdrant** run as containers with **persistent Docker volumes**

This is intentionally a “v1” deployment. Once the app is stable, you can split it into a more cloud-native 4-service architecture (see the last section).

---

## Outcome (what “done” means)

- Anyone can access the app from the public internet via a domain like `https://yourdomain.com`
- The Streamlit UI is the primary public entrypoint (search engines can crawl it if it’s public)
- The FastAPI backend is reachable internally by the UI (and optionally publicly under `/api`)
- Postgres and Qdrant have durable storage on the VM

---

## Tech stack (this time)

### Cloud (GCP)
- **Compute Engine VM**: the single server that runs all containers
- **VPC Firewall rules**: allow inbound `80/tcp` and `443/tcp` (and optionally `22/tcp` for SSH)
- **(Optional) Cloud DNS**: manage domain records inside GCP

### Container/runtime
- **Docker** + **Docker Compose** (Compose v2 plugin)
- **Docker volumes**: persist Postgres and Qdrant data

### Reverse proxy / web edge
- **Nginx** container:
  - Terminates HTTP (and optionally HTTPS)
  - Routes `/` → Streamlit UI
  - Routes `/api/` → FastAPI backend

### Application services (containers)
- **FastAPI** (backend)
- **Streamlit** (UI)
- **PostgreSQL** (database)
- **Qdrant** (vector database)

### Secrets & configuration
- Keep non-secret defaults in `app/global.yaml` (OK for Git)
- Keep secrets on the VM in a **local-only** `.env` file (do not commit)
  - Example secrets: `OPENAI_API_KEY`, DB passwords, any auth keys

---

## Repo work required (before touching GCP)

We need the repo to be “container-ready”:

1) Add `Dockerfile` for the FastAPI service
2) Add `Dockerfile` for the Streamlit UI service
3) Add a production compose file (recommended name): `docker-compose.prod.yml` containing:
   - `api` (built from repo)
   - `ui` (built from repo)
   - `postgres` (image `postgres:16`)
   - `qdrant` (image `qdrant/qdrant:latest`)
   - `nginx` (image `nginx:alpine`, with a mounted config)
   - volumes: `pg_data`, `qdrant_data`

Note: this repo currently has `docker-compose.yml` for Postgres + Qdrant only. `docker-compose.prod.yml` is the “everything” file for the VM.

---

## GCP steps (single VM)

### 0) Prereqs
- A domain you control (recommended for HTTPS + discoverability)
- SSH key access to the VM (do not rely only on password auth)
- Local tools: `gcloud` (optional but convenient)

### 1) Create a GCP project and enable billing
In Google Cloud Console:
- Create a project
- Ensure billing is enabled (free trial still requires billing setup)

### 2) Create a Compute Engine VM
Recommended baseline for this stack:
- Machine: `e2-standard-2` (adjust as needed)
- Boot disk: Ubuntu LTS (e.g., 22.04/24.04)
- Disk size: increase if you expect large embeddings/data
- Network tags / firewall: allow `http-server` and `https-server`

Minimum inbound ports:
- `80/tcp` (HTTP)
- `443/tcp` (HTTPS)
- `22/tcp` (SSH) — restrict by IP if possible

### 3) Install Docker + Compose on the VM
SSH into the VM and install Docker Engine and the Compose plugin.

After install, verify:
- `docker --version`
- `docker compose version`

### 4) Get the code onto the VM
Options:
- `git clone` the repo onto the VM
- or copy a release artifact (later, CI/CD can automate this)

### 5) Create the VM-only `.env` file (secrets)
On the VM (in the repo directory), create a file like `.env.prod`:
- `OPENAI_API_KEY=...`
- any other secrets you need

Important:
- Do **not** commit this file
- Do **not** paste secrets into `app/global.yaml`

### 6) Configure app settings
Keep `app/global.yaml` for non-secret defaults. For production you typically override:
- `POSTGRESQL_URL` → points to `postgres` container (not `localhost`)
- `QDRANT_URL` → points to `qdrant` container (not `localhost`)
- `EMBEDDING_PROFILE.provider` → recommended: `"openai"` for deployment

If the app currently only reads from `app/global.yaml`, we’ll keep using it for now.
Later, we should add **env var overrides** so URLs/secrets don’t need to be in YAML.

### 7) Start the stack (Docker Compose)
From the repo root on the VM:
- Build images: `docker compose -f docker-compose.prod.yml build`
- Start: `docker compose -f docker-compose.prod.yml up -d`
- Check: `docker compose -f docker-compose.prod.yml ps`
- Logs: `docker compose -f docker-compose.prod.yml logs -f --tail=200`

### 8) Verify from the public internet
- Visit `http://<VM_EXTERNAL_IP>/` (UI)
- Confirm UI can call API endpoints successfully

### 9) Add a domain + HTTPS (recommended)

1) Point DNS to the VM external IP:
   - `A` record: `yourdomain.com` → `<VM_EXTERNAL_IP>`
   - optionally `A` record: `www.yourdomain.com` → `<VM_EXTERNAL_IP>`

2) Enable HTTPS:
   - Option A (simpler long-term): run cert issuance/renewal on the VM (e.g., Let’s Encrypt) and mount certs into Nginx
   - Option B (more “GCP”): put a Google Cloud HTTPS Load Balancer in front (more moving parts; usually overkill for v1)

For v1, Option A is recommended.

### 10) Make it “stay up”
At minimum:
- Configure auto-restart: Compose `restart: unless-stopped` on services
- (Optional) Create a `systemd` unit to run `docker compose up -d` on boot
- Monitor disk usage (vector data can grow)

---

## Nginx routing model (one domain)

We’ll use one public domain and route based on path:
- `/` → Streamlit UI container port (usually `8501`)
- `/api/` → FastAPI container port (usually `8000`)

This keeps the app shareable as a single website URL, which is ideal for your “search engine accessible” goal.

---

## Data durability & backups (v1 reality)

In the single-VM approach:
- **Postgres data** lives on the VM disk via a Docker volume
- **Qdrant storage** lives on the VM disk via a Docker volume

That means:
- If the VM is deleted without preserving disks/volumes, you lose data
- You should plan a simple backup routine:
  - Postgres: periodic `pg_dump`
  - Qdrant: snapshot/backup strategy (or keep it rebuildable from source docs)

---

## Security basics (do these early)

- Use SSH keys, disable password SSH if possible
- Restrict SSH ingress by IP (firewall rule)
- Keep `postgres` and `qdrant` ports **not publicly exposed**
  - Only Nginx should be reachable from the public internet
- Keep secrets out of Git (`.env.prod` on VM only)

---

## Expansion path: move from single-VM → “4-service” best practice

Once the app is stable and you want easier scaling/ops:

1) Move **FastAPI** to **Cloud Run**
2) Move **Streamlit** to **Cloud Run**
3) Move **Postgres** to **Cloud SQL**
4) Keep **Qdrant** on a dedicated **Compute Engine VM** (or replace with a managed vector service)
5) Put secrets in **Secret Manager**
6) Add CI/CD:
   - Build/push images to Artifact Registry (or GHCR)
   - Deploy Cloud Run revisions on `main` pushes

This preserves the same logical services, but gives you:
- independent scaling of UI/API
- managed Postgres durability/backups
- cleaner security boundaries
