# 🚀 Deployment Stack

### Registry: GitHub Container Registry (GHCR)

* **Target:** `ghcr.io/<username>/<repo>:latest`
* **Access:** Package visibility must be set to **Public** to allow Azure to pull the image without complex credential management.

### Compute: Azure Container Apps (ACA)

* **Scaling:** Set `min-replicas: 0` and `max-replicas: 1` to enable serverless "scale-to-zero."
* **Ingress:** Enable External Ingress; map the target port to the one exposed in the Dockerfile (e.g., `8080`).
* **Configuration:** Inject all sensitive credentials (OpenAI Key, DB strings) as **Secrets** within the ACA environment.

### AI Logic: OpenAI API

* **Models:** Use `text-embedding-3-small` for vectorization and `gpt-4o-mini` for reasoning.
* **Optimization:** Remove all local Ollama dependencies and model weights. Rely strictly on HTTP requests to OpenAI endpoints to keep the image footprint small.

### CI/CD: GitHub Actions

* **Workflow:** Trigger on `push` to `main`.
* **Process:** Build the Docker image  Push to GHCR  Deploy revision to Azure Container App.