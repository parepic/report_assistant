# Base image for both API and UI containers.
# Uses PDM to install locked dependencies without creating a venv.
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PDM_USE_VENV=false

WORKDIR /app

# System deps kept minimal; add OS packages only if runtime requires them.
RUN pip install --no-cache-dir pdm

# Install dependencies first for better layer caching.
COPY pyproject.toml pdm.lock /app/
RUN pdm config python.use_venv false \
    && pdm install --prod --no-editable

# Copy application code last.
COPY . /app

# Default command is overridden by docker-compose per service.
CMD ["python", "-V"]
