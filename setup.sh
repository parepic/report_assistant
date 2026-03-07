#!/usr/bin/env bash
# Start production-style Docker Compose stack.
# Assumes a valid .env file already exists in the repo root.

docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
