# Qwen-Free-API + DeepSeek-Free-API merged bridge (free plan: 1 service, 2 bridges)
# nginx :80 -> qwen :8080 (default) + deepseek :8000 (under /deepseek/<secret>/)

# ---- stage 1: qwen go binary ----
FROM golang:1.25-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /qwen-api .

# ---- stage 2: deepseek python deps (cached separately) ----
FROM python:3.12-slim AS dsdeps
WORKDIR /ds
COPY deepseek/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- final ----
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends nginx supervisor ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# qwen
COPY --from=builder /qwen-api /app/qwen-api
COPY .env.example /app/.env.example

# deepseek
COPY --from=dsdeps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dsdeps /usr/local/bin/uvicorn /usr/local/bin/uvicorn
COPY deepseek/ /app/ds/

# front
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/supervisord.conf /etc/supervisord.conf
COPY deploy/nginx-start.sh /app/nginx-start.sh
RUN rm -rf /etc/nginx/sites-enabled /etc/nginx/sites-available/default && ls -la /etc/nginx/ || true
RUN echo "--- nginx dirs after cleanup ---" && find /etc/nginx -maxdepth 2 | sort

ENV HOST=0.0.0.0 PORT=8080 SERVER_INTERACTIVE_LOGIN=0
EXPOSE 80
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
