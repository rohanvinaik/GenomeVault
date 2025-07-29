# Deploy locally with Docker & Compose

## Build & run (foreground)
```bash
docker compose up --build api
```

## Detach / stop
```bash
./scripts/dev_up.sh
./scripts/dev_down.sh
```

## Health check
Open http://localhost:8000/health

## Notes
- Runtime stack: Python 3.11 on slim image; FastAPI + Uvicorn.
- Pinned to Pydantic v1 and FastAPI 0.103.x to match code validators.
