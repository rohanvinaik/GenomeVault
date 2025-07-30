# Local Observability Stack (Prometheus + Grafana)

## Start
```bash
docker compose -f docker-compose.obsv.yml up --build
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default admin/admin)

Dashboards auto-provision using the included overview.
