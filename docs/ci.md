# Continuous Integration (GitHub Actions)

- Runs on push/PR to `clean-slate`
- Python 3.11, installs `requirements-dev.txt`
- Runs `pytest -q`
- Builds Docker image on successful tests

> To add publishing, extend `docker-build` with registry login and `docker push`.
