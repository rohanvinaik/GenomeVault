cat > genomevault/api/app.py <<'PY'
from .main import app  # re-export the canonical FastAPI app
PY
