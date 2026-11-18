#!/usr/bin/env bash
# MORPH-DA one-shot setup: virtualenv + all dependencies + sanity checks.
#
#   bash scripts/setup.sh                    # install, run tests
#   bash scripts/setup.sh --skip-tests       # install only
#   bash scripts/setup.sh --ollama-model qwen3:8b   # also pull a local model
#
# Environment overrides:
#   MORPHDA_VENV         virtualenv directory           (default: .venv)
#   MORPHDA_INDEX_URL    package index to install from  (default: https://pypi.org/simple)
#   MORPHDA_PYTHON       interpreter to build the venv  (default: first python >= 3.11 found)
#
# User/global pip config and proxy variables are deliberately ignored when they
# are unreachable, so the script works on a plain internet connection.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV="${MORPHDA_VENV:-.venv}"
INDEX="${MORPHDA_INDEX_URL:-https://pypi.org/simple}"
RUN_TESTS=1
OLLAMA_MODEL=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-tests)   RUN_TESTS=0 ;;
    --ollama-model) OLLAMA_MODEL="${2:?--ollama-model needs a model name}"; shift ;;
    -h|--help)      sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

say()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# ── 1. Python >= 3.11 ────────────────────────────────────────────────────────
say "Locating Python >= 3.11"
PY=""
for cand in "${MORPHDA_PYTHON:-}" python3.13 python3.12 python3.11 python3 python; do
  [ -n "$cand" ] || continue
  command -v "$cand" >/dev/null 2>&1 || continue
  if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    PY="$(command -v "$cand")"; break
  fi
done
[ -n "$PY" ] || die "Python >= 3.11 not found. Install it (e.g. 'brew install python@3.12') or set MORPHDA_PYTHON."
echo "Using $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])'))"

# ── 2. Drop stale proxy settings ─────────────────────────────────────────────
# A proxy exported by a shell profile (e.g. from another network) makes every
# download hang. Keep it only if it actually accepts connections.
proxy_reachable() {
  "$PY" -I - "$1" <<'EOF'
import socket, sys
from urllib.parse import urlparse
raw = sys.argv[1]
u = urlparse(raw if "//" in raw else "//" + raw)
try:
    socket.create_connection((u.hostname, u.port or 80), timeout=4).close()
except (OSError, TypeError):
    sys.exit(1)
EOF
}

for var in https_proxy HTTPS_PROXY http_proxy HTTP_PROXY all_proxy ALL_PROXY; do
  val="${!var:-}"
  [ -n "$val" ] || continue
  if ! proxy_reachable "$val"; then
    warn "$var points to an unreachable proxy; ignoring it for this run."
    unset "$var"
  fi
done

# ── 3. Virtualenv ────────────────────────────────────────────────────────────
say "Creating virtualenv at $VENV"
if command -v uv >/dev/null 2>&1; then
  INSTALLER=uv
  uv venv --quiet --python "$PY" "$VENV"
  pip_install() { uv pip install --quiet --python "$VENV/bin/python" --index-url "$INDEX" "$@"; }
else
  INSTALLER=pip
  "$PY" -m venv "$VENV"
  # --isolated: ignore user/global pip.conf and PIP_* variables.
  pip_install() { "$VENV/bin/python" -m pip --isolated install --quiet --index-url "$INDEX" "$@"; }
  "$VENV/bin/python" -m pip --isolated install --quiet --index-url "$INDEX" --upgrade pip
fi
echo "Installer: $INSTALLER | index: $INDEX"

# ── 4. Dependencies ──────────────────────────────────────────────────────────
say "Installing MORPH-DA with dev + llm extras"
pip_install -e ".[dev,llm]"

# ── 5. Verify ────────────────────────────────────────────────────────────────
say "Verifying imports"
"$VENV/bin/python" -I - <<'EOF'
import importlib
mods = ["pandas", "numpy", "scipy", "statsmodels", "libcst", "langgraph",
        "pydantic", "yaml", "typer", "requests", "anthropic", "openai", "pytest", "morphda"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m} ({exc.__class__.__name__}: {exc})")
if missing:
    raise SystemExit("Missing/broken: " + "; ".join(missing))
print("all", len(mods), "modules import cleanly")
EOF

say "Benchmark summary"
"$VENV/bin/morphda" info

if [ "$RUN_TESTS" -eq 1 ]; then
  say "Running test suite"
  "$VENV/bin/python" -m pytest -q
fi

# ── 6. Optional local model ──────────────────────────────────────────────────
if [ -n "$OLLAMA_MODEL" ]; then
  say "Pulling local model: $OLLAMA_MODEL"
  if command -v ollama >/dev/null 2>&1; then
    ollama pull "$OLLAMA_MODEL"
  else
    warn "ollama not installed; install Ollama first, then re-run with --ollama-model."
  fi
fi

cat <<EOF

Setup complete.

  source $VENV/bin/activate
  morphda validate                 # 101 reference programs, 0 relation violations expected
  python scripts/run_natural_agents.py --help
EOF
