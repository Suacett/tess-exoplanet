#!/usr/bin/env bash
# bootstrap.sh — One-time setup for the TESS exoplanet pipeline
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'

step()  { echo -e "\n${BOLD}▶ $*${RESET}"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()  { echo -e "  ${YELLOW}!${RESET} $*"; }
die()   { echo -e "\n${RED}ERROR:${RESET} $*" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Python 3.12 check ──────────────────────────────────────────────────────
step "Checking Python 3.12..."

PY312=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "$ver" == "3.12" ]]; then
            PY312="$candidate"
            break
        fi
    fi
done

if [[ -z "$PY312" ]]; then
    die "Python 3.12 not found.\n\n  Ubuntu 24.04:\n    sudo apt install python3.12 python3.12-venv\n\n  Ubuntu 22.04 / 20.04 (deadsnakes PPA):\n    sudo add-apt-repository ppa:deadsnakes/ppa\n    sudo apt update\n    sudo apt install python3.12 python3.12-venv\n\nThen re-run this script."
fi

ok "Found Python 3.12 at $(command -v "$PY312")"

# ── 2. Virtual environment ────────────────────────────────────────────────────
step "Creating virtual environment at ./venv..."

if [[ -d venv ]]; then
    warn "venv already exists — skipping creation"
else
    "$PY312" -m venv venv
    ok "Created venv"
fi

source venv/bin/activate
ok "Activated venv ($(python --version))"

# ── 3. Install Python dependencies ───────────────────────────────────────────
step "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "All packages installed"

# ── 4. Create runtime directory tree ─────────────────────────────────────────
step "Creating data directories..."
mkdir -p data/tess data/results data/candidates logs
ok "data/tess  data/results  data/candidates  logs"

# ── 5. Pull ExoMiner++ container image ───────────────────────────────────────
step "Pulling ExoMiner++ container image (~4 GB, one-time download)..."

if ! command -v podman &>/dev/null; then
    warn "Podman not found — skipping image pull."
    warn "Install Podman: https://podman.io/docs/installation"
    warn "Then run:  podman pull ghcr.io/nasa/exominer:latest"
else
    if podman image exists ghcr.io/nasa/exominer:latest 2>/dev/null; then
        ok "ExoMiner++ image already present"
    else
        podman pull ghcr.io/nasa/exominer:latest
        ok "ExoMiner++ image pulled"
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}Setup complete.${RESET}\n"
echo -e "Next steps:\n"
echo -e "  ${BOLD}source venv/bin/activate${RESET}"
echo -e "  ${BOLD}streamlit run scripts/dashboard.py${RESET}   # full dashboard"
echo -e "  ${BOLD}python scripts/hunt.py --sector 10${RESET}   # run a sector search\n"
echo -e "See README.md for full usage."
