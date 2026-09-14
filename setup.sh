#!/usr/bin/env bash
# First-run setup for the nav-stack module (builtin SLAM + nav; no ROS).
#
# Creates the Python venv and pip-installs module requirements.
set -euo pipefail

cd "$(dirname "$0")"

VENV_NAME=".venv"
PYTHON=${PYTHON:-python3}

log() { echo "[nav-stack setup] $*"; }
die() { echo "[nav-stack setup] ERROR: $*" >&2; exit 1; }

ensure_python_venv() {
    if ! "$PYTHON" --version >/dev/null 2>&1; then
        die "python3 not found on PATH"
    fi
    if [ ! -d "${VENV_NAME}" ]; then
        log "creating virtualenv in ${VENV_NAME}"
        if ! "$PYTHON" -m venv --system-site-packages "${VENV_NAME}"; then
            die "failed to create ${VENV_NAME}; install python3-venv (e.g. apt install python3-venv) and re-run setup"
        fi
    fi
    # shellcheck disable=SC1091
    source "${VENV_NAME}/bin/activate"
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    log "Python dependencies installed"
}

ensure_python_venv
log "setup complete (builtin slam + nav)"
