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

# Debian/Ubuntu images often ship python3 without ensurepip. Install the
# matching venv package when we are root or have passwordless sudo.
install_venv_package() {
    command -v apt-get >/dev/null 2>&1 || return 1
    local sudo=""
    if [ "$(id -u)" != "0" ]; then
        sudo -n true 2>/dev/null || return 1
        sudo="sudo -n"
    fi
    local ver
    ver=$("$PYTHON" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
    log "installing python${ver}-venv"
    export DEBIAN_FRONTEND=noninteractive
    $sudo apt-get update -qq && $sudo apt-get install -y -qq "python${ver}-venv"
}

ensure_python_venv() {
    if ! "$PYTHON" --version >/dev/null 2>&1; then
        die "python3 not found on PATH"
    fi
    # A failed earlier run can leave a .venv without bin/activate; start over.
    if [ -d "${VENV_NAME}" ] && [ ! -f "${VENV_NAME}/bin/activate" ]; then
        log "removing incomplete ${VENV_NAME}"
        rm -rf "${VENV_NAME}"
    fi
    if [ ! -d "${VENV_NAME}" ]; then
        log "creating virtualenv in ${VENV_NAME}"
        if ! "$PYTHON" -m venv --system-site-packages "${VENV_NAME}"; then
            rm -rf "${VENV_NAME}"
            install_venv_package || die "failed to create ${VENV_NAME}; install python3-venv (e.g. apt install python3-venv) and re-run setup"
            "$PYTHON" -m venv --system-site-packages "${VENV_NAME}" \
                || { rm -rf "${VENV_NAME}"; die "failed to create ${VENV_NAME} after installing python3-venv"; }
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
