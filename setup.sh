#!/usr/bin/env bash
# First-run setup for the nav-stack module (builtin SLAM + nav; no ROS).
#
# Creates the Python venv and pip-installs module + vision requirements.
set -euo pipefail

cd "$(dirname "$0")"

VENV_NAME=".venv"
PYTHON=${PYTHON:-python3}

log() { echo "[nav-stack setup] $*"; }
die() { echo "[nav-stack setup] ERROR: $*" >&2; exit 1; }

install_vision_deps() {
    [ -f requirements-vision.txt ] || return 0
    log "installing vision deps (onnxruntime)"
    # Prefer OpenVINO EP on Intel NUCs; fall back to CPU onnxruntime.
    if ! pip install -r requirements-vision.txt -q; then
        log "onnxruntime-openvino failed; trying onnxruntime (CPU)"
        pip install "onnxruntime>=1.17.0" -q || die "failed to install onnxruntime"
    fi
}

fetch_dinov2_if_needed() {
    local out="src/semantic/models/dinov2_vits14_patches_448.onnx"
    if [ -f "$out" ] && [ "$(wc -c < "$out" | tr -d ' ')" -gt 1000000 ]; then
        log "DINOv2 ONNX already present ($(du -h "$out" | awk '{print $1}'))"
        return 0
    fi
    if [ ! -f tools/fetch_dinov2_onnx.py ]; then
        return 0
    fi
    log "fetching DINOv2 ONNX weights…"
    python tools/fetch_dinov2_onnx.py || log "DINOv2 fetch failed (vision will stay synthetic until weights are present)"
}

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
    install_vision_deps
    log "Python dependencies installed"
    fetch_dinov2_if_needed
}

ensure_python_venv
log "setup complete (builtin slam + nav + vision)"
