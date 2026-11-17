#!/usr/bin/env bash
# First-run setup for the nav-stack module (builtin SLAM + nav; no ROS).
#
# Installs the module's own Python with uv, so the module does not depend on
# the Python the OS ships (Ubuntu 20.04 has 3.8; viam-sdk needs 3.10+), then
# creates .venv and installs requirements.txt into it.
#
# uv, its Python builds and its download cache live under
# ~/.viam/nav-stack/uv and are shared by every module version, so an upgrade
# does not download Python again.
#
# If uv cannot be fetched, setup falls back to a venv from the system python3
# when that is new enough. NAV_STACK_USE_UV=0 forces the fallback.
set -euo pipefail

cd "$(dirname "$0")"

VENV_NAME=".venv"
PYTHON=${PYTHON:-python3}
PYTHON_VERSION=${NAV_STACK_PYTHON_VERSION:-3.12}
UV_VERSION="0.11.32"
UV_HOME=${NAV_STACK_UV_HOME:-${HOME:-$PWD}/.viam/nav-stack/uv}
UV_BIN_DIR="${UV_HOME}/bin/${UV_VERSION}"

log() { echo "[nav-stack setup] $*"; }
die() { echo "[nav-stack setup] ERROR: $*" >&2; exit 1; }

# Download $1 to $2 with whatever the host has.
fetch() {
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1"
    elif command -v "$PYTHON" >/dev/null 2>&1; then
        "$PYTHON" -c 'import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])' "$1" "$2"
    else
        return 1
    fi
}

# Print the path of a usable uv, installing the pinned version if needed.
find_uv() {
    if [ -x "${UV_BIN_DIR}/uv" ]; then
        echo "${UV_BIN_DIR}/uv"
        return 0
    fi
    local installer
    installer=$(mktemp)
    if fetch "https://astral.sh/uv/${UV_VERSION}/install.sh" "$installer" \
        && mkdir -p "$UV_BIN_DIR" \
        && UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh "$installer" >&2 \
        && [ -x "${UV_BIN_DIR}/uv" ]; then
        rm -f "$installer"
        echo "${UV_BIN_DIR}/uv"
        return 0
    fi
    rm -f "$installer"
    if command -v uv >/dev/null 2>&1; then
        log "could not install uv ${UV_VERSION}; using $(command -v uv)" >&2
        command -v uv
        return 0
    fi
    return 1
}

# True when .venv has a working Python new enough for the requirements.
venv_ok() {
    [ -f "${VENV_NAME}/bin/activate" ] \
        && "${VENV_NAME}/bin/python" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null
}

setup_with_uv() {
    local uv
    uv=$(find_uv) || return 1
    export UV_PYTHON_INSTALL_DIR="${UV_HOME}/python"
    export UV_CACHE_DIR="${UV_HOME}/cache"
    export UV_PYTHON_PREFERENCE=only-managed
    export UV_NO_PROGRESS=1

    log "using uv $("$uv" --version | awk '{print $2}') with Python ${PYTHON_VERSION}"
    # --no-bin: do not drop a python3.X launcher into ~/.local/bin.
    "$uv" python install --no-bin "$PYTHON_VERSION" || return 1
    if ! venv_ok; then
        rm -rf "${VENV_NAME}"
        log "creating virtualenv in ${VENV_NAME}"
        # --seed adds pip: enabling vision later pip-installs its runtime.
        "$uv" venv --python "$PYTHON_VERSION" --seed "${VENV_NAME}" || return 1
    fi
    "$uv" pip install --python "${VENV_NAME}/bin/python" -r requirements.txt || return 1
}

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

setup_with_system_python() {
    if ! "$PYTHON" --version >/dev/null 2>&1; then
        die "python3 is not on PATH and no uv-managed Python was set up"
    fi
    if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
        die "system $("$PYTHON" --version 2>&1) is too old (need 3.10+) and no uv-managed Python was set up; unset NAV_STACK_USE_UV=0, or check that this host can reach astral.sh and github.com"
    fi
    # A failed earlier run can leave a .venv without bin/activate; start over.
    if [ -d "${VENV_NAME}" ] && [ ! -f "${VENV_NAME}/bin/activate" ]; then
        log "removing incomplete ${VENV_NAME}"
        rm -rf "${VENV_NAME}"
    fi
    if [ ! -d "${VENV_NAME}" ]; then
        log "creating virtualenv in ${VENV_NAME} with $("$PYTHON" --version 2>&1)"
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
}

if [ "${NAV_STACK_USE_UV:-1}" != "0" ]; then
    if setup_with_uv; then
        log "Python dependencies installed with uv"
        log "setup complete (builtin slam + nav)"
        exit 0
    fi
    log "uv setup failed; falling back to the system python3"
    rm -rf "${VENV_NAME}"
fi
setup_with_system_python
log "Python dependencies installed with pip"
log "setup complete (builtin slam + nav)"
