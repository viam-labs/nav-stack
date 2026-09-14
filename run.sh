#!/usr/bin/env bash
# Entrypoint for the nav-stack module. Activates the Python venv and starts the
# module server (builtin SLAM + nav only; no ROS).
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

exec python -m src.main "$@"
