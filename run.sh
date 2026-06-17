#!/usr/bin/env bash
# Entrypoint for the nav-stack module. Sources the ROS2 environment (so rclpy and
# the ROS2 message/launch packages are importable), activates the Python venv, and
# starts the module server.
set -euo pipefail

cd "$(dirname "$0")"

# Load ROS paths written by setup.sh when the module env block omits ROS_ENV.
if [ -f ".ros_env" ]; then
    # shellcheck disable=SC1091
    source ".ros_env"
fi

# Source ROS2 if provided. ROS_ENV is the path to the ROS2 setup.bash, optionally
# followed by ':'-separated overlay setup.bash files (matching the convention used
# by viam-soleng:viam-ros2-integration).
if [ -n "${ROS_ENV:-}" ]; then
    IFS=':' read -ra _ros_setups <<< "${ROS_ENV}"
    for _setup in "${_ros_setups[@]}"; do
        if [ -f "${_setup}" ]; then
            # shellcheck disable=SC1090
            source "${_setup}"
        fi
    done
fi
if [ -n "${OVERLAYS:-}" ]; then
    IFS=':' read -ra _overlays <<< "${OVERLAYS}"
    for _setup in "${_overlays[@]}"; do
        if [ -f "${_setup}" ]; then
            # shellcheck disable=SC1090
            source "${_setup}"
        fi
    done
fi

# Work around shared-memory transport issues when viam-server runs as root by
# defaulting to a UDP-only FastDDS profile if the user supplied one.
if [ -n "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
    export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

exec python -m src.main "$@"
