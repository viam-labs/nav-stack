#!/usr/bin/env bash
# Entrypoint for the nav-stack module. Sources the ROS2 environment (so rclpy and
# the ROS2 message/launch packages are importable), activates the Python venv, and
# starts the module server.
set -euo pipefail

cd "$(dirname "$0")"

# ROS setup.bash scripts reference optional vars (e.g. AMENT_TRACE_SETUP_FILES) that
# are unset on a fresh shell; set -u would abort while sourcing them.
source_ros_setup() {
    local setup_file="$1"
    set +u
    # shellcheck disable=SC1090
    source "${setup_file}"
    set -u
}

# Load ROS paths written by setup.sh when the module env block omits ROS_ENV.
if [ -f ".ros_env" ]; then
    # shellcheck disable=SC1091
    set -a
    source ".ros_env"
    set +a
fi

# Source ROS2 if provided. ROS_ENV is the path to the ROS2 setup.bash, optionally
# followed by ':'-separated overlay setup.bash files (matching the convention used
# by viam-soleng:viam-ros2-integration).
if [ -n "${ROS_ENV:-}" ]; then
    IFS=':' read -ra _ros_setups <<< "${ROS_ENV}"
    for _setup in "${_ros_setups[@]}"; do
        if [ -f "${_setup}" ]; then
            source_ros_setup "${_setup}"
        fi
    done
fi
if [ -n "${OVERLAYS:-}" ]; then
    IFS=':' read -ra _overlays <<< "${OVERLAYS}"
    for _setup in "${_overlays[@]}"; do
        if [ -f "${_setup}" ]; then
            source_ros_setup "${_setup}"
        fi
    done
fi

# Keep DDS settings consistent for the in-process bridge and Nav2 child processes.
# viam-server often runs as root; LOCALHOST discovery avoids root/non-root SHM splits
# and (with a unique ROS_DOMAIN_ID) blocks cross-machine /map crosstalk on the LAN.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
# ROS logs default to stderr; route to stdout so WARN lines do not appear as
# modmanager StdErr "error" stream entries.
export RCUTILS_LOGGING_USE_STDOUT="${RCUTILS_LOGGING_USE_STDOUT:-1}"

# Default to UDP-only FastDDS transport. Shared-memory + root + SIGKILL'd Nav2
# leftovers frequently leave /dev/shm segments whose robust mutexes hang the
# next participant create (Nav2 procs alive at 0% CPU, never on the graph).
# Explicit FASTRTPS_DEFAULT_PROFILES_FILE / FASTDDS_BUILTIN_TRANSPORTS still win.
if [ -z "${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
    export FASTDDS_BUILTIN_TRANSPORTS="${FASTDDS_BUILTIN_TRANSPORTS:-UDPv4}"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Apply / persist a stable non-zero ROS_DOMAIN_ID when unset (also sets the
# discovery defaults above if somehow cleared). Explicit env always wins.
python -c "from src.ros.dds_env import apply_dds_isolation; apply_dds_isolation()"

exec python -m src.main "$@"
