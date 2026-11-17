#!/usr/bin/env bash
# Entrypoint for the nav-stack module. Activates the Python venv and starts the
# module server (builtin SLAM + nav only).
set -euo pipefail

cd "$(dirname "$0")"

# shellcheck disable=SC1091
source .venv/bin/activate

# The module's numpy work is many small matrix ops on the SLAM / nav threads.
# Pip numpy ships OpenBLAS, which by default spins one worker per core and
# busy-waits between calls; that showed up as hundreds of percent CPU while
# the robot sat still. One BLAS thread per call is faster here, and the
# Python threads still run in parallel across cores.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# glibc gives each thread its own malloc arena; with ~10 threads churning
# multi-MB grids that fragments into a large resident set. Two arenas keep
# RSS bounded with no measurable cost at this allocation rate.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

exec python -m src.main "$@"
