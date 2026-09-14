#!/usr/bin/env bash
# Bring up nav-stack builtin simulation under a local viam-server.
#
# Usage:
#   ./scripts/run_local_sim.sh           # setup venv if needed, then serve
#   ./scripts/run_local_sim.sh --print   # write resolved config and exit
#   BIND=:8082 ./scripts/run_local_sim.sh
#
# Then point the Viam app / CLI at localhost (default bind :8081, --no-tls),
# teleop sim-base, map with slam, navigate with nav / nav-stack-ui.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

PRINT_ONLY=0
EXTRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    --print|-n) PRINT_ONLY=1 ;;
    *) EXTRA_ARGS+=("${arg}") ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: required command not found: $1" >&2
    exit 1
  }
}

need_cmd python3
need_cmd viam-server

if [ ! -x "${ROOT}/run.sh" ]; then
  echo "error: ${ROOT}/run.sh missing or not executable" >&2
  exit 1
fi

if [ ! -d "${ROOT}/.venv" ]; then
  echo "[local-sim] creating Python venv via setup.sh (ROS-free)…"
  REQUIRE_ROS=0 "${ROOT}/setup.sh"
fi

BIND_ADDRESS="${BIND:-:8081}"
MAPS_DIR="${MAPS_DIR:-${ROOT}/.local/sim-maps}"
OUT_CFG="${OUT_CFG:-${ROOT}/sample_configs/.local_sim.generated.json}"
TEMPLATE="${TEMPLATE:-${ROOT}/sample_configs/local_sim.json}"

mkdir -p "${MAPS_DIR}" "$(dirname "${OUT_CFG}")"

python3 - "${TEMPLATE}" "${OUT_CFG}" "${ROOT}" "${BIND_ADDRESS}" "${MAPS_DIR}" <<'PY'
import json
import sys
from pathlib import Path

template, out_cfg, root, bind, maps_dir = sys.argv[1:6]
root_p = Path(root).resolve()
data = json.loads(Path(template).read_text())
data.pop("_comment", None)
data.setdefault("network", {})["bind_address"] = bind
for mod in data.get("modules") or []:
    if mod.get("type", "local") == "local" or "executable_path" in mod:
        mod["type"] = "local"
        mod["executable_path"] = str(root_p / "run.sh")
for svc in data.get("services") or []:
    attrs = svc.setdefault("attributes", {})
    if "maps_dir" in attrs:
        attrs["maps_dir"] = maps_dir
Path(out_cfg).write_text(json.dumps(data, indent=2) + "\n")
print(out_cfg)
PY

echo "[local-sim] wrote ${OUT_CFG}"
echo "[local-sim] maps_dir=${MAPS_DIR}"
echo "[local-sim] bind=${BIND_ADDRESS}"

if [ "${PRINT_ONLY}" = "1" ]; then
  exit 0
fi

echo "[local-sim] starting: viam-server -config ${OUT_CFG} -no-tls -allow-insecure-creds ${EXTRA_ARGS[*]:-}"
echo "[local-sim] teleop base=sim-base · slam=slam · nav=nav · camera=nav-view"
echo "[local-sim] reset pose: DoCommand on sim-base {\"command\":\"reset\"}"
echo "[local-sim] nav-stack-ui: VITE_HOST=http://localhost${BIND_ADDRESS} (no API keys)"
exec viam-server -config "${OUT_CFG}" -no-tls -allow-insecure-creds "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
