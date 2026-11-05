#!/usr/bin/env bash
# Packages the module into a module.tar.gz archive for upload to the Viam registry.
set -euo pipefail

cd "$(dirname "$0")"

# Exclude large ONNX weights from the module archive (fetch separately on the
# robot via tools/fetch_dinov2_onnx.py / GitHub release assets).
tar -czf module.tar.gz \
    --exclude='src/semantic/models/*.onnx' \
    --exclude='**/__pycache__' \
    --exclude='*.pyc' \
    meta.json \
    requirements.txt \
    requirements-vision.txt \
    setup.sh \
    run.sh \
    src \
    sample_configs \
    scripts \
    tools

echo "Wrote module.tar.gz"
