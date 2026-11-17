#!/usr/bin/env bash
# Packages the module into a module.tar.gz archive for upload to the Viam registry.
set -euo pipefail

cd "$(dirname "$0")"

# Exclude large ONNX weights from the module archive. The module downloads
# them (tools/fetch_*_onnx.py, GitHub release assets) the first time vision
# is enabled; requirements-vision.txt is installed then too.
tar -czf module.tar.gz \
    --exclude='src/semantic/models/*.onnx' \
    --exclude='src/semantic/models/*.partial' \
    --exclude='src/semantic/models/*.stamp' \
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
