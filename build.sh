#!/usr/bin/env bash
# Packages the module into a module.tar.gz archive for upload to the Viam registry.
set -euo pipefail

cd "$(dirname "$0")"

tar -czf module.tar.gz \
    meta.json \
    requirements.txt \
    setup.sh \
    run.sh \
    src \
    params \
    launch

echo "Wrote module.tar.gz"
