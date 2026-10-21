#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=2cab8dbb2af9a43ec6443712f011f50ecf58790b
if [[ ! -d upstream/fdtdx/.git ]]; then
    git clone https://github.com/ymahlau/fdtdx.git upstream/fdtdx
    git -C upstream/fdtdx checkout --detach "$revision"
fi
[[ "$(git -C upstream/fdtdx rev-parse HEAD)" == "$revision" ]]
[[ -z "$(git -C upstream/fdtdx status --porcelain)" ]]
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12.15 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:3] == (3, 12, 15), "Expected Python 3.12.15"'
uv pip sync --python .venv/bin/python requirements.lock
