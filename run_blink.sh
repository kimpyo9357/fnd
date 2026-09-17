#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "${root_dir}/workflows/flow.py" blink "$@"
