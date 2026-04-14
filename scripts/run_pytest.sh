#!/usr/bin/env bash
# ROS Humble's pytest plugins (launch_testing*) can break collection; disable autoload.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec python3 -m pytest tests/ "$@"
