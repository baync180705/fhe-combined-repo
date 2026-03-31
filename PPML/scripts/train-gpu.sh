#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/gpu-env.sh
exec cargo run --release -p ppml_train --features gpu --bin train "$@"
