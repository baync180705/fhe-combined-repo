#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source ./scripts/gpu-env.sh
exec maturin develop --release -m blindml/Cargo.toml --features gpu "$@"
