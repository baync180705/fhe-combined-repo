#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec maturin develop --release -m blindml/Cargo.toml "$@"
