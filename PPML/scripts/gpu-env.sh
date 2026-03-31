#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CUDA_ROOT:-}" ]]; then
  for candidate in /opt/cuda /usr/local/cuda; do
    if [[ -d "${candidate}" ]]; then
      CUDA_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CUDA_ROOT:-}" ]]; then
  echo "CUDA_ROOT is not set and no CUDA toolkit was found under /opt/cuda or /usr/local/cuda" >&2
  exit 1
fi

export PATH="${CUDA_ROOT}/bin:${PATH}"
export CUDACXX="${CUDACXX:-${CUDA_ROOT}/bin/nvcc}"
export CMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-${CUDA_ROOT}/bin/nvcc}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-${CUDA_ROOT}}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR:-${CUDA_ROOT}}"
export LIBRARY_PATH="${CUDA_ROOT}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }-L native=${CUDA_ROOT}/lib64"
export CC="${CC:-gcc}"
export CXX="${CXX:-g++}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-${CXX}}"

echo "Configured CUDA toolchain from ${CUDA_ROOT}"
