#!/usr/bin/env bash

set -euo pipefail

echo "Configuring PPML GPU build environment for Lightning AI / CUDA cloud instances..."

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

cat <<'EOF'
This script fixes the common CUDA 12 vs GCC 13 mismatch by forcing a CUDA-compatible toolchain.

If you are inside Conda/Anaconda, the preferred path is:
  conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake -y

If you have sudo access on the host, the system path is:
  sudo apt-get update
  sudo apt-get install -y gcc-11 g++-11 cmake
EOF

if command -v conda >/dev/null 2>&1; then
  echo "Conda detected. Installing compatible compiler toolchain with conda-forge..."
  conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake -y

  export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
  export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
else
  echo "Conda not detected. Expecting system gcc-11/g++-11 to be available."
  export CC="${CC:-/usr/bin/gcc-11}"
  export CXX="${CXX:-/usr/bin/g++-11}"
fi

if [[ ! -x "${CC}" ]]; then
  echo "Compiler not found at CC=${CC}"
  echo "Install gcc-11 first, then rerun this script."
  exit 1
fi

if [[ ! -x "${CXX}" ]]; then
  echo "Compiler not found at CXX=${CXX}"
  echo "Install g++-11 first, then rerun this script."
  exit 1
fi

if command -v nvcc >/dev/null 2>&1; then
  export CUDACXX="$(command -v nvcc)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDACXX=/usr/local/cuda/bin/nvcc
elif [[ -x /opt/cuda/bin/nvcc ]]; then
  export CUDACXX=/opt/cuda/bin/nvcc
else
  echo "nvcc not found. Install the CUDA toolkit before building with --features gpu."
  exit 1
fi

export CUDAHOSTCXX="${CXX}"
export CMAKE_CUDA_COMPILER="${CUDACXX}"
export CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$(cd "$(dirname "${CUDACXX}")/.." && pwd)}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR:-${CUDAToolkit_ROOT}}"
export LIBRARY_PATH="${CUDAToolkit_ROOT}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${CUDAToolkit_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "Using compiler toolchain:"
echo "  CC=${CC}"
echo "  CXX=${CXX}"
echo "  CUDACXX=${CUDACXX}"
echo "  CUDAToolkit_ROOT=${CUDAToolkit_ROOT}"

echo "Cleaning old Rust build artifacts..."
cargo clean

echo "Building blindml with GPU support via maturin..."
maturin develop --release -m blindml/Cargo.toml --features gpu

echo "GPU-enabled blindml build complete."
