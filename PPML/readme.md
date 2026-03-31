[![Fhenix](https://img.shields.io/badge/Fhenix-Compatible_Off--Chain_Engine-4f46e5)](#)
[![Rust](https://img.shields.io/badge/Rust-Core_Engine-orange)](#)
[![Python](https://img.shields.io/badge/Python-BlindML-blue)](#)
[![TFHE-rs](https://img.shields.io/badge/TFHE--rs-Encrypted_ML-111827)](#)

# BlindML / PPML

BlindML is a privacy-preserving machine learning engine for training and exporting encrypted models with Fully Homomorphic Encryption.

This repository is now intentionally scoped to the off-chain engine only:

- Rust FHE runtime and encrypted tensor math
- Python developer bridge via PyO3/Maturin
- local CPU/GPU training and inference scripts
- model export artifacts for downstream application layers

It no longer contains smart contracts, Hardhat deployment code, frontend code, or backend APIs.

That application and blockchain layer now lives in the private `blindference` repository.

## Fhenix compatibility

This package is Fhenix-compatible at the system level.

That means:

- model weights can be exported from this engine for downstream ingestion by the Fhenix-facing application layer
- quantized encrypted model state produced here is intended to bridge into the on-chain inference workflow
- `PPML` is the off-chain encrypted ML engine, while `blindference` is the Fhenix/web3 execution and product layer

In short:

- `PPML` trains and exports encrypted model state
- `blindference` handles contracts, wallet UX, deployment, backend coordination, and browser/user interaction

## Architecture

### 1. The Core Engine: `ppml_core`

`ppml_core` is the cryptographic runtime.

It owns:

- fixed-point quantization
- encrypted tensor operations
- forward and backward model passes
- encrypted optimization
- model export helpers

This crate is the mathematical source of truth for the FHE learning flow.

### 2. The Python Layer: `blindml`

`blindml` wraps the Rust engine with PyO3 and ships as a Python extension via Maturin.

It gives data scientists a notebook-friendly API for:

- creating a context
- fitting encrypted models
- exporting trained parameters

This layer stays intentionally thin and delegates encrypted computation to `ppml_core`.

### 3. The Training Binary: `ppml_train`

`ppml_train` is the executable training entrypoint.

It is responsible for:

- running the training loop
- serializing the final exported model state
- writing `model_export.json` for downstream consumers

### 4. The Multi-Target Build System

The project builds for CPU by default.

GPU support is opt-in through Cargo feature flags:

```bash
--features gpu
```

When `gpu` is enabled:

- CUDA-backed paths are enabled
- heavy encrypted workloads can be accelerated on supported NVIDIA hardware

When `gpu` is not enabled:

- the standard CPU path is used
- no CUDA toolkit is required

This gives three practical operating modes:

- CPU-only training
- GPU-accelerated training
- CPU-only inference/export validation

## Repository Layout

```text
PPML/
├── blindml/                   # PyO3 + maturin Python package
├── ppml_core/                 # Rust FHE engine, tensors, quantization, exporters
├── ppml_train/                # Training and local inference binaries
├── scripts/                   # CPU/GPU helper scripts
├── setup_cloud_gpu.sh         # Cloud GPU bootstrap for Linux/NVIDIA environments
├── test_blindml.py            # Synthetic smoke test
├── test_pima.py               # Stable local test entrypoint
├── requirements.txt           # Python runtime dependencies
└── .cargo/config.toml         # Workspace-level build settings
```

## Environment and configuration

`PPML` does not require the frontend/backend/web3 env files used by `blindference`.

There is no need to create:

- `frontend/.env.local`
- `backend/.env`
- `fhenix_inference/.env`

inside this repo, because those belong to the application repository.

For `PPML`, the main local setup pieces are:

- a Python virtual environment for the wrapper/tests
- a Rust toolchain for the core engine
- optional CUDA support for GPU builds

Typical local bootstrap:

```bash
cd /home/abhieren/Drive/Projects/Buildathon/Fhenix/PPML
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use the Python bridge, also build the extension:

```bash
maturin develop --release -m blindml/Cargo.toml
```

## Execution paths

Choose the path that matches your hardware and goal.

### Case 1: Local Testing (CPU-only)

This is the default local development path.

Setup:

```bash
cd /home/abhieren/Drive/Projects/Buildathon/Fhenix/PPML
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Build the Python package:

```bash
maturin develop --release -m blindml/Cargo.toml
```

Run a smoke test:

```bash
python test_pima.py
```

This path is ideal for:

- local validation
- API development
- notebook prototyping
- debugging the Python wrapper

### Case 2: Cloud GPU Training

This path is for heavy encrypted training workloads and requires CUDA.

Use it when:

- training is too slow on CPU
- the machine has a supported NVIDIA GPU
- you are running on a Linux CUDA instance or cloud notebook

Bootstrap:

```bash
cd /home/abhieren/Drive/Projects/Buildathon/Fhenix/PPML
bash setup_cloud_gpu.sh
```

The GPU build path uses:

```bash
maturin develop --release --features gpu
```

This is the recommended path for:

- remote Linux CUDA workstations
- cloud GPU boxes
- long encrypted training jobs

### Case 3: Exporting Model State for Application Integration

After training completes, the engine exports model state for downstream use.

The important integration artifact is:

- `model_export.json`

This file is intended to be consumed by the application layer in `blindference`.

By convention, the exporter writes:

- [`PPML/model_export.json`](/home/abhieren/Drive/Projects/Buildathon/Fhenix/PPML/model_export.json)

The backend in `blindference` can read this file to expose training/export status.

## What lives in `blindference`

The following moved out of this repo and now live in the private application repository:

- Fhenix-compatible smart contracts
- Hardhat deployment workspace
- frontend wallet and browser encryption flow
- backend upload/download bridge
- ABI/artifact handling for app consumption

So if you are looking for:

- contract deployment
- frontend `.env.local`
- Sepolia/Fhenix web3 setup
- backend API routes

those belong in `blindference`, not in `PPML`.

## Typical workflow

1. Train or fine-tune the encrypted model in `PPML`
2. Export the final model state as `model_export.json`
3. Move to `blindference` for contract deployment, app integration, and user-facing inference flow

## Where to look next

If you need:

- contract deployment
- Sepolia addresses
- frontend wallet configuration
- backend MongoDB configuration
- browser-side encryption/decryption

go to:

- [`blindference/README.md`](/home/abhieren/Drive/Projects/Buildathon/Fhenix/blindference/README.md)

## Notes

- `PPML` is engine-first, not app-first
- the repo is intentionally kept free of smart-contract and frontend/backend product code
- this separation protects the proprietary app/deployment layer while keeping the FHE engine self-contained
