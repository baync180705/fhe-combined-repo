# Blindference

`blindference` is the application and orchestration layer for the Blindference buildathon submission.

It owns:

- Fhenix/CoFHE smart contracts and deployment scripts
- the FastAPI metadata/orchestration backend
- the React frontend for role-aware product flows
- wallet authentication, permits, encryption, and decryption UX

The off-chain encrypted ML engine lives separately in [`PPML`](/home/budhayan/Documents/hackathon/fenix-hackathon/PPML).

## Role Model

Blindference now has two explicit product roles:

- `data_source`
  - uploads CSV datasets that the backend encrypts into PPML-compatible artifacts
  - requests blind inference
  - tracks results and request metadata
- `ai_lab`
  - manages wallet-bound profile metadata pinned to IPFS
  - activates its lab identity on-chain through `ModelRegistry`
  - downloads encrypted datasets, uploads encrypted model artifacts, and operates the supply side

### Source of truth

- Wallet address: user identity
- Signed wallet authentication: backend session / JWT issuance
- `ModelRegistry.registerLab(...)`: canonical AI Lab activation
- IPFS: canonical profile documents
- MongoDB: metadata pointers, orchestration, dataset manifests, submission tracking

MongoDB is intentionally **not** the authority for AI Lab identity. It stores application metadata and operational records only.

## Repository layout

```text
blindference/
├── backend/            # FastAPI router, GridFS bridge, model status endpoint
├── frontend/           # React + Vite application
├── fhenix_inference/   # Hardhat workspace, contracts, deploy scripts, artifacts
└── README.md
```

## What goes where

### `fhenix_inference/`

Use this directory for:

- Solidity contracts
- Hardhat config
- Sepolia deployment
- contract artifacts

### `backend/`

Use this directory for:

- signed wallet auth
- profile URI pointers
- MongoDB / GridFS integration
- dataset encryption orchestration and manifest tracking
- model artifact uploads linked to datasets
- inference submission tracking
- bridging the frontend to off-chain model export status

### `frontend/`

Use this directory for:

- wallet connect + signature auth
- role-aware onboarding and navigation
- CoFHE/Fhenix browser encryption
- AI Lab activation and model registration
- Data Source dataset upload and request tracking
- AI Lab dataset browsing and model provenance views
- permit-based result decryption

## Environment files

There are three separate env locations in this repo.

### 1. Contract deployment env

Create:

- `fhenix_inference/.env`

From:

- `fhenix_inference/.env.example`

Example:

```env
SEPOLIA_RPC_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY
DEPLOYER_PRIVATE_KEY=0xYOUR_PRIVATE_KEY
FEE_TREASURY=0xYOUR_WALLET_ADDRESS
```

Meaning:

- `SEPOLIA_RPC_URL`: Ethereum Sepolia RPC endpoint
- `DEPLOYER_PRIVATE_KEY`: deployer wallet private key with `0x` prefix
- `FEE_TREASURY`: address receiving platform fees

### 2. Frontend env

Create:

- `frontend/.env.local`

From:

- `frontend/.env.example`

Example:

```env
VITE_BLIND_INFERENCE_ADDRESS=0xYOUR_BLIND_INFERENCE_ADDRESS
VITE_INFERENCE_ENGINE_ADDRESS=0xYOUR_INFERENCE_ENGINE_ADDRESS
VITE_MODEL_REGISTRY_ADDRESS=0xYOUR_MODEL_REGISTRY_ADDRESS
VITE_PAYMENT_ESCROW_ADDRESS=0xYOUR_PAYMENT_ESCROW_ADDRESS
VITE_PAYMENT_TOKEN_ADDRESS=0xYOUR_PAYMENT_TOKEN_ADDRESS
VITE_DEFAULT_MODEL_ID=1
VITE_DEFAULT_REQUEST_ID=1
VITE_PINATA_JWT=your_pinata_jwt
VITE_IPFS_GATEWAY_BASE_URL=https://gateway.pinata.cloud/ipfs
```

Meaning:

- `VITE_BLIND_INFERENCE_ADDRESS`: main external inference entrypoint used by the portal
- `VITE_INFERENCE_ENGINE_ADDRESS`: explicit inference engine address
- `VITE_MODEL_REGISTRY_ADDRESS`: model registry contract address
- `VITE_PAYMENT_ESCROW_ADDRESS`: escrow contract address
- `VITE_PAYMENT_TOKEN_ADDRESS`: BFHE/mock payment token address
- `VITE_DEFAULT_MODEL_ID`: default model selection in the portal
- `VITE_DEFAULT_REQUEST_ID`: optional fallback request id for demos
- `VITE_PINATA_JWT`: browser pinning token used to publish profile JSON to IPFS
- `VITE_IPFS_GATEWAY_BASE_URL`: gateway used to fetch `ipfs://...` profile documents

These values should come from the contract deploy output.

### 3. Backend env

Create:

- `backend/.env`

Example:

```env
MONGO_URI=mongodb://localhost:27017
JWT_SECRET=replace-me-in-non-demo-environments
```

If you are using MongoDB Atlas, use the full Atlas connection string instead.

## Install instructions

### Contracts

```bash
cd blindference/fhenix_inference
npm install
```

### Backend

```bash
cd blindference/backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Frontend

```bash
cd blindference/frontend
npm install
```

## Contract deployment

Deploy to Sepolia from the Hardhat workspace:

```bash
cd blindference/fhenix_inference
npx hardhat compile
npx hardhat run scripts/deploy.ts --network sepolia
```

After deployment:

1. copy the printed addresses
2. place them into `frontend/.env.local`
3. restart the frontend dev server

## Running the stack

Use separate terminals.

### Terminal 1: Backend

```bash
cd blindference/backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Backend health check:

- `http://127.0.0.1:8000/health`

### Terminal 2: Frontend

```bash
cd blindference/frontend
npm run dev -- --force
```

Open the URL printed by Vite.

### Optional: Re-deploy contracts

If contracts changed:

```bash
cd blindference/fhenix_inference
npx hardhat run scripts/deploy.ts --network sepolia
```

Then update `frontend/.env.local` with the new addresses.

### Optional: PPML dataset verification

After a dataset artifact has been downloaded from Blindference, verify that PPML can reattach the serialized tensors:

```bash
cd PPML
cargo run -p ppml_train --bin verify_dataset -- \
  --input /path/to/downloaded_dataset_export.json \
  --key-cache .cache/dataset_keys_q16f8.bin
```

The command succeeds only if the downloaded feature and label tensors deserialize correctly with the PPML TFHE key cache.

## Sepolia wallet setup

The frontend is currently configured for Ethereum Sepolia.

Make sure:

- MetaMask is on Sepolia
- the connected wallet has Sepolia ETH for gas
- the hospital wallet has enough payment tokens for inference

## End-to-end usage flow

### 1. Onboarding

1. Open the frontend
2. Connect MetaMask on Sepolia
3. Choose one role:
   - `Data Source`
   - `AI Lab`
4. Sign the backend authentication message
5. Land in the role-specific workspace

What happens:

- the backend creates a short-lived nonce
- the wallet signs a deterministic Blindference auth message
- the backend verifies the signature and issues a JWT
- the role is tied to the connected wallet for the app shell

### 2. AI Lab flow

1. Connect with the AI Lab wallet
2. Open `Profile` and complete the app-layer profile
3. Open `Lab`
4. Activate the AI Lab on-chain through `ModelRegistry.registerLab(profileURI)`
5. Open `Datasets` and inspect the encrypted training-dataset catalog
6. Download one PPML-compatible dataset artifact
7. Open `Models` and upload an encrypted model artifact linked to exactly one dataset

What happens:

- the profile metadata is stored off-chain in Mongo
- the AI Lab authority is activated on-chain
- downloaded datasets are PPML-compatible encrypted tensor artifacts
- model uploads carry explicit dataset provenance
- dataset cards act as the provenance hub and list all linked trained models

### 3. Data Source flow

1. Connect with a different Data Source wallet
2. Open `Profile` and complete the app-layer profile
3. Open `Upload` to submit a CSV dataset
4. Let the backend convert it into a PPML-compatible encrypted dataset artifact
5. Open `Marketplace` to browse registered models
6. Open `Portal` to request blind inference
7. Approve BFHE if needed
8. Submit inference
9. Decrypt the result with a permit

What happens:

- dataset artifacts are uploaded to GridFS as PPML-compatible encrypted exports
- Mongo stores only manifest/orchestration metadata for those uploads
- the browser encrypts the private inference inputs locally
- the contract computes over encrypted inputs and encrypted weights
- the result handle is ACL-granted to the requesting wallet
- the Data Source signs a permit and decrypts the score off-chain in the UI
- Mongo stores submission tracking metadata, not the protocol authority

## Architecture notes

### On-chain authority

- AI Lab activation
- model registry state
- inference pricing
- encrypted inference execution

### Off-chain metadata / orchestration

- JWT-backed session metadata
- wallet-bound user profiles
- dataset manifests
- submission tracking records
- GridFS file storage
- PPML export status

### PPML relationship

- `PPML` encrypts datasets for training, verifies downloaded dataset artifacts, and exports encrypted model artifacts
- `blindference` handles user roles, product UX, contracts, wallet flows, and metadata orchestration
- dataset downloads can now be checked with `ppml_train --bin verify_dataset` before entering the training flow

## Current role surfaces

- `data_source`
  - `Profile`
  - `Upload`
  - `Marketplace`
  - `Portal`
- `ai_lab`
  - `Profile`
  - `Lab`
  - `Datasets`
  - `Models`

## Frontend ABI source

The frontend imports contract artifacts directly from:

- `fhenix_inference/artifacts/contracts/...`

There is no separate cross-repo ABI export step anymore.

## Troubleshooting

### TypeScript module resolution

This branch includes ambient declarations for several Fhenix/CoFHE modules to keep local type-checking unblocked in environments where those package declarations are incomplete. Runtime still depends on the corresponding packages being installed.

### Wallet connects but FHE client fails

Check:

- MetaMask is on Sepolia
- the contract addresses in `frontend/.env.local` are the latest deployed ones
- you restarted Vite after changing env files

### Model registration reverts with `AI lab not registered`

Use the Lab Dashboard with the AI lab wallet first. The wallet must register as a lab before uploading a model.

### Inference works but decrypt fails

Check:

- the same hospital wallet that submitted the request is trying to decrypt
- the frontend was restarted after CoFHE SDK or env changes
- the result handle came from the current deployed inference contract

### Frontend loads slowly

This app uses browser-side FHE, wasm, workers, and permit storage. First load in dev mode is heavier than a typical React app. The current code defers CoFHE client initialization until it is actually needed.

## Security notes

- Never commit real private keys to git
- Keep `frontend/.env.local` local-only
- Keep `backend/.env` local-only
- Rotate any private key or Mongo credential that has been exposed in logs, screenshots, or chat
