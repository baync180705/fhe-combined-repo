# Wave 1 Submission Flow

## Demo goal

Show that Blindference supports two clear actors in a confidential AI marketplace:

- `Data Source`
- `AI Lab`

while keeping:

- wallet identity cryptographic
- AI Lab authority on-chain
- PPML-compatible encrypted training artifacts off-chain
- encrypted compute in the Fhenix flow
- MongoDB limited to metadata/orchestration

## Suggested demo path

### 1. Connect as AI Lab

1. Connect MetaMask with wallet A.
2. Choose `AI Lab`.
3. Sign the Blindference authentication message.
4. Open `Profile` and show the wallet-bound app metadata.
5. Open `Lab` and activate the AI Lab on-chain.
6. Open `Datasets` and show the encrypted training-dataset catalog.
7. Download one PPML-compatible dataset artifact.
8. Open `Models` and upload an encrypted model artifact linked to that dataset.

Call out:

- wallet-signed authentication
- on-chain AI Lab activation through `ModelRegistry`
- dataset downloads are PPML-compatible encrypted tensor artifacts
- model uploads carry explicit dataset provenance

### 2. Switch to Data Source

1. Switch MetaMask to wallet B.
2. Choose `Data Source`.
3. Sign the Blindference authentication message.
4. Open `Profile` and show the wallet-bound app metadata.
5. Open `Upload` and upload a CSV dataset.
6. Explain that the backend encrypts it into a `tfhe-rs` radix artifact compatible with PPML.
7. Open `Marketplace` and choose a model.
8. Open `Portal` and submit blind inference.
9. Decrypt the result.

Call out:

- Data Source CSV is turned into a PPML-compatible encrypted dataset artifact
- Mongo only stores manifests / orchestration metadata
- on-chain contracts handle pricing and encrypted compute
- only the requesting wallet can decrypt the result

## What to say clearly

- Wallet address is the user identity.
- Backend auth is signature-based, not address-by-POST.
- AI Lab identity is canonical on-chain.
- MongoDB is not the protocol authority.
- PPML is the off-chain encrypted ML engine.
- Blindference is the product, wallet, contract, and orchestration layer.
- Datasets are the provenance hub object linking uploaded training data to downstream model artifacts.

## Known boundaries

- Profile metadata is off-chain.
- Dataset manifests and submission tracking are off-chain.
- AI Lab authority and encrypted inference are on-chain.
- Dataset artifact verification is available through the PPML `verify_dataset` binary.
