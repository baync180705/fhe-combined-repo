# PPML Dataset Compatibility Contract

Blindinference dataset uploads for training are now normalized into a PPML-compatible encrypted dataset artifact.

## Source Of Truth

- Encryption backend: `tfhe-rs` radix
- Serialization format: `EncryptedTensor::to_bytes()`
- Quantization profile: `QuantConfig::q16f8()`
- Artifact schema owner: [`PPML/ppml_core/src/export.rs`](/home/budhayan/Documents/hackathon/fenix-hackathon/PPML/ppml_core/src/export.rs)
- Encryptor entrypoint: [`PPML/ppml_train/src/bin/encrypt_dataset.rs`](/home/budhayan/Documents/hackathon/fenix-hackathon/PPML/ppml_train/src/bin/encrypt_dataset.rs)

## Upload Contract

1. The data source uploads a plaintext CSV dataset to the backend.
2. The backend parses CSV rows, separates feature columns from the label column, and writes a temporary encryption request JSON.
3. The Rust encryptor loads or generates the PPML TFHE key cache.
4. Features and labels are encrypted as PPML `EncryptedTensor` values.
5. The encryptor writes a `ppml_encrypted_dataset` JSON artifact.
6. The backend stores that artifact in GridFS and stores only metadata/orchestration fields in Mongo.

## Verification

The downloaded artifact can be verified with:

```bash
cd PPML
cargo run -p ppml_train --bin verify_dataset -- \
  --input /path/to/downloaded_dataset_export.json \
  --key-cache .cache/dataset_keys_q16f8.bin
```

This command reattaches the serialized feature and label tensors to a PPML `FheContext` and checks that the tensor shapes match the exported metadata.

## Stored Artifact Shape

The persisted artifact contains:

- `schema_version`
- `artifact_type = "ppml_encrypted_dataset"`
- `metadata`
  - `backend`
  - `encryption_scheme = "tfhe-rs-radix"`
  - `source_format`
  - `row_count`
  - `feature_count`
  - `label_count`
  - `label_column_index`
  - `feature_names`
  - `label_name`
  - `quantization`
- `encrypted_tensors`
  - `features`
  - `labels`

Each encrypted tensor contains:

- `rows`
- `cols`
- `bytes`

`bytes` is the PPML-native serialized `EncryptedTensor`, so downloaded artifacts can be reattached to an `FheContext` and consumed by PPML-side training code without a second conversion layer.

## Backend Metadata Contract

Mongo manifests store:

- wallet owner
- GridFS file id
- source/original filename
- artifact hash
- quantization metadata
- tensor dimensions and encrypted byte lengths
- visibility and status

Mongo is not the authority for encryption semantics. The PPML export schema is.
