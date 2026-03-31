use std::env;
use std::fs;
use std::path::PathBuf;

use ppml_core::context::FheContext;
use ppml_core::tensor::{EncryptedTensor, FheTensorOps, TensorError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct DatasetArtifact {
    schema_version: u32,
    artifact_type: String,
    metadata: DatasetMetadata,
    encrypted_tensors: DatasetEncryptedTensors,
}

#[derive(Debug, Deserialize)]
struct DatasetMetadata {
    backend: String,
    encryption_scheme: String,
    source_format: String,
    row_count: usize,
    feature_count: usize,
    label_count: usize,
    label_column_index: usize,
}

#[derive(Debug, Deserialize)]
struct DatasetEncryptedTensors {
    features: TensorArtifact,
    labels: TensorArtifact,
}

#[derive(Debug, Deserialize)]
struct TensorArtifact {
    rows: usize,
    cols: usize,
    bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
struct VerificationSummary {
    schema_version: u32,
    artifact_type: String,
    backend: String,
    encryption_scheme: String,
    source_format: String,
    row_count: usize,
    feature_count: usize,
    label_count: usize,
    label_column_index: usize,
}

fn main() -> Result<(), TensorError> {
    let args = Args::parse(env::args().skip(1))?;

    let artifact_bytes = fs::read(&args.input)
        .map_err(|error| TensorError::Io(format!("failed to read dataset artifact: {error}")))?;
    let artifact: DatasetArtifact = serde_json::from_slice(&artifact_bytes).map_err(|error| {
        TensorError::Io(format!("failed to parse dataset artifact JSON: {error}"))
    })?;

    if artifact.artifact_type != "ppml_encrypted_dataset" {
        return Err(TensorError::Io(format!(
            "unexpected artifact_type {}, expected ppml_encrypted_dataset",
            artifact.artifact_type
        )));
    }

    let (_client_key, ctx) = FheContext::load_or_generate(&args.key_cache).map_err(|error| {
        TensorError::Io(format!("failed to load or generate tfhe keys: {error}"))
    })?;

    let features =
        EncryptedTensor::from_bytes(&artifact.encrypted_tensors.features.bytes, ctx.clone())?;
    let labels = EncryptedTensor::from_bytes(&artifact.encrypted_tensors.labels.bytes, ctx)?;

    if features.shape().rows() != artifact.metadata.row_count
        || features.shape().cols() != artifact.metadata.feature_count
    {
        return Err(TensorError::InvalidShape(format!(
            "feature tensor shape mismatch: metadata {}x{}, tensor {}x{}",
            artifact.metadata.row_count,
            artifact.metadata.feature_count,
            features.shape().rows(),
            features.shape().cols()
        )));
    }

    if labels.shape().rows() != artifact.metadata.row_count
        || labels.shape().cols() != artifact.metadata.label_count
    {
        return Err(TensorError::InvalidShape(format!(
            "label tensor shape mismatch: metadata {}x{}, tensor {}x{}",
            artifact.metadata.row_count,
            artifact.metadata.label_count,
            labels.shape().rows(),
            labels.shape().cols()
        )));
    }

    if artifact.encrypted_tensors.features.rows != features.shape().rows()
        || artifact.encrypted_tensors.features.cols != features.shape().cols()
    {
        return Err(TensorError::InvalidShape(
            "feature tensor artifact metadata does not match serialized tensor".to_string(),
        ));
    }

    if artifact.encrypted_tensors.labels.rows != labels.shape().rows()
        || artifact.encrypted_tensors.labels.cols != labels.shape().cols()
    {
        return Err(TensorError::InvalidShape(
            "label tensor artifact metadata does not match serialized tensor".to_string(),
        ));
    }

    let summary = VerificationSummary {
        schema_version: artifact.schema_version,
        artifact_type: artifact.artifact_type,
        backend: artifact.metadata.backend,
        encryption_scheme: artifact.metadata.encryption_scheme,
        source_format: artifact.metadata.source_format,
        row_count: artifact.metadata.row_count,
        feature_count: artifact.metadata.feature_count,
        label_count: artifact.metadata.label_count,
        label_column_index: artifact.metadata.label_column_index,
    };

    println!(
        "{}",
        serde_json::to_string(&summary).map_err(|error| TensorError::Io(format!(
            "failed to serialize verification summary: {error}"
        )))?
    );

    Ok(())
}

struct Args {
    input: PathBuf,
    key_cache: PathBuf,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, TensorError> {
        let mut input = None;
        let mut key_cache = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = args.next().map(PathBuf::from),
                "--key-cache" => key_cache = args.next().map(PathBuf::from),
                unexpected => {
                    return Err(TensorError::InvalidShape(format!(
                        "unexpected verify_dataset argument: {unexpected}"
                    )))
                }
            }
        }

        let usage =
            "usage: cargo run -p ppml_train --bin verify_dataset -- --input dataset_export.json --key-cache dataset_keys.bin";

        Ok(Self {
            input: input.ok_or_else(|| TensorError::InvalidShape(usage.to_string()))?,
            key_cache: key_cache.ok_or_else(|| TensorError::InvalidShape(usage.to_string()))?,
        })
    }
}
