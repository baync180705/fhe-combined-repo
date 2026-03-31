use std::env;
use std::fs;
use std::path::PathBuf;

use ppml_core::context::FheContext;
use ppml_core::export::write_dataset_export;
use ppml_core::quantization::{config::QuantConfig, quantizer::Quantizer};
use ppml_core::tensor::{FheTensorOps, TensorError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct DatasetEncryptionRequest {
    source_format: String,
    label_column_index: usize,
    feature_rows: Vec<Vec<f64>>,
    label_rows: Vec<Vec<f64>>,
    feature_names: Option<Vec<String>>,
    label_name: Option<String>,
}

#[derive(Debug, Serialize)]
struct EncryptionSummary {
    backend: String,
    row_count: usize,
    feature_count: usize,
    label_count: usize,
    output_path: String,
}

fn main() -> Result<(), TensorError> {
    let args = Args::parse(env::args().skip(1))?;

    let request_bytes = fs::read(&args.input)
        .map_err(|error| TensorError::Io(format!("failed to read encryption request: {error}")))?;
    let request: DatasetEncryptionRequest =
        serde_json::from_slice(&request_bytes).map_err(|error| {
            TensorError::Io(format!("failed to parse encryption request JSON: {error}"))
        })?;

    validate_request(&request)?;

    if let Some(parent) = args.key_cache.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            TensorError::Io(format!("failed to create key cache directory: {error}"))
        })?;
    }
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            TensorError::Io(format!("failed to create output directory: {error}"))
        })?;
    }

    let quant = QuantConfig::q16f8();
    let quantizer = Quantizer::new(quant.clone());
    let (client_key, ctx) = FheContext::load_or_generate(&args.key_cache).map_err(|error| {
        TensorError::Io(format!("failed to load or generate tfhe keys: {error}"))
    })?;

    let features = quantizer.encrypt_tensor(&request.feature_rows, &client_key, ctx.clone())?;
    let labels = quantizer.encrypt_tensor(&request.label_rows, &client_key, ctx.clone())?;

    write_dataset_export(
        &args.output,
        &features,
        &labels,
        &quant,
        ctx.backend_name(),
        &request.source_format,
        request.label_column_index,
        request.feature_names,
        request.label_name,
    )?;

    let summary = EncryptionSummary {
        backend: ctx.backend_name().to_string(),
        row_count: features.shape().rows(),
        feature_count: features.shape().cols(),
        label_count: labels.shape().cols(),
        output_path: args.output.display().to_string(),
    };

    println!(
        "{}",
        serde_json::to_string(&summary).map_err(|error| TensorError::Io(format!(
            "failed to serialize encryption summary: {error}"
        )))?
    );

    Ok(())
}

fn validate_request(request: &DatasetEncryptionRequest) -> Result<(), TensorError> {
    if request.feature_rows.is_empty() {
        return Err(TensorError::InvalidShape(
            "dataset encryption requires at least one feature row".to_string(),
        ));
    }
    if request.label_rows.is_empty() {
        return Err(TensorError::InvalidShape(
            "dataset encryption requires at least one label row".to_string(),
        ));
    }
    if request.feature_rows.len() != request.label_rows.len() {
        return Err(TensorError::InvalidShape(format!(
            "feature row count {} must match label row count {}",
            request.feature_rows.len(),
            request.label_rows.len()
        )));
    }

    let feature_width = request.feature_rows[0].len();
    if feature_width == 0 {
        return Err(TensorError::InvalidShape(
            "feature rows must contain at least one value".to_string(),
        ));
    }
    for row in &request.feature_rows {
        if row.len() != feature_width {
            return Err(TensorError::InvalidShape(
                "all feature rows must have the same width".to_string(),
            ));
        }
    }

    let label_width = request.label_rows[0].len();
    if label_width == 0 {
        return Err(TensorError::InvalidShape(
            "label rows must contain at least one value".to_string(),
        ));
    }
    for row in &request.label_rows {
        if row.len() != label_width {
            return Err(TensorError::InvalidShape(
                "all label rows must have the same width".to_string(),
            ));
        }
    }

    Ok(())
}

struct Args {
    input: PathBuf,
    output: PathBuf,
    key_cache: PathBuf,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, TensorError> {
        let mut input = None;
        let mut output = None;
        let mut key_cache = None;

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => input = args.next().map(PathBuf::from),
                "--output" => output = args.next().map(PathBuf::from),
                "--key-cache" => key_cache = args.next().map(PathBuf::from),
                unexpected => {
                    return Err(TensorError::InvalidShape(format!(
                        "unexpected encrypt_dataset argument: {unexpected}"
                    )))
                }
            }
        }

        let usage = "usage: cargo run -p ppml_train --bin encrypt_dataset -- --input request.json --output dataset_export.json --key-cache dataset_keys.bin";

        Ok(Self {
            input: input.ok_or_else(|| TensorError::InvalidShape(usage.to_string()))?,
            output: output.ok_or_else(|| TensorError::InvalidShape(usage.to_string()))?,
            key_cache: key_cache.ok_or_else(|| TensorError::InvalidShape(usage.to_string()))?,
        })
    }
}
