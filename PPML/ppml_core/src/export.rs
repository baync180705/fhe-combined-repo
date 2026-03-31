use std::fs;
use std::path::Path;

use serde::Serialize;

use crate::model::LogisticModel;
use crate::quantization::config::QuantConfig;
use crate::tensor::{EncryptedTensor, FheTensorOps, TensorError};

#[derive(Debug, Serialize)]
pub struct ModelExport {
    pub schema_version: u32,
    pub model_type: &'static str,
    pub weights: Vec<i64>,
    pub bias: Vec<i64>,
    pub metadata: ModelMetadata,
    pub encrypted_tensors: EncryptedTensorArtifacts,
}

#[derive(Debug, Serialize)]
pub struct ModelMetadata {
    pub backend: String,
    pub quantization: QuantizationMetadata,
    pub layers: Vec<LayerMetadata>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizationMetadata {
    pub frac_bits: u32,
    pub total_bits: u32,
    pub scale: i64,
    pub q_min: i64,
    pub q_max: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerMetadata {
    pub name: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub element_count: usize,
    pub encrypted_byte_len: usize,
}

#[derive(Debug, Serialize)]
pub struct EncryptedTensorArtifacts {
    pub weights: EncryptedTensorArtifact,
    pub bias: EncryptedTensorArtifact,
}

#[derive(Debug, Clone, Serialize)]
pub struct EncryptedTensorArtifact {
    pub rows: usize,
    pub cols: usize,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Serialize)]
pub struct DatasetExport {
    pub schema_version: u32,
    pub artifact_type: &'static str,
    pub metadata: DatasetMetadata,
    pub encrypted_tensors: DatasetEncryptedTensorArtifacts,
}

#[derive(Debug, Serialize)]
pub struct DatasetMetadata {
    pub backend: String,
    pub encryption_scheme: &'static str,
    pub source_format: String,
    pub row_count: usize,
    pub feature_count: usize,
    pub label_count: usize,
    pub label_column_index: usize,
    pub feature_names: Option<Vec<String>>,
    pub label_name: Option<String>,
    pub quantization: QuantizationMetadata,
    pub layers: Vec<LayerMetadata>,
}

#[derive(Debug, Serialize)]
pub struct DatasetEncryptedTensorArtifacts {
    pub features: EncryptedTensorArtifact,
    pub labels: EncryptedTensorArtifact,
}

pub fn write_model_export(
    path: &Path,
    model: &LogisticModel,
    quant: &QuantConfig,
    backend_name: &str,
    quantized_weights: Vec<i64>,
    quantized_bias: Vec<i64>,
) -> Result<(), TensorError> {
    let weight_bytes = model.weights.to_bytes()?;
    let bias_bytes = model.bias.to_bytes()?;

    let export = ModelExport {
        schema_version: 1,
        model_type: "encrypted_logistic_regression",
        weights: quantized_weights,
        bias: quantized_bias,
        metadata: ModelMetadata {
            backend: backend_name.to_string(),
            quantization: QuantizationMetadata {
                frac_bits: quant.frac_bits,
                total_bits: quant.total_bits,
                scale: quant.scale,
                q_min: quant.q_min,
                q_max: quant.q_max,
            },
            layers: vec![
                LayerMetadata {
                    name: "weights",
                    rows: model.weights.shape().rows(),
                    cols: model.weights.shape().cols(),
                    element_count: model.weights.shape().rows() * model.weights.shape().cols(),
                    encrypted_byte_len: weight_bytes.len(),
                },
                LayerMetadata {
                    name: "bias",
                    rows: model.bias.shape().rows(),
                    cols: model.bias.shape().cols(),
                    element_count: model.bias.shape().rows() * model.bias.shape().cols(),
                    encrypted_byte_len: bias_bytes.len(),
                },
            ],
        },
        encrypted_tensors: EncryptedTensorArtifacts {
            weights: EncryptedTensorArtifact {
                rows: model.weights.shape().rows(),
                cols: model.weights.shape().cols(),
                bytes: weight_bytes,
            },
            bias: EncryptedTensorArtifact {
                rows: model.bias.shape().rows(),
                cols: model.bias.shape().cols(),
                bytes: bias_bytes,
            },
        },
    };

    let json = serde_json::to_vec_pretty(&export)
        .map_err(|error| TensorError::Io(format!("while serializing model export: {error}")))?;
    fs::write(path, json)
        .map_err(|error| TensorError::Io(format!("while writing model export: {error}")))?;
    Ok(())
}

pub fn write_dataset_export(
    path: &Path,
    features: &EncryptedTensor,
    labels: &EncryptedTensor,
    quant: &QuantConfig,
    backend_name: &str,
    source_format: &str,
    label_column_index: usize,
    feature_names: Option<Vec<String>>,
    label_name: Option<String>,
) -> Result<(), TensorError> {
    if features.shape().rows() != labels.shape().rows() {
        return Err(TensorError::InvalidShape(format!(
            "dataset features rows {} must match label rows {}",
            features.shape().rows(),
            labels.shape().rows()
        )));
    }

    let feature_bytes = features.to_bytes()?;
    let label_bytes = labels.to_bytes()?;

    let export = DatasetExport {
        schema_version: 1,
        artifact_type: "ppml_encrypted_dataset",
        metadata: DatasetMetadata {
            backend: backend_name.to_string(),
            encryption_scheme: "tfhe-rs-radix",
            source_format: source_format.to_string(),
            row_count: features.shape().rows(),
            feature_count: features.shape().cols(),
            label_count: labels.shape().cols(),
            label_column_index,
            feature_names,
            label_name,
            quantization: QuantizationMetadata {
                frac_bits: quant.frac_bits,
                total_bits: quant.total_bits,
                scale: quant.scale,
                q_min: quant.q_min,
                q_max: quant.q_max,
            },
            layers: vec![
                LayerMetadata {
                    name: "features",
                    rows: features.shape().rows(),
                    cols: features.shape().cols(),
                    element_count: features.shape().rows() * features.shape().cols(),
                    encrypted_byte_len: feature_bytes.len(),
                },
                LayerMetadata {
                    name: "labels",
                    rows: labels.shape().rows(),
                    cols: labels.shape().cols(),
                    element_count: labels.shape().rows() * labels.shape().cols(),
                    encrypted_byte_len: label_bytes.len(),
                },
            ],
        },
        encrypted_tensors: DatasetEncryptedTensorArtifacts {
            features: EncryptedTensorArtifact {
                rows: features.shape().rows(),
                cols: features.shape().cols(),
                bytes: feature_bytes,
            },
            labels: EncryptedTensorArtifact {
                rows: labels.shape().rows(),
                cols: labels.shape().cols(),
                bytes: label_bytes,
            },
        },
    };

    let json = serde_json::to_vec_pretty(&export)
        .map_err(|error| TensorError::Io(format!("while serializing dataset export: {error}")))?;
    fs::write(path, json)
        .map_err(|error| TensorError::Io(format!("while writing dataset export: {error}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::context::FheContext;
    use crate::model::LogisticModel;
    use crate::quantization::quantizer::Quantizer;

    #[test]
    fn writes_frontend_compatible_model_export_json() {
        let quant = QuantConfig::q16f8();
        let quantizer = Quantizer::new(quant.clone());
        let (client_key, ctx) = FheContext::generate_keys_q16f8();
        let model =
            LogisticModel::zeros(3, quantizer, &client_key, ctx.clone()).expect("zero model");

        let temp_path = std::env::temp_dir().join(format!(
            "model_export_{}.json",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock drift")
                .as_nanos()
        ));

        write_model_export(
            &temp_path,
            &model,
            &quant,
            ctx.backend_name(),
            vec![1, 2, 3],
            vec![0],
        )
        .expect("export succeeds");

        let json = fs::read_to_string(&temp_path).expect("export file readable");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid json");

        assert_eq!(parsed["weights"], serde_json::json!([1, 2, 3]));
        assert_eq!(parsed["bias"], serde_json::json!([0]));
        assert_eq!(
            parsed["metadata"]["quantization"]["scale"],
            serde_json::json!(8)
        );
        assert_eq!(
            parsed["metadata"]["layers"][0]["rows"],
            serde_json::json!(3)
        );
        assert!(
            parsed["encrypted_tensors"]["weights"]["bytes"]
                .as_array()
                .expect("byte array")
                .len()
                > 0
        );

        let _ = fs::remove_file(temp_path);
    }
}
