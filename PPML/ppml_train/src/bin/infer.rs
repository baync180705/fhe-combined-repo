use std::env;
use std::fs;

use ppml_core::context::FheContext;
use ppml_core::quantization::{config::QuantConfig, quantizer::Quantizer};
use ppml_core::tensor::{EncryptedTensor, FheTensorOps, TensorError, TensorShape};
use tracing::info;

fn main() -> Result<(), TensorError> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();

    // Single-point inference intentionally uses the default CPU ServerKey build.
    // For a one-vector dot product, bypassing CUDA avoids GPU launch and transfer overhead.

    let features = parse_features(env::args().skip(1))?;
    let quant = QuantConfig::q16f8();
    let quantizer = Quantizer::new(quant.clone());
    let (client_key, ctx) = FheContext::generate_keys_q16f8();
    info!(
        backend = ctx.backend_name(),
        "initialized inference backend"
    );

    let weights_bytes = fs::read("fhenix_weights.bin")
        .map_err(|error| TensorError::Io(format!("failed to read weights export: {error}")))?;
    let bias_bytes = fs::read("fhenix_bias.bin")
        .map_err(|error| TensorError::Io(format!("failed to read bias export: {error}")))?;

    let weights = EncryptedTensor::from_bytes(&weights_bytes, ctx.clone())?;
    let bias = EncryptedTensor::from_bytes(&bias_bytes, ctx.clone())?;

    if weights.shape().rows() != features.len() {
        return Err(TensorError::InvalidShape(format!(
            "expected {} features based on exported weights, received {}",
            weights.shape().rows(),
            features.len()
        )));
    }

    let input = quantizer.encrypt_quantized(
        &features,
        TensorShape::from_2d(1, features.len())?,
        &client_key,
        ctx,
    )?;

    let logit = input
        .matmul(&weights)?
        .truncate(quant.frac_bits)?
        .add(&bias)?;

    let logit_q = quantizer.decrypt_quantized(&logit.data()[0], &client_key);
    let logit_f = quantizer.dequantize_scalar(logit_q);
    let predicted_positive = logit_f > 0.0;

    println!("Encrypted CPU inference completed.");
    println!("Logit: {:.6}", logit_f);
    println!("Predicted positive class: {}", predicted_positive);

    Ok(())
}

fn parse_features(args: impl Iterator<Item = String>) -> Result<Vec<i64>, TensorError> {
    let mut parsed = Vec::new();
    for value in args {
        let float_value = value.parse::<f64>().map_err(|error| {
            TensorError::InvalidShape(format!("invalid feature value '{value}': {error}"))
        })?;
        parsed.push(QuantConfig::q16f8().quantize(float_value));
    }

    if parsed.is_empty() {
        return Err(TensorError::InvalidShape(
            "provide at least one feature value, e.g. cargo run -p ppml_train --bin infer -- 0.45 0.52"
                .to_string(),
        ));
    }

    Ok(parsed)
}
