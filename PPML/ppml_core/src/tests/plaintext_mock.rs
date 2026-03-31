use crate::quantization::{config::QuantConfig, quantizer::Quantizer};
use crate::tensor::PlaintextTensor;

pub fn q16f8_quantizer() -> Quantizer {
    Quantizer::new(QuantConfig::q16f8())
}

pub fn quantize_rows(rows: &[Vec<f64>]) -> Vec<Vec<i64>> {
    let quantizer = q16f8_quantizer();
    rows.iter()
        .map(|row| {
            row.iter()
                .map(|value| quantizer.quantize_scalar(*value))
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn tensor_from_f64(rows: &[Vec<f64>]) -> PlaintextTensor {
    PlaintextTensor::from_vec2(quantize_rows(rows)).expect("valid tensor fixture")
}
