use crate::quantization::config::QuantConfig;
use crate::tensor::{FheTensorOps, PlaintextTensor, TensorShape};
use crate::tests::plaintext_mock::{q16f8_quantizer, tensor_from_f64};

#[test]
fn q16f8_round_trip_stays_close() {
    let config = QuantConfig::q16f8();
    let original = 0.375_f64;
    let quantized = config.quantize(original);
    let restored = config.dequantize(quantized);

    assert!((restored - original).abs() < 0.01);
    assert_eq!(quantized, 96);
    assert_eq!(config.total_bits, 16);
}

#[test]
fn truncate_after_multiply_restores_scale() {
    let quantizer = q16f8_quantizer();
    let left = quantizer.quantize_scalar(0.5);
    let right = quantizer.quantize_scalar(0.25);
    let product = PlaintextTensor::from_vec2(vec![vec![left * right]]).unwrap();
    let truncated = quantizer.truncate_after_mul(product).unwrap();

    assert_eq!(truncated.data()[0], quantizer.quantize_scalar(0.125));
}

#[test]
fn broadcasting_adds_scalar_bias_across_batch() {
    let logits = tensor_from_f64(&[vec![0.5], vec![0.25], vec![-0.25]]);
    let bias = tensor_from_f64(&[vec![0.125]]);
    let result = logits.add(&bias).unwrap();

    let expected = tensor_from_f64(&[vec![0.625], vec![0.375], vec![-0.125]]);
    assert_eq!(result, expected);
}

#[test]
fn broadcast_shape_matches_expected_dimensions() {
    let batch = TensorShape::from_2d(32, 1).unwrap();
    let bias = TensorShape::from_2d(1, 1).unwrap();
    let out = TensorShape::broadcast_shape(&batch, &bias).unwrap();

    assert_eq!(out.dims(), &[32, 1]);
}
