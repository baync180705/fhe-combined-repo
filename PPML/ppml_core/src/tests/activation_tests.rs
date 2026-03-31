use crate::activation::LutEngine;
use crate::context::FheContext;
use crate::quantization::quantizer::Quantizer;
use crate::tensor::{FheTensorOps, TensorShape};

#[test]
fn sigmoid_lut_matches_float_sigmoid_with_small_error() {
    let (client_key, ctx) = FheContext::generate_keys_q16f8();
    let quantizer = Quantizer::new(ctx.quant_config.clone());
    let lut =
        LutEngine::from_client_key(&client_key, ctx.wopbs_key.clone(), ctx.quant_config.clone());

    let input = 0.75_f64;
    let encrypted = quantizer
        .encrypt_quantized(
            &[quantizer.quantize_scalar(input)],
            TensorShape::from_2d(1, 1).unwrap(),
            &client_key,
            ctx,
        )
        .unwrap();

    let result_ct = lut.sigmoid(&encrypted.data()[0]);
    let decrypted_q: u64 = client_key.decrypt(&result_ct);
    let decrypted_float = (decrypted_q as i16 as f64) / 256.0;
    let expected_float = 0.679178;

    assert!(
        (decrypted_float - expected_float).abs() < 0.05,
        "output float={}, expected float={}",
        decrypted_float,
        expected_float
    );
}

#[test]
fn fused_truncate_and_sigmoid_matches_manual_path() {
    let (client_key, ctx) = FheContext::generate_keys_q16f8();
    let quantizer = Quantizer::new(ctx.quant_config.clone());
    let lut =
        LutEngine::from_client_key(&client_key, ctx.wopbs_key.clone(), ctx.quant_config.clone());

    let wide_value = quantizer.quantize_scalar(0.5) * quantizer.quantize_scalar(0.25);
    let encrypted = quantizer
        .encrypt_quantized(
            &[wide_value],
            TensorShape::from_2d(1, 1).unwrap(),
            &client_key,
            ctx,
        )
        .unwrap();

    let manual_truncated = encrypted.truncate(quantizer.config.frac_bits).unwrap();
    let manual_sigmoid = lut.sigmoid(&manual_truncated.data()[0]);
    let fused_sigmoid = lut.fused_truncate_and_sigmoid(&encrypted.data()[0]);

    let manual_q = quantizer.decrypt_quantized(&manual_sigmoid, &client_key);
    let fused_q = quantizer.decrypt_quantized(&fused_sigmoid, &client_key);

    assert_eq!(manual_q, fused_q);
}
