use std::time::Instant;

use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128;

use crate::context::FheContext;
use crate::quantization::quantizer::Quantizer;
use crate::tensor::{FheTensorOps, TensorShape};

#[test]
fn noise_metadata_tracks_add_sub_and_scalar_mul_costs() {
    let (client_key, ctx) = FheContext::generate_keys_q16f8();
    let quantizer = Quantizer::new(ctx.quant_config.clone());
    let shape = TensorShape::from_2d(1, 1).unwrap();

    let lhs = quantizer
        .encrypt_quantized(
            &[ctx.quant_config.quantize(0.25)],
            shape.clone(),
            &client_key,
            ctx.clone(),
        )
        .unwrap();
    let rhs = quantizer
        .encrypt_quantized(&[ctx.quant_config.quantize(0.5)], shape, &client_key, ctx)
        .unwrap();

    let added = lhs.add(&rhs).unwrap();
    assert_eq!(added.noise_level().level, 1);

    let subtracted = added.sub(&rhs).unwrap();
    assert_eq!(subtracted.noise_level().level, 2);

    let scaled = subtracted.scalar_mul(3).unwrap();
    assert_eq!(scaled.noise_level().level, 4);
}

#[test]
fn truncate_resets_theoretical_noise_budget() {
    let (client_key, ctx) = FheContext::generate_keys_q16f8();
    let quantizer = Quantizer::new(ctx.quant_config.clone());
    let shape = TensorShape::from_2d(1, 1).unwrap();

    let lhs = quantizer
        .encrypt_quantized(
            &[ctx.quant_config.quantize(0.5)],
            shape.clone(),
            &client_key,
            ctx.clone(),
        )
        .unwrap();
    let rhs = quantizer
        .encrypt_quantized(&[ctx.quant_config.quantize(0.25)], shape, &client_key, ctx)
        .unwrap();

    let product = lhs.matmul(&rhs).unwrap();
    assert_eq!(product.noise_level().level, 4);

    let truncated = product
        .truncate(
            PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128
                .message_modulus
                .0
                .ilog2(),
        )
        .unwrap();
    assert_eq!(truncated.noise_level().level, 0);
}

#[test]
fn benchmark_single_30_feature_dot_product_latency() {
    let (client_key, ctx) = FheContext::generate_keys_q16f8();
    let quantizer = Quantizer::new(ctx.quant_config.clone());

    let lhs_values = (0..30)
        .map(|idx| ctx.quant_config.quantize((idx as f64 + 1.0) / 64.0))
        .collect::<Vec<_>>();
    let rhs_values = (0..30)
        .map(|idx| ctx.quant_config.quantize((idx as f64 + 2.0) / 96.0))
        .collect::<Vec<_>>();

    let lhs = quantizer
        .encrypt_quantized(
            &lhs_values,
            TensorShape::from_2d(1, 30).unwrap(),
            &client_key,
            ctx.clone(),
        )
        .unwrap();
    let rhs = quantizer
        .encrypt_quantized(
            &rhs_values,
            TensorShape::from_2d(30, 1).unwrap(),
            &client_key,
            ctx,
        )
        .unwrap();

    let started = Instant::now();
    let product = lhs.matmul(&rhs).unwrap();
    let truncated = product.truncate(8).unwrap();
    let elapsed = started.elapsed();

    assert_eq!(truncated.shape().dims(), &[1, 1]);
    println!("30-feature encrypted dot product latency: {:?}", elapsed);
}
