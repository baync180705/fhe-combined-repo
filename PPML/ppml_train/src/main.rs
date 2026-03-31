use std::fs;
use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tfhe::integer::RadixClientKey;
use tracing::info;

use ppml_core::context::FheContext;
use ppml_core::export::write_model_export;
use ppml_core::model::LogisticModel;
use ppml_core::noise::NoiseScheduler;
use ppml_core::optimizer::SgdOptimizer;
use ppml_core::quantization::{config::QuantConfig, quantizer::Quantizer};
use ppml_core::tensor::{EncryptedTensor, FheTensorOps, TensorError, TensorShape};

const NUM_FEATURES: usize = 10;
const NUM_SAMPLES: usize = 50;
const EPOCHS: usize = 2;
const BATCH_SIZE: usize = 32;
const SCALE: f64 = 8.0;

fn main() -> Result<(), TensorError> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    let quant = QuantConfig::q16f8();
    let quantizer = Quantizer::new(quant.clone());
    let (client_key, ctx) = FheContext::generate_keys_q16f8();
    info!(backend = ctx.backend_name(), "initialized FHE backend");
    let (features, labels) = generate_synthetic_dataset(&quant)?;
    let mut model =
        LogisticModel::zeros(NUM_FEATURES, quantizer.clone(), &client_key, ctx.clone())?;
    let optimizer = SgdOptimizer {
        learning_rate_q: 3,
        frac_bits: quant.frac_bits,
    };
    let mut scheduler = NoiseScheduler::new(ctx.max_noise_budget);

    for epoch in 0..EPOCHS {
        info!(epoch = epoch + 1, total_epochs = EPOCHS, "starting epoch");
        let mut epoch_loss_sum = 0.0;
        let mut epoch_correct = 0usize;
        let mut epoch_examples = 0usize;
        let mut epoch_grad_w_l1 = 0.0;
        let mut epoch_grad_b_l1 = 0.0;

        for start in (0..NUM_SAMPLES).step_by(BATCH_SIZE) {
            let end = (start + BATCH_SIZE).min(NUM_SAMPLES);
            let batch_x = quantizer.encrypt_quantized(
                &features[start..end].concat(),
                TensorShape::from_2d(end - start, NUM_FEATURES)?,
                &client_key,
                ctx.clone(),
            )?;
            let batch_y = quantizer.encrypt_quantized(
                &labels[start..end].concat(),
                TensorShape::from_2d(end - start, 1)?,
                &client_key,
                ctx.clone(),
            )?;

            let predictions = model.forward(&batch_x, &mut scheduler)?;
            let (grad_w, grad_b) =
                model.backward(&batch_x, &predictions, &batch_y, &mut scheduler)?;
            let batch_metrics = evaluate_batch(&predictions, &batch_y, &quantizer, &client_key);
            let grad_w_l1 = tensor_mean_abs(&grad_w, &quantizer, &client_key);
            let grad_b_l1 = tensor_mean_abs(&grad_b, &quantizer, &client_key);
            optimizer.step(&mut model, &grad_w, &grad_b, &mut scheduler)?;

            epoch_loss_sum += batch_metrics.loss_sum;
            epoch_correct += batch_metrics.correct;
            epoch_examples += batch_metrics.samples;
            epoch_grad_w_l1 += grad_w_l1;
            epoch_grad_b_l1 += grad_b_l1;

            info!(
                epoch = epoch + 1,
                batch_start = start,
                batch_end = end,
                batch_loss = format_args!("{:.6}", batch_metrics.mean_loss()),
                batch_accuracy = format_args!("{:.2}%", batch_metrics.accuracy() * 100.0),
                grad_w_l1 = format_args!("{:.6}", grad_w_l1),
                grad_b_l1 = format_args!("{:.6}", grad_b_l1),
                weight_noise = model.weights.noise_level().level,
                bias_noise = model.bias.noise_level().level,
                "batch update"
            );
        }

        let weights = decrypt_parameter_vector(&model.weights, &quantizer, &client_key);
        let bias = decrypt_parameter_vector(&model.bias, &quantizer, &client_key);
        let weights_f = dequantize_values(&weights);
        let bias_f = dequantize_values(&bias);
        let accuracy = plaintext_accuracy(&features, &labels, &weights_f, &bias_f);

        info!(
            epoch = epoch + 1,
            train_loss = format_args!("{:.6}", epoch_loss_sum / epoch_examples as f64),
            train_accuracy = format_args!(
                "{:.2}%",
                (epoch_correct as f64 / epoch_examples as f64) * 100.0
            ),
            plaintext_accuracy = format_args!("{:.2}%", accuracy),
            mean_grad_w_l1 = format_args!("{:.6}", epoch_grad_w_l1 / batches_per_epoch() as f64),
            mean_grad_b_l1 = format_args!("{:.6}", epoch_grad_b_l1 / batches_per_epoch() as f64),
            "epoch summary"
        );

        println!(
            "[debug][epoch {}] Quantized Weights: {:?}",
            epoch + 1,
            weights
        );
        println!("[debug][epoch {}] Quantized Bias: {:?}", epoch + 1, bias);
        println!("[Epoch {}] Accuracy: {:.1}%", epoch + 1, accuracy);
    }

    let final_quantized_weights = decrypt_parameter_vector(&model.weights, &quantizer, &client_key);
    let final_quantized_bias = decrypt_parameter_vector(&model.bias, &quantizer, &client_key);

    fs::write("fhenix_weights.bin", model.weights.to_bytes()?).map_err(io_to_tensor_error)?;
    fs::write("fhenix_bias.bin", model.bias.to_bytes()?).map_err(io_to_tensor_error)?;
    write_model_export(
        &ppml_root().join("model_export.json"),
        &model,
        &quant,
        ctx.backend_name(),
        final_quantized_weights,
        final_quantized_bias,
    )?;
    println!("Encrypted model successfully exported for deployment!");

    info!("phase 3 encrypted training loop completed");
    Ok(())
}

fn batches_per_epoch() -> usize {
    NUM_SAMPLES.div_ceil(BATCH_SIZE)
}

#[derive(Clone, Copy, Debug, Default)]
struct BatchMetrics {
    loss_sum: f64,
    correct: usize,
    samples: usize,
}

impl BatchMetrics {
    fn mean_loss(self) -> f64 {
        self.loss_sum / self.samples as f64
    }

    fn accuracy(self) -> f64 {
        self.correct as f64 / self.samples as f64
    }
}

fn evaluate_batch(
    predictions: &EncryptedTensor,
    labels: &EncryptedTensor,
    quantizer: &Quantizer,
    client_key: &RadixClientKey,
) -> BatchMetrics {
    let mut metrics = BatchMetrics::default();

    for (prediction, label) in predictions.data().iter().zip(labels.data().iter()) {
        let pred_q = quantizer.decrypt_quantized(prediction, client_key);
        let label_q = quantizer.decrypt_quantized(label, client_key);
        let pred = quantizer.dequantize_scalar(pred_q).clamp(0.0, 1.0);
        let target = quantizer.dequantize_scalar(label_q).clamp(0.0, 1.0);
        let clipped_pred = pred.clamp(1e-6, 1.0 - 1e-6);
        let predicted_label = usize::from(pred >= 0.5);
        let actual_label = usize::from(target >= 0.5);

        metrics.loss_sum +=
            -(target * clipped_pred.ln() + (1.0 - target) * (1.0 - clipped_pred).ln());
        metrics.correct += usize::from(predicted_label == actual_label);
        metrics.samples += 1;
    }

    metrics
}

fn tensor_mean_abs(
    tensor: &EncryptedTensor,
    quantizer: &Quantizer,
    client_key: &RadixClientKey,
) -> f64 {
    let sum = tensor
        .data()
        .iter()
        .map(|ct| quantizer.decrypt_quantized(ct, client_key))
        .map(|value| quantizer.dequantize_scalar(value).abs())
        .sum::<f64>();

    sum / tensor.data().len() as f64
}

fn evaluate_dataset(
    features: &[Vec<i64>],
    labels: &[Vec<i64>],
    quantizer: &Quantizer,
    client_key: &RadixClientKey,
    ctx: &std::sync::Arc<FheContext>,
    model: &mut LogisticModel,
    scheduler: &mut NoiseScheduler,
) -> Result<BatchMetrics, TensorError> {
    let all_x = quantizer.encrypt_quantized(
        &features.concat(),
        TensorShape::from_2d(NUM_SAMPLES, NUM_FEATURES)?,
        client_key,
        ctx.clone(),
    )?;
    let all_y = quantizer.encrypt_quantized(
        &labels.concat(),
        TensorShape::from_2d(NUM_SAMPLES, 1)?,
        client_key,
        ctx.clone(),
    )?;
    let predictions = model.forward(&all_x, scheduler)?;
    Ok(evaluate_batch(&predictions, &all_y, quantizer, client_key))
}

fn decrypt_parameter_vector(
    tensor: &EncryptedTensor,
    quantizer: &Quantizer,
    client_key: &RadixClientKey,
) -> Vec<i64> {
    tensor
        .data()
        .iter()
        .map(|ct| quantizer.decrypt_quantized(ct, client_key))
        .collect()
}

fn dequantize_values(values: &[i64]) -> Vec<f64> {
    values.iter().map(|value| *value as f64 / SCALE).collect()
}

fn plaintext_accuracy(
    features: &[Vec<i64>],
    labels: &[Vec<i64>],
    weights: &[f64],
    bias: &[f64],
) -> f64 {
    let bias = bias.first().copied().unwrap_or(0.0);
    let correct = features
        .iter()
        .zip(labels.iter())
        .filter(|(sample, label)| {
            let logit = sample
                .iter()
                .zip(weights.iter())
                .fold(bias, |acc, (feature, weight)| {
                    acc + (*feature as f64 / SCALE) * *weight
                });
            let prediction = 1.0 / (1.0 + (-logit).exp());
            let predicted_label = usize::from(prediction >= 0.5);
            let actual_label = usize::from(label[0] > 0);
            predicted_label == actual_label
        })
        .count();

    (correct as f64 / features.len() as f64) * 100.0
}

fn io_to_tensor_error(error: std::io::Error) -> TensorError {
    TensorError::Io(format!("while exporting encrypted model: {error}"))
}

fn ppml_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ppml_train lives under the PPML workspace root")
        .to_path_buf()
}

fn generate_synthetic_dataset(
    quant: &QuantConfig,
) -> Result<(Vec<Vec<i64>>, Vec<Vec<i64>>), TensorError> {
    let mut rng = StdRng::seed_from_u64(7);
    let true_weights: [f64; NUM_FEATURES] = [2.4, -1.7, 1.9, 0.0, 0.0, 1.5, -1.1, 0.0, 0.8, -0.6];
    let bias = -0.15;

    let mut features = Vec::with_capacity(NUM_SAMPLES);
    let mut labels = Vec::with_capacity(NUM_SAMPLES);

    for _ in 0..NUM_SAMPLES {
        let mut sample_f = Vec::with_capacity(NUM_FEATURES);
        for _ in 0..NUM_FEATURES {
            sample_f.push(rng.gen_range(-1.0..=1.0));
        }

        let mut logit = bias;
        for (feature, weight) in sample_f.iter().zip(true_weights.iter()) {
            logit += feature * weight;
        }

        let noise =
            ((rng.gen_range(-1.0..=1.0) + rng.gen_range(-1.0..=1.0) + rng.gen_range(-1.0..=1.0))
                / 3.0)
                * 0.35;
        let label = if logit + noise > 0.0 { 1.0 } else { 0.0 };

        let quantized_sample = sample_f
            .into_iter()
            .map(|value| quant.quantize(value.clamp(-1.0, 1.0)))
            .collect::<Vec<_>>();
        features.push(quantized_sample);
        labels.push(vec![quant.quantize(label)]);
    }

    Ok((features, labels))
}
