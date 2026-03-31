use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use ppml_core::context::FheContext;
use ppml_core::export::write_model_export;
use ppml_core::model::LogisticModel;
use ppml_core::noise::NoiseScheduler;
use ppml_core::optimizer::SgdOptimizer;
use ppml_core::quantization::{config::QuantConfig, quantizer::Quantizer};
use ppml_core::tensor::{EncryptedTensor, FheTensorOps, TensorError, TensorShape};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tfhe::integer::RadixClientKey;

const DEFAULT_SCALE: f64 = 8.0;
const FEATURE_CLIP_MIN: f64 = -2.0;
const FEATURE_CLIP_MAX: f64 = 2.0;

#[pyclass(unsendable)]
struct BlindContext {
    client_key: RadixClientKey,
    fhe_context: Arc<FheContext>,
}

#[pymethods]
impl BlindContext {
    #[staticmethod]
    fn generate() -> Self {
        let cache_path = PathBuf::from("fhe_keys.bin");
        let (client_key, fhe_context) = FheContext::load_or_generate(&cache_path)
            .unwrap_or_else(|_| FheContext::generate_keys_q16f8());
        println!(
            "BlindContext initialized with {} backend",
            fhe_context.backend_name()
        );
        let _ = io::stdout().flush();
        Self {
            client_key,
            fhe_context,
        }
    }
}

#[pyclass(unsendable)]
struct BlindLogisticRegression {
    input_features: usize,
    model: Option<LogisticModel>,
    quantizer: Quantizer,
    last_quantized_weights: Vec<i64>,
    last_quantized_bias: Vec<i64>,
    backend_name: String,
}

#[pymethods]
impl BlindLogisticRegression {
    #[new]
    fn new(input_features: usize) -> PyResult<Self> {
        if input_features == 0 {
            return Err(PyValueError::new_err(
                "input_features must be greater than zero",
            ));
        }

        Ok(Self {
            input_features,
            model: None,
            quantizer: Quantizer::new(QuantConfig::q16f8()),
            last_quantized_weights: Vec::new(),
            last_quantized_bias: Vec::new(),
            backend_name: "unknown".to_string(),
        })
    }

    fn fit(
        &mut self,
        context: PyRef<'_, BlindContext>,
        x: Vec<Vec<f32>>,
        y: Vec<f32>,
        epochs: usize,
        batch_size: usize,
        learning_rate: f32,
    ) -> PyResult<()> {
        validate_training_inputs(
            self.input_features,
            &x,
            &y,
            epochs,
            batch_size,
            learning_rate,
        )?;

        let quant = QuantConfig::q16f8();
        self.quantizer = Quantizer::new(quant.clone());
        self.backend_name = context.fhe_context.backend_name().to_string();

        let (quantized_x, clipping_stats) = quantize_features(&x, &quant);
        let quantized_y = quantize_labels(&y, &quant);
        let label_positive_rate =
            (y.iter().filter(|value| **value >= 0.5).count() as f64 / y.len() as f64) * 100.0;

        println!(
            "   Effective quantized learning rate: {:.3} (q={})",
            quant.dequantize(quant.quantize(learning_rate as f64)),
            quant.quantize(learning_rate as f64),
        );
        println!(
            "   Feature clipping stats: low_clipped={} high_clipped={}",
            clipping_stats.low_clipped, clipping_stats.high_clipped,
        );
        println!(
            "   Training label positive rate: {:.1}% (a constant predictor will match this baseline if it always predicts 1)",
            label_positive_rate,
        );
        let _ = io::stdout().flush();

        if self.model.is_none() {
            self.model = Some(
                LogisticModel::zeros(
                    self.input_features,
                    self.quantizer.clone(),
                    &context.client_key,
                    Arc::clone(&context.fhe_context),
                )
                .map_err(tensor_to_pyerr)?,
            );
        }

        let model = self
            .model
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("model was not initialized"))?;

        let mut scheduler = NoiseScheduler::new(context.fhe_context.max_noise_budget);
        let optimizer = SgdOptimizer {
            learning_rate_q: quant.quantize(learning_rate as f64),
            frac_bits: quant.frac_bits,
        };

        for epoch in 0..epochs {
            let mut epoch_loss_sum = 0.0;
            let mut epoch_correct = 0usize;
            let mut epoch_examples = 0usize;
            let mut batch_index = 0usize;
            let batch_total = quantized_x.len().div_ceil(batch_size);
            let epoch_order = build_epoch_order(quantized_x.len(), epoch);

            for start in (0..epoch_order.len()).step_by(batch_size) {
                let end = (start + batch_size).min(epoch_order.len());
                let batch_indices = &epoch_order[start..end];
                let batch_x_values = batch_indices
                    .iter()
                    .flat_map(|idx| quantized_x[*idx].iter().copied())
                    .collect::<Vec<_>>();
                let batch_y_values = batch_indices
                    .iter()
                    .flat_map(|idx| quantized_y[*idx].iter().copied())
                    .collect::<Vec<_>>();
                let batch_x = self
                    .quantizer
                    .encrypt_quantized(
                        &batch_x_values,
                        TensorShape::from_2d(batch_indices.len(), self.input_features)
                            .map_err(tensor_to_pyerr)?,
                        &context.client_key,
                        Arc::clone(&context.fhe_context),
                    )
                    .map_err(tensor_to_pyerr)?;
                let batch_y = self
                    .quantizer
                    .encrypt_quantized(
                        &batch_y_values,
                        TensorShape::from_2d(batch_indices.len(), 1).map_err(tensor_to_pyerr)?,
                        &context.client_key,
                        Arc::clone(&context.fhe_context),
                    )
                    .map_err(tensor_to_pyerr)?;

                let predictions = model
                    .forward(&batch_x, &mut scheduler)
                    .map_err(tensor_to_pyerr)?;
                let (grad_w, grad_b) = model
                    .backward(&batch_x, &predictions, &batch_y, &mut scheduler)
                    .map_err(tensor_to_pyerr)?;
                let batch_metrics =
                    evaluate_batch(&predictions, &batch_y, &self.quantizer, &context.client_key);

                optimizer
                    .step(model, &grad_w, &grad_b, &mut scheduler)
                    .map_err(tensor_to_pyerr)?;

                epoch_loss_sum += batch_metrics.loss_sum;
                epoch_correct += batch_metrics.correct;
                epoch_examples += batch_metrics.samples;
                batch_index += 1;

                println!(
                    "  [debug][epoch {}/{}][batch {}/{}][samples {}..{}] loss={:.6} train_acc={:.1}% weight_noise={} bias_noise={}",
                    epoch + 1,
                    epochs,
                    batch_index,
                    batch_total,
                    start,
                    end,
                    batch_metrics.mean_loss(),
                    batch_metrics.accuracy() * 100.0,
                    model.weights.noise_level().level,
                    model.bias.noise_level().level,
                );
                let _ = io::stdout().flush();
            }

            let weights =
                decrypt_parameter_vector(&model.weights, &self.quantizer, &context.client_key);
            let bias = decrypt_parameter_vector(&model.bias, &self.quantizer, &context.client_key);
            let weights_f = dequantize_values(&weights);
            let bias_f = dequantize_values(&bias);
            let accuracy = plaintext_accuracy(&quantized_x, &quantized_y, &weights_f, &bias_f);
            let mean_loss = epoch_loss_sum / epoch_examples as f64;
            let train_accuracy = (epoch_correct as f64 / epoch_examples as f64) * 100.0;

            println!(
                "[Epoch {}] Loss: {:.6} | Online Batch Accuracy: {:.1}% | Plaintext Accuracy: {:.1}%",
                epoch + 1,
                mean_loss,
                train_accuracy,
                accuracy
            );
            self.last_quantized_weights = weights;
            self.last_quantized_bias = bias;
            let _ = io::stdout().flush();
        }

        Ok(())
    }

    fn export_model(&self, path: String) -> PyResult<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model is not trained yet"))?;

        let export_dir = PathBuf::from(path);
        fs::create_dir_all(&export_dir).map_err(|err| {
            PyRuntimeError::new_err(format!("failed to create export dir: {err}"))
        })?;

        let weights_path = export_dir.join("fhenix_weights.bin");
        let bias_path = export_dir.join("fhenix_bias.bin");

        fs::write(
            &weights_path,
            model.weights.to_bytes().map_err(tensor_to_pyerr)?,
        )
        .map_err(io_to_pyerr)?;
        fs::write(&bias_path, model.bias.to_bytes().map_err(tensor_to_pyerr)?)
            .map_err(io_to_pyerr)?;
        write_model_export(
            &ppml_root().join("model_export.json"),
            model,
            &self.quantizer.config,
            &self.backend_name,
            self.last_quantized_weights.clone(),
            self.last_quantized_bias.clone(),
        )
        .map_err(tensor_to_pyerr)?;

        println!("Encrypted model successfully exported for deployment!");
        let _ = io::stdout().flush();
        Ok(())
    }
}

#[pymodule]
fn blindml(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<BlindContext>()?;
    module.add_class::<BlindLogisticRegression>()?;
    Ok(())
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

fn validate_training_inputs(
    input_features: usize,
    x: &[Vec<f32>],
    y: &[f32],
    epochs: usize,
    batch_size: usize,
    learning_rate: f32,
) -> PyResult<()> {
    if x.is_empty() {
        return Err(PyValueError::new_err("x must not be empty"));
    }
    if x.len() != y.len() {
        return Err(PyValueError::new_err(format!(
            "x/y length mismatch: {} samples vs {} labels",
            x.len(),
            y.len()
        )));
    }
    if epochs == 0 {
        return Err(PyValueError::new_err("epochs must be greater than zero"));
    }
    if batch_size == 0 {
        return Err(PyValueError::new_err(
            "batch_size must be greater than zero",
        ));
    }
    if learning_rate <= 0.0 {
        return Err(PyValueError::new_err("learning_rate must be positive"));
    }
    if x.iter().any(|row| row.len() != input_features) {
        return Err(PyValueError::new_err(format!(
            "each input row must contain exactly {input_features} features"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
struct ClippingStats {
    low_clipped: usize,
    high_clipped: usize,
}

fn quantize_features(x: &[Vec<f32>], quant: &QuantConfig) -> (Vec<Vec<i64>>, ClippingStats) {
    let mut stats = ClippingStats::default();
    let quantized = x
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| {
                    let value = *value as f64;
                    let clipped = value.clamp(FEATURE_CLIP_MIN, FEATURE_CLIP_MAX);
                    if clipped == FEATURE_CLIP_MIN && value < FEATURE_CLIP_MIN {
                        stats.low_clipped += 1;
                    }
                    if clipped == FEATURE_CLIP_MAX && value > FEATURE_CLIP_MAX {
                        stats.high_clipped += 1;
                    }
                    quant.quantize(clipped)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    (quantized, stats)
}

fn quantize_labels(y: &[f32], quant: &QuantConfig) -> Vec<Vec<i64>> {
    y.iter()
        .map(|value| vec![quant.quantize((*value).clamp(0.0, 1.0) as f64)])
        .collect()
}

fn build_epoch_order(len: usize, epoch: usize) -> Vec<usize> {
    let mut order = (0..len).collect::<Vec<_>>();
    let salt = epoch.wrapping_mul(0x9e37_79b9);
    order.sort_by_key(|idx| idx.wrapping_mul(1_103_515_245).wrapping_add(salt));
    order
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
    values
        .iter()
        .map(|value| *value as f64 / DEFAULT_SCALE)
        .collect()
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
                    acc + (*feature as f64 / DEFAULT_SCALE) * *weight
                });
            let prediction = 1.0 / (1.0 + (-logit).exp());
            let predicted_label = usize::from(prediction >= 0.5);
            let actual_label = usize::from(label[0] > 0);
            predicted_label == actual_label
        })
        .count();

    (correct as f64 / features.len() as f64) * 100.0
}

fn tensor_to_pyerr(error: TensorError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn io_to_pyerr(error: std::io::Error) -> PyErr {
    PyRuntimeError::new_err(format!("failed to export encrypted model: {error}"))
}

fn ppml_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("blindml lives under the PPML workspace root")
        .to_path_buf()
}
