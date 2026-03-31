use crate::model::LogisticModel;
use crate::optimizer::SgdOptimizer;
use crate::tensor::{FheTensorOps, PlaintextTensor};
use crate::tests::plaintext_mock::{q16f8_quantizer, tensor_from_f64};

#[test]
fn forward_outputs_probabilities_in_q16f8_range() {
    let quantizer = q16f8_quantizer();
    let mut model = LogisticModel::zeros(2, quantizer.clone()).unwrap();
    model.weights = tensor_from_f64(&[vec![0.75], vec![-0.5]]);
    model.bias = tensor_from_f64(&[vec![0.1]]);

    let x = tensor_from_f64(&[vec![0.5, 0.25], vec![-0.5, 0.75]]);
    let predictions = model.forward(&x).unwrap();

    for value in predictions.data() {
        assert!(*value >= 0);
        assert!(*value <= quantizer.config.scale);
    }
}

#[test]
fn backward_returns_expected_gradient_shapes() {
    let quantizer = q16f8_quantizer();
    let model = LogisticModel::zeros(3, quantizer).unwrap();
    let x = tensor_from_f64(&[vec![1.0, 0.0, -1.0], vec![0.5, 0.5, 0.5]]);
    let predictions = tensor_from_f64(&[vec![0.75], vec![0.25]]);
    let labels = tensor_from_f64(&[vec![1.0], vec![0.0]]);

    let (grad_w, grad_b) = model.backward(&x, &predictions, &labels).unwrap();

    assert_eq!(grad_w.shape().dims(), &[3, 1]);
    assert_eq!(grad_b.shape().dims(), &[1, 1]);
}

#[test]
fn optimizer_step_reduces_batch_loss_for_simple_problem() {
    let quantizer = q16f8_quantizer();
    let mut model = LogisticModel::zeros(2, quantizer.clone()).unwrap();
    let optimizer = SgdOptimizer {
        learning_rate_q: quantizer.quantize_scalar(0.2),
        frac_bits: quantizer.config.frac_bits,
    };

    let x = tensor_from_f64(&[
        vec![0.9, 0.8],
        vec![0.7, 0.6],
        vec![-0.8, -0.9],
        vec![-0.6, -0.7],
    ]);
    let labels = tensor_from_f64(&[vec![1.0], vec![1.0], vec![0.0], vec![0.0]]);

    let before = mse_loss(&model.forward(&x).unwrap(), &labels, quantizer.config.scale);
    for _ in 0..30 {
        let predictions = model.forward(&x).unwrap();
        let (grad_w, grad_b) = model.backward(&x, &predictions, &labels).unwrap();
        optimizer.step(&mut model, &grad_w, &grad_b).unwrap();
    }
    let after = mse_loss(&model.forward(&x).unwrap(), &labels, quantizer.config.scale);

    assert!(
        after < before,
        "expected loss to decrease, before={before}, after={after}"
    );
}

fn mse_loss(predictions: &PlaintextTensor, labels: &PlaintextTensor, scale: i64) -> i64 {
    predictions
        .data()
        .iter()
        .zip(labels.data().iter())
        .map(|(pred, label)| {
            let diff = pred - label;
            (diff * diff) / scale
        })
        .sum::<i64>()
        / predictions.data().len() as i64
}
