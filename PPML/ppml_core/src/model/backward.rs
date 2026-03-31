use crate::model::forward::LogisticModel;
use crate::noise::NoiseScheduler;
use crate::tensor::{EncryptedTensor, FheOp, FheTensorOps, TensorError, TensorShape};

impl LogisticModel {
    pub fn backward(
        &mut self,
        x: &EncryptedTensor,
        predictions: &EncryptedTensor,
        labels: &EncryptedTensor,
        scheduler: &mut NoiseScheduler,
    ) -> Result<(EncryptedTensor, EncryptedTensor), TensorError> {
        let predictions =
            super::forward::maybe_bootstrap("predictions", predictions, scheduler, FheOp::Sub)?;
        let labels = super::forward::maybe_bootstrap("labels", labels, scheduler, FheOp::Sub)?;
        let error = predictions.sub(&labels)?;
        scheduler.record_op("predictions", FheOp::Sub)?;
        scheduler.sync_tensor("error", &error);

        let x_t = x.transpose()?;
        scheduler.sync_tensor("x_transposed", &x_t);
        let x_t = super::forward::maybe_bootstrap("x_transposed", &x_t, scheduler, FheOp::CtCtMul)?;
        let error = super::forward::maybe_bootstrap("error", &error, scheduler, FheOp::CtCtMul)?;
        let grad_raw = x_t.matmul(&error)?;
        scheduler.record_op("x_transposed", FheOp::CtCtMul)?;
        scheduler.sync_tensor("grad_raw", &grad_raw);

        let grad_raw =
            super::forward::maybe_bootstrap("grad_raw", &grad_raw, scheduler, FheOp::Truncate)?;
        let grad_w = grad_raw.truncate_approximate(self.quantizer.config.frac_bits)?;
        scheduler.record_bootstrap("grad_raw");
        scheduler.sync_tensor("grad_w", &grad_w);

        let error_for_bias =
            super::forward::maybe_bootstrap("error", &error, scheduler, FheOp::Add)?;
        let grad_b = sum_axis0_with_scheduler(&error_for_bias, scheduler, "grad_b_acc")?;
        scheduler.sync_tensor("grad_b", &grad_b);
        Ok((grad_w, grad_b))
    }
}

fn sum_axis0_with_scheduler(
    tensor: &EncryptedTensor,
    scheduler: &mut NoiseScheduler,
    tensor_id: &str,
) -> Result<EncryptedTensor, TensorError> {
    if tensor.shape().dims().len() != 2 {
        return Err(TensorError::InvalidShape(
            "sum_axis0_with_scheduler requires a rank-2 tensor".to_string(),
        ));
    }

    let rows = tensor.shape().rows();
    let cols = tensor.shape().cols();
    let mut out = Vec::with_capacity(cols);

    for col in 0..cols {
        let first = EncryptedTensor::from_parts(
            vec![tensor.data()[col].clone()],
            TensorShape::from_2d(1, 1)?,
            tensor.noise_level(),
            tensor.ctx.clone(),
        )?;
        let mut accumulator = first;
        scheduler.sync_tensor(tensor_id, &accumulator);

        for row in 1..rows {
            accumulator =
                super::forward::maybe_bootstrap(tensor_id, &accumulator, scheduler, FheOp::Add)?;

            let element = EncryptedTensor::from_parts(
                vec![tensor.data()[row * cols + col].clone()],
                TensorShape::from_2d(1, 1)?,
                tensor.noise_level(),
                tensor.ctx.clone(),
            )?;

            accumulator = accumulator.add(&element)?;
            scheduler.record_op(tensor_id, FheOp::Add)?;
            scheduler.sync_tensor(tensor_id, &accumulator);
        }

        out.push(accumulator.data()[0].clone());
    }

    let grad_b_noise = scheduler
        .state
        .get(tensor_id)
        .cloned()
        .unwrap_or_else(|| tensor.noise_level());

    EncryptedTensor::from_parts(
        out,
        TensorShape::from_2d(1, cols)?,
        grad_b_noise,
        tensor.ctx.clone(),
    )
}
