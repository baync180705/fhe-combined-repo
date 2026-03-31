use std::sync::Arc;

use rand::seq::SliceRandom;
use rand::thread_rng;
use tracing::debug;

use crate::activation::LutEngine;
use crate::context::FheContext;
use crate::noise::{NoiseScheduler, SchedulerDecision};
use crate::quantization::quantizer::Quantizer;
use crate::tensor::{EncryptedTensor, FheOp, FheTensorOps, TensorError};

#[derive(Clone)]
pub struct LogisticModel {
    pub weights: EncryptedTensor,
    pub bias: EncryptedTensor,
    pub lut: Arc<LutEngine>,
    pub quantizer: Quantizer,
}

impl LogisticModel {
    pub fn zeros(
        num_features: usize,
        quantizer: Quantizer,
        client_key: &tfhe::integer::RadixClientKey,
        ctx: Arc<FheContext>,
    ) -> Result<Self, TensorError> {
        let discrete_weight_values = [-4_i64, -3, -2, -1, 1, 2, 3, 4];
        let mut rng = thread_rng();
        let initial_weights = (0..num_features)
            .map(|_| {
                *discrete_weight_values
                    .choose(&mut rng)
                    .expect("non-empty init support")
            })
            .collect::<Vec<_>>();
        let weights = quantizer.encrypt_quantized(
            &initial_weights,
            crate::tensor::TensorShape::from_2d(num_features, 1)?,
            client_key,
            Arc::clone(&ctx),
        )?;
        let bias = quantizer.encrypt_quantized(
            &[0_i64],
            crate::tensor::TensorShape::from_2d(1, 1)?,
            client_key,
            Arc::clone(&ctx),
        )?;
        Ok(Self {
            weights,
            bias,
            lut: Arc::new(LutEngine::from_client_key(
                client_key,
                ctx.wopbs_key.clone(),
                ctx.quant_config.clone(),
            )),
            quantizer,
        })
    }

    pub fn forward(
        &mut self,
        x: &EncryptedTensor,
        scheduler: &mut NoiseScheduler,
    ) -> Result<EncryptedTensor, TensorError> {
        let scale_shift = 1_i64 << self.quantizer.config.frac_bits;
        self.weights = maybe_bootstrap("weights", &self.weights, scheduler, FheOp::CtCtMul)?;
        scheduler.sync_tensor("x_batch", x);
        debug!("forward: encrypted matmul");
        let logit_raw = x.matmul(&self.weights)?;
        scheduler.record_op("weights", FheOp::CtCtMul)?;
        scheduler.sync_tensor("logit_raw", &logit_raw);

        self.bias = maybe_bootstrap("bias", &self.bias, scheduler, FheOp::Add)?;
        let bias_scaled = self.bias.scalar_mul(scale_shift)?;
        debug!("forward: add bias");
        let biased = logit_raw.add(&bias_scaled)?;
        scheduler.sync_tensor("logit_biased", &biased);

        let biased = maybe_bootstrap("logit_biased", &biased, scheduler, FheOp::Bootstrap)?;
        debug!("forward: fused truncate + sigmoid LUT");
        let activations = EncryptedTensor::from_parts(
            biased
                .data()
                .iter()
                .map(|ct| self.lut.truncate_then_sigmoid(ct))
                .collect(),
            biased.shape().clone(),
            crate::tensor::NoiseMetadata::fresh(biased.noise_level().budget),
            Arc::clone(&biased.ctx),
        )?;
        scheduler.record_bootstrap("logit_biased");
        Ok(activations)
    }
}

impl std::fmt::Debug for LogisticModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogisticModel")
            .field("weights_shape", &self.weights.shape())
            .field("bias_shape", &self.bias.shape())
            .field("quantizer", &self.quantizer)
            .finish_non_exhaustive()
    }
}

pub(crate) fn maybe_bootstrap(
    tensor_id: &str,
    tensor: &EncryptedTensor,
    scheduler: &mut NoiseScheduler,
    upcoming_op: FheOp,
) -> Result<EncryptedTensor, TensorError> {
    match scheduler.check_and_schedule(tensor_id, tensor, upcoming_op) {
        SchedulerDecision::ProceedDirect => Ok(tensor.clone()),
        SchedulerDecision::BootstrapFirst => {
            let bootstrapped = tensor.bootstrap()?;
            scheduler.record_bootstrap(tensor_id);
            Ok(bootstrapped)
        }
    }
}
