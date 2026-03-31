use crate::model::LogisticModel;
use crate::noise::{NoiseScheduler, SchedulerDecision};
use crate::tensor::{EncryptedTensor, FheOp, FheTensorOps, TensorError, DEFAULT_OP_SAFE_THRESHOLD};

#[derive(Clone, Debug)]
pub struct SgdOptimizer {
    pub learning_rate_q: i64,
    pub frac_bits: u32,
}

impl SgdOptimizer {
    pub fn step(
        &self,
        model: &mut LogisticModel,
        grad_w: &EncryptedTensor,
        grad_b: &EncryptedTensor,
        scheduler: &mut NoiseScheduler,
    ) -> Result<(), TensorError> {
        let lr_q = self.learning_rate_q as u64;

        let mut grad_w_owned = grad_w.clone();
        if scheduler.check_and_schedule("grad_w", &grad_w_owned, FheOp::ScalarMul(lr_q))
            == SchedulerDecision::BootstrapFirst
        {
            grad_w_owned = grad_w_owned.bootstrap()?;
            scheduler.record_bootstrap("grad_w");
        }
        let scaled_grad_w = grad_w_owned.scalar_mul(self.learning_rate_q)?;
        scheduler.record_op("grad_w", FheOp::ScalarMul(lr_q))?;
        let scaled_grad_w =
            refresh_binary_sub_rhs("scaled_grad_w", scaled_grad_w, &model.weights, scheduler)?;

        if scheduler.check_and_schedule("model.weights", &model.weights, FheOp::Sub)
            == SchedulerDecision::BootstrapFirst
        {
            model.weights = model.weights.bootstrap()?;
            scheduler.record_bootstrap("model.weights");
        }
        model.weights = model.weights.sub(&scaled_grad_w)?;
        model.weights = model.weights.force_refresh()?;
        scheduler.record_bootstrap("model.weights");
        scheduler.sync_tensor("model.weights", &model.weights);

        let mut grad_b_owned = grad_b.clone();
        if scheduler.check_and_schedule("grad_b", &grad_b_owned, FheOp::ScalarMul(lr_q))
            == SchedulerDecision::BootstrapFirst
        {
            grad_b_owned = grad_b_owned.bootstrap()?;
            scheduler.record_bootstrap("grad_b");
        }
        let lr_grad_b = grad_b_owned.scalar_mul(self.learning_rate_q)?;
        scheduler.record_op("grad_b", FheOp::ScalarMul(lr_q))?;
        let lr_grad_b = refresh_binary_sub_rhs("scaled_grad_b", lr_grad_b, &model.bias, scheduler)?;

        if scheduler.check_and_schedule("model.bias", &model.bias, FheOp::Sub)
            == SchedulerDecision::BootstrapFirst
        {
            model.bias = model.bias.bootstrap()?;
            scheduler.record_bootstrap("model.bias");
        }
        model.bias = model.bias.sub(&lr_grad_b)?;
        model.bias = model.bias.force_refresh()?;
        scheduler.record_bootstrap("model.bias");
        scheduler.sync_tensor("model.bias", &model.bias);
        Ok(())
    }
}

fn refresh_binary_sub_rhs(
    tensor_id: &str,
    tensor: EncryptedTensor,
    lhs: &EncryptedTensor,
    scheduler: &mut NoiseScheduler,
) -> Result<EncryptedTensor, TensorError> {
    let lhs_level = lhs.noise_level().level;
    let rhs_level = tensor.noise_level().level;
    let would_overflow = lhs_level.max(rhs_level).saturating_add(1) > lhs.noise_level().budget;

    if rhs_level >= DEFAULT_OP_SAFE_THRESHOLD || would_overflow {
        let refreshed = tensor.bootstrap()?;
        scheduler.record_bootstrap(tensor_id);
        Ok(refreshed)
    } else {
        Ok(tensor)
    }
}
