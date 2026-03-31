use std::collections::HashMap;

use tracing::debug;

use crate::tensor::{
    EncryptedTensor, FheOp, FheTensorOps, NoiseMetadata, TensorError, DEFAULT_OP_SAFE_THRESHOLD,
    HEAVY_OP_COST, HEAVY_OP_SAFE_THRESHOLD,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulerDecision {
    ProceedDirect,
    BootstrapFirst,
}

#[derive(Clone, Debug)]
pub struct NoiseScheduler {
    pub state: HashMap<String, NoiseMetadata>,
    pub budget: u32,
}

impl NoiseScheduler {
    pub fn new(budget: u32) -> Self {
        Self {
            state: HashMap::new(),
            budget,
        }
    }

    pub fn sync_tensor(&mut self, tensor_id: &str, tensor: &EncryptedTensor) {
        self.state
            .insert(tensor_id.to_string(), tensor.noise_level());
    }

    pub fn check_and_schedule(
        &mut self,
        tensor_id: &str,
        tensor: &EncryptedTensor,
        upcoming_op: FheOp,
    ) -> SchedulerDecision {
        self.sync_tensor(tensor_id, tensor);
        let noise = self
            .state
            .entry(tensor_id.to_string())
            .or_insert_with(|| NoiseMetadata::fresh(self.budget));
        let op_cost = op_cost(upcoming_op);
        let threshold = safe_threshold(upcoming_op, self.budget);
        let decision =
            if noise.level >= threshold || noise.level.saturating_add(op_cost) > self.budget {
                SchedulerDecision::BootstrapFirst
            } else {
                SchedulerDecision::ProceedDirect
            };
        debug!(
            tensor_id,
            ?upcoming_op,
            current_noise = noise.level,
            budget = self.budget,
            ?decision,
            "noise scheduler check"
        );
        decision
    }

    pub fn record_op(&mut self, tensor_id: &str, op: FheOp) -> Result<(), TensorError> {
        let noise = self
            .state
            .entry(tensor_id.to_string())
            .or_insert_with(|| NoiseMetadata::fresh(self.budget));
        noise.apply(op)?;
        debug!(
            tensor_id,
            ?op,
            new_noise = noise.level,
            ops_since_bootstrap = noise.ops_since_bootstrap,
            "noise scheduler record"
        );
        Ok(())
    }

    pub fn record_bootstrap(&mut self, tensor_id: &str) {
        let noise = self
            .state
            .entry(tensor_id.to_string())
            .or_insert_with(|| NoiseMetadata::fresh(self.budget));
        noise.reset();
        debug!(tensor_id, "noise scheduler bootstrap reset");
    }
}

fn op_cost(op: FheOp) -> u32 {
    match op {
        FheOp::Init | FheOp::Truncate | FheOp::Bootstrap => 0,
        FheOp::Add | FheOp::Sub => 1,
        FheOp::ScalarMul(0 | 1) => 0,
        FheOp::ScalarMul(_) => HEAVY_OP_COST,
        FheOp::CtCtMul => HEAVY_OP_COST,
    }
}

fn safe_threshold(op: FheOp, budget: u32) -> u32 {
    match op {
        FheOp::Init | FheOp::Truncate | FheOp::Bootstrap => budget.saturating_add(1),
        FheOp::CtCtMul => HEAVY_OP_SAFE_THRESHOLD,
        FheOp::ScalarMul(0 | 1) => budget.saturating_add(1),
        FheOp::ScalarMul(_) => HEAVY_OP_SAFE_THRESHOLD,
        FheOp::Add | FheOp::Sub => DEFAULT_OP_SAFE_THRESHOLD.min(budget),
    }
}
