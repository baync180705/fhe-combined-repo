use serde::{Deserialize, Serialize};

use crate::tensor::ops::TensorError;

pub const HEAVY_OP_SAFE_THRESHOLD: u32 = 2;
pub const DEFAULT_OP_SAFE_THRESHOLD: u32 = 3;
pub const HEAVY_OP_COST: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FheOp {
    Init,
    Add,
    Sub,
    ScalarMul(u64),
    CtCtMul,
    Truncate,
    Bootstrap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoiseMetadata {
    pub level: u32,
    pub budget: u32,
    pub ops_since_bootstrap: usize,
}

impl NoiseMetadata {
    pub fn fresh(budget: u32) -> Self {
        Self {
            level: 0,
            budget,
            ops_since_bootstrap: 0,
        }
    }

    pub fn merge(left: &Self, right: &Self) -> Self {
        Self {
            level: left.level.max(right.level),
            budget: left.budget.max(right.budget),
            ops_since_bootstrap: left.ops_since_bootstrap.max(right.ops_since_bootstrap),
        }
    }

    pub fn apply(&mut self, op: FheOp) -> Result<(), TensorError> {
        match op {
            FheOp::Init => {}
            FheOp::Truncate | FheOp::Bootstrap => self.reset(),
            FheOp::Add | FheOp::Sub => {
                self.level = self.level.saturating_add(1);
                self.ops_since_bootstrap += 1;
            }
            FheOp::ScalarMul(scalar) => {
                self.level = self.level.saturating_add(Self::scalar_mul_cost(scalar));
                self.ops_since_bootstrap += 1;
            }
            FheOp::CtCtMul => {
                self.level = self.level.saturating_add(HEAVY_OP_COST);
                self.ops_since_bootstrap += 1;
            }
        }

        if self.level > self.budget {
            return Err(TensorError::NoiseBudgetExceeded {
                level: self.level,
                budget: self.budget,
            });
        }

        Ok(())
    }

    pub fn reset(&mut self) {
        self.level = 0;
        self.ops_since_bootstrap = 0;
    }

    fn scalar_mul_cost(scalar: u64) -> u32 {
        match scalar {
            0 | 1 => 0,
            _ => HEAVY_OP_COST,
        }
    }
}
