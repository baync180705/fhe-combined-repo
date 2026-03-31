use thiserror::Error;

use crate::tensor::noise::NoiseMetadata;
use crate::tensor::shape::TensorShape;

#[derive(Debug, Error)]
pub enum TensorError {
    #[error("invalid shape: {0}")]
    InvalidShape(String),
    #[error("broadcast mismatch: left={left:?}, right={right:?}")]
    BroadcastMismatch { left: Vec<usize>, right: Vec<usize> },
    #[error("matmul dimension mismatch: left={left:?}, right={right:?}")]
    MatmulMismatch { left: Vec<usize>, right: Vec<usize> },
    #[error("index out of bounds on axis {axis}: index={index}, dim={dim}")]
    IndexOutOfBounds {
        axis: usize,
        index: usize,
        dim: usize,
    },
    #[error("division by zero")]
    DivisionByZero,
    #[error("noise budget exceeded: level={level}, budget={budget}")]
    NoiseBudgetExceeded { level: u32, budget: u32 },
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(&'static str),
    #[error("i/o error: {0}")]
    Io(String),
}

pub trait FheTensorOps: Sized + Clone {
    fn add(&self, other: &Self) -> Result<Self, TensorError>;
    fn sub(&self, other: &Self) -> Result<Self, TensorError>;
    fn scalar_mul(&self, scalar: i64) -> Result<Self, TensorError>;
    fn div_scalar(&self, scalar: i64) -> Result<Self, TensorError>;
    fn matmul(&self, other: &Self) -> Result<Self, TensorError>;
    fn truncate(&self, frac_bits: u32) -> Result<Self, TensorError>;
    fn noise_level(&self) -> NoiseMetadata;
    fn shape(&self) -> &TensorShape;
}
