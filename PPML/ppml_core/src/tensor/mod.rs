pub mod encrypted;
pub mod noise;
pub mod ops;
pub mod plaintext;
pub mod shape;

pub use encrypted::EncryptedTensor;
pub use noise::{
    FheOp, NoiseMetadata, DEFAULT_OP_SAFE_THRESHOLD, HEAVY_OP_COST, HEAVY_OP_SAFE_THRESHOLD,
};
pub use ops::{FheTensorOps, TensorError};
pub use plaintext::PlaintextTensor;
pub use shape::TensorShape;
