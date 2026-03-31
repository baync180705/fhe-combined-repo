use std::sync::Arc;

#[cfg(not(feature = "gpu"))]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tfhe::integer::RadixCiphertext;

#[cfg(feature = "gpu")]
use tfhe::integer::gpu::ciphertext::CudaUnsignedRadixCiphertext;

use crate::context::FheContext;
use crate::tensor::noise::{
    FheOp, NoiseMetadata, DEFAULT_OP_SAFE_THRESHOLD, HEAVY_OP_SAFE_THRESHOLD,
};
use crate::tensor::ops::{FheTensorOps, TensorError};
use crate::tensor::shape::TensorShape;

#[derive(Clone, Serialize, Deserialize)]
pub struct EncryptedTensor {
    pub(crate) data: Vec<RadixCiphertext>,
    pub(crate) shape: TensorShape,
    pub(crate) noise: NoiseMetadata,
    #[serde(skip_serializing, skip_deserializing, default = "missing_fhe_context")]
    pub(crate) ctx: Arc<FheContext>,
}

#[derive(Serialize, Deserialize)]
struct EncryptedTensorSnapshot {
    data: Vec<RadixCiphertext>,
    shape: TensorShape,
    noise: NoiseMetadata,
}

impl std::fmt::Debug for EncryptedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedTensor")
            .field("shape", &self.shape)
            .field("noise", &self.noise)
            .field("len", &self.data.len())
            .finish()
    }
}

impl EncryptedTensor {
    pub fn new(
        data: Vec<RadixCiphertext>,
        shape: TensorShape,
        ctx: Arc<FheContext>,
    ) -> Result<Self, TensorError> {
        if data.len() != shape.elem_count() {
            return Err(TensorError::InvalidShape(format!(
                "tensor data length {} does not match shape {:?}",
                data.len(),
                shape.dims()
            )));
        }

        Ok(Self {
            data,
            shape,
            noise: NoiseMetadata::fresh(ctx.max_noise_budget),
            ctx,
        })
    }

    pub fn from_parts(
        data: Vec<RadixCiphertext>,
        shape: TensorShape,
        noise: NoiseMetadata,
        ctx: Arc<FheContext>,
    ) -> Result<Self, TensorError> {
        if data.len() != shape.elem_count() {
            return Err(TensorError::InvalidShape(format!(
                "tensor data length {} does not match shape {:?}",
                data.len(),
                shape.dims()
            )));
        }

        Ok(Self {
            data,
            shape,
            noise,
            ctx,
        })
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        let out_shape = self.shape.transpose_2d()?;
        let rows = self.shape.rows();
        let cols = self.shape.cols();
        let mut out = vec![self.zero_ciphertext(); self.data.len()];

        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = self.data[row * cols + col].clone();
            }
        }

        Self::from_parts(out, out_shape, self.noise.clone(), Arc::clone(&self.ctx))
    }

    pub fn data(&self) -> &[RadixCiphertext] {
        &self.data
    }

    pub fn noise(&self) -> &NoiseMetadata {
        &self.noise
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, TensorError> {
        let snapshot = EncryptedTensorSnapshot {
            data: self.data.clone(),
            shape: self.shape.clone(),
            noise: self.noise.clone(),
        };
        bincode::serialize(&snapshot).map_err(|error| {
            TensorError::Io(format!("failed to serialize encrypted tensor: {error}"))
        })
    }

    pub fn from_bytes(bytes: &[u8], ctx: Arc<FheContext>) -> Result<Self, TensorError> {
        let snapshot: EncryptedTensorSnapshot = bincode::deserialize(bytes).map_err(|error| {
            TensorError::Io(format!("failed to deserialize encrypted tensor: {error}"))
        })?;
        Self::from_parts(snapshot.data, snapshot.shape, snapshot.noise, ctx)
    }

    fn binary_noise(&self, other: &Self, op: FheOp) -> Result<NoiseMetadata, TensorError> {
        let mut noise = NoiseMetadata::merge(&self.noise, &other.noise);
        noise.budget = self.ctx.max_noise_budget;
        noise.apply(op)?;
        Ok(noise)
    }

    fn unary_noise(&self, op: FheOp) -> Result<NoiseMetadata, TensorError> {
        let mut noise = self.noise.clone();
        noise.budget = self.ctx.max_noise_budget;
        noise.apply(op)?;
        Ok(noise)
    }

    fn refresh_for_heavy_op(&self) -> Result<Self, TensorError> {
        if self.noise.level >= HEAVY_OP_SAFE_THRESHOLD {
            self.bootstrap()
        } else {
            Ok(self.clone())
        }
    }

    fn project_accumulation_noise(
        &self,
        acc_noise: &NoiseMetadata,
        incoming_noise: &NoiseMetadata,
        op: FheOp,
    ) -> Result<NoiseMetadata, TensorError> {
        let mut projected = NoiseMetadata::merge(acc_noise, incoming_noise);
        projected.budget = self.ctx.max_noise_budget;
        projected.apply(op)?;
        Ok(projected)
    }

    #[cfg(feature = "gpu")]
    fn project_matmul_output_noise(
        &self,
        other: &Self,
        reductions: usize,
    ) -> Result<NoiseMetadata, TensorError> {
        let mut acc_noise = NoiseMetadata::fresh(self.ctx.max_noise_budget);
        let mut product_noise = NoiseMetadata::merge(&self.noise, &other.noise);
        product_noise.budget = self.ctx.max_noise_budget;
        product_noise.apply(FheOp::CtCtMul)?;

        for _ in 0..reductions {
            if acc_noise.level >= DEFAULT_OP_SAFE_THRESHOLD {
                acc_noise.reset();
            }
            acc_noise = self.project_accumulation_noise(&acc_noise, &product_noise, FheOp::Add)?;
        }

        Ok(acc_noise)
    }

    #[cfg(feature = "gpu")]
    pub fn truncate_approximate(&self, frac_bits: u32) -> Result<Self, TensorError> {
        // GPU builds currently fall back to the exact shift path. The approximate
        // optimization here is intended for the CPU gradient-update regime.
        self.truncate(frac_bits)
    }
}

fn missing_fhe_context() -> Arc<FheContext> {
    panic!("deserializing EncryptedTensor requires an FheContext to be reattached")
}

#[cfg(feature = "gpu")]
impl EncryptedTensor {
    pub fn force_refresh(&self) -> Result<Self, TensorError> {
        let refreshed = self.bootstrap()?;
        let noise = NoiseMetadata::fresh(self.ctx.max_noise_budget);
        Self::from_parts(
            refreshed.data.clone(),
            refreshed.shape.clone(),
            noise,
            Arc::clone(&refreshed.ctx),
        )
    }

    pub fn bootstrap(&self) -> Result<Self, TensorError> {
        let sk = Arc::clone(&self.ctx.server_key);
        let streams = self.ctx.streams();
        let out = self
            .upload_data(self.data())
            .into_iter()
            .map(|ct| {
                let zero = sk.create_trivial_radix::<u64, CudaUnsignedRadixCiphertext>(
                    0u64,
                    self.ctx.params.num_blocks,
                    streams,
                );
                let refreshed = sk.add(&ct, &zero, streams);
                self.download_ciphertext(&refreshed)
            })
            .collect::<Vec<_>>();
        let mut noise = self.noise.clone();
        noise.reset();
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    pub fn sum_axis0(&self) -> Result<Self, TensorError> {
        if self.shape.dims().len() != 2 {
            return Err(TensorError::InvalidShape(
                "sum_axis0 requires a rank-2 tensor".to_string(),
            ));
        }
        let source = self.refresh_for_heavy_op()?;
        let rows = self.shape.rows();
        let cols = self.shape.cols();
        let sk = Arc::clone(&source.ctx.server_key);
        let streams = source.ctx.streams();
        let gpu_data = source.upload_data(source.data());
        let mut out = Vec::with_capacity(cols);
        let mut out_noise = NoiseMetadata::fresh(source.ctx.max_noise_budget);

        for col in 0..cols {
            let mut acc = sk.create_trivial_radix::<u64, CudaUnsignedRadixCiphertext>(
                0u64,
                source.ctx.params.num_blocks,
                streams,
            );
            let mut acc_noise = NoiseMetadata::fresh(source.ctx.max_noise_budget);
            for row in 0..rows {
                acc = source.bootstrap_accumulator_if_needed(acc, &mut acc_noise)?;
                acc = sk.add(&acc, &gpu_data[row * cols + col], streams);
                acc_noise =
                    source.project_accumulation_noise(&acc_noise, &source.noise, FheOp::Add)?;
            }
            out.push(source.download_ciphertext(&acc));
            out_noise = acc_noise;
        }

        Self::from_parts(
            out,
            TensorShape::from_2d(1, cols)?,
            out_noise,
            Arc::clone(&source.ctx),
        )
    }

    fn zero_ciphertext(&self) -> RadixCiphertext {
        let zero = self
            .ctx
            .server_key
            .create_trivial_radix::<u64, CudaUnsignedRadixCiphertext>(
                0u64,
                self.ctx.params.num_blocks,
                self.ctx.streams(),
            );
        self.download_ciphertext(&zero)
    }

    fn upload_data(&self, data: &[RadixCiphertext]) -> Vec<CudaUnsignedRadixCiphertext> {
        let streams = self.ctx.streams();
        data.iter()
            .map(|ct| CudaUnsignedRadixCiphertext::from_radix_ciphertext(ct, streams))
            .collect()
    }

    fn download_ciphertext(&self, ct: &CudaUnsignedRadixCiphertext) -> RadixCiphertext {
        ct.to_radix_ciphertext(self.ctx.streams())
    }

    fn bootstrap_accumulator_if_needed(
        &self,
        acc: CudaUnsignedRadixCiphertext,
        acc_noise: &mut NoiseMetadata,
    ) -> Result<CudaUnsignedRadixCiphertext, TensorError> {
        if acc_noise.level >= DEFAULT_OP_SAFE_THRESHOLD {
            let refreshed = self.bootstrap_gpu_ciphertext(&acc);
            acc_noise.reset();
            Ok(refreshed)
        } else {
            Ok(acc)
        }
    }

    fn bootstrap_gpu_ciphertext(
        &self,
        ct: &CudaUnsignedRadixCiphertext,
    ) -> CudaUnsignedRadixCiphertext {
        let streams = self.ctx.streams();
        let zero = self
            .ctx
            .server_key
            .create_trivial_radix::<u64, CudaUnsignedRadixCiphertext>(
                0u64,
                self.ctx.params.num_blocks,
                streams,
            );
        self.ctx.server_key.add(ct, &zero, streams)
    }

    fn elementwise_gpu_op<F>(&self, other: &Self, mut f: F, op: FheOp) -> Result<Self, TensorError>
    where
        F: FnMut(
            &tfhe::integer::gpu::CudaServerKey,
            &CudaUnsignedRadixCiphertext,
            &CudaUnsignedRadixCiphertext,
            &tfhe::core_crypto::gpu::CudaStreams,
        ) -> CudaUnsignedRadixCiphertext,
    {
        let out_shape = TensorShape::broadcast_shape(&self.shape, &other.shape)?;
        let sk = Arc::clone(&self.ctx.server_key);
        let streams = self.ctx.streams();
        let left = self.upload_data(self.data());
        let right = self.upload_data(other.data());
        let mut out = Vec::with_capacity(out_shape.elem_count());

        for flat in 0..out_shape.elem_count() {
            let left_idx = self.shape.broadcast_flat_index(&out_shape, flat)?;
            let right_idx = other.shape.broadcast_flat_index(&out_shape, flat)?;
            let result = f(&sk, &left[left_idx], &right[right_idx], streams);
            out.push(self.download_ciphertext(&result));
        }

        let noise = self.binary_noise(other, op)?;
        Self::from_parts(out, out_shape, noise, Arc::clone(&self.ctx))
    }
}

#[cfg(feature = "gpu")]
impl FheTensorOps for EncryptedTensor {
    fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.elementwise_gpu_op(
            other,
            |sk, lhs, rhs, streams| sk.add(lhs, rhs, streams),
            FheOp::Add,
        )
    }

    fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.elementwise_gpu_op(
            other,
            |sk, lhs, rhs, streams| sk.sub(lhs, rhs, streams),
            FheOp::Sub,
        )
    }

    fn scalar_mul(&self, scalar: i64) -> Result<Self, TensorError> {
        if scalar < 0 {
            return Err(TensorError::UnsupportedOperation(
                "negative scalar multiplication is not supported on unsigned radix ciphertexts",
            ));
        }

        let sk = Arc::clone(&self.ctx.server_key);
        let streams = self.ctx.streams();
        let scalar_u64 = scalar as u64;
        let out = self
            .upload_data(self.data())
            .into_iter()
            .map(|ct| sk.scalar_mul(&ct, scalar_u64, streams))
            .map(|ct| self.download_ciphertext(&ct))
            .collect::<Vec<_>>();
        let noise = self.unary_noise(FheOp::ScalarMul(scalar_u64))?;
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    fn div_scalar(&self, scalar: i64) -> Result<Self, TensorError> {
        if scalar == 0 {
            return Err(TensorError::DivisionByZero);
        }
        if scalar < 0 {
            return Err(TensorError::UnsupportedOperation(
                "negative scalar division is not supported on unsigned radix ciphertexts",
            ));
        }

        let sk = Arc::clone(&self.ctx.server_key);
        let streams = self.ctx.streams();
        let out = self
            .upload_data(self.data())
            .into_iter()
            .map(|ct| sk.scalar_div(&ct, scalar as u64, streams))
            .map(|ct| self.download_ciphertext(&ct))
            .collect::<Vec<_>>();
        let mut noise = self.noise.clone();
        noise.reset();
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape.dims().len() != 2 || other.shape.dims().len() != 2 {
            return Err(TensorError::InvalidShape(
                "matmul requires two rank-2 tensors".to_string(),
            ));
        }

        let m = self.shape.rows();
        let k = self.shape.cols();
        let rhs_rows = other.shape.rows();
        let n = other.shape.cols();

        if k != rhs_rows {
            return Err(TensorError::MatmulMismatch {
                left: self.shape.dims().to_vec(),
                right: other.shape.dims().to_vec(),
            });
        }

        let lhs = self.refresh_for_heavy_op()?;
        let rhs = other.refresh_for_heavy_op()?;
        let sk = Arc::clone(&lhs.ctx.server_key);
        let streams = lhs.ctx.streams();
        let left = lhs.upload_data(lhs.data());
        let right = rhs.upload_data(rhs.data());
        let num_blocks = lhs.ctx.params.num_blocks;
        let mut product_noise = NoiseMetadata::merge(&lhs.noise, &rhs.noise);
        product_noise.budget = lhs.ctx.max_noise_budget;
        product_noise.apply(FheOp::CtCtMul)?;
        let mut out = Vec::with_capacity(m * n);

        for row in 0..m {
            for col in 0..n {
                let mut acc = sk.create_trivial_radix::<u64, CudaUnsignedRadixCiphertext>(
                    0u64, num_blocks, streams,
                );
                let mut acc_noise = NoiseMetadata::fresh(lhs.ctx.max_noise_budget);
                for depth in 0..k {
                    let lhs_ct = &left[row * k + depth];
                    let rhs_ct = &right[depth * n + col];
                    let product = sk.mul(lhs_ct, rhs_ct, streams);
                    acc = lhs.bootstrap_accumulator_if_needed(acc, &mut acc_noise)?;
                    acc = sk.add(&acc, &product, streams);
                    acc_noise =
                        lhs.project_accumulation_noise(&acc_noise, &product_noise, FheOp::Add)?;
                }
                out.push(lhs.download_ciphertext(&acc));
            }
        }

        let noise = lhs.project_matmul_output_noise(&rhs, k)?;
        Self::from_parts(
            out,
            TensorShape::from_2d(m, n)?,
            noise,
            Arc::clone(&lhs.ctx),
        )
    }

    fn truncate(&self, frac_bits: u32) -> Result<Self, TensorError> {
        let sk = Arc::clone(&self.ctx.server_key);
        let streams = self.ctx.streams();
        let out = self
            .upload_data(self.data())
            .into_iter()
            .map(|ct| sk.scalar_right_shift(&ct, frac_bits, streams))
            .map(|ct| self.download_ciphertext(&ct))
            .collect::<Vec<_>>();
        let noise = self.unary_noise(FheOp::Truncate)?;
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    fn noise_level(&self) -> NoiseMetadata {
        self.noise.clone()
    }

    fn shape(&self) -> &TensorShape {
        &self.shape
    }
}

#[cfg(not(feature = "gpu"))]
impl EncryptedTensor {
    pub fn force_refresh(&self) -> Result<Self, TensorError> {
        let mut out = self.data.clone();
        let sk = Arc::clone(&self.ctx.server_key);
        out.par_iter_mut()
            .for_each(|ct| sk.full_propagate_parallelized(ct));

        let noise = NoiseMetadata::fresh(self.ctx.max_noise_budget);
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    pub fn bootstrap(&self) -> Result<Self, TensorError> {
        self.force_refresh()
    }

    pub fn sum_axis0(&self) -> Result<Self, TensorError> {
        if self.shape.dims().len() != 2 {
            return Err(TensorError::InvalidShape(
                "sum_axis0 requires a rank-2 tensor".to_string(),
            ));
        }
        let source = self.refresh_for_heavy_op()?;
        let rows = source.shape.rows();
        let cols = source.shape.cols();
        let sk = Arc::clone(&source.ctx.server_key);
        let mut out = Vec::with_capacity(cols);
        let mut out_noise = NoiseMetadata::fresh(source.ctx.max_noise_budget);

        for col in 0..cols {
            let mut acc =
                sk.create_trivial_radix::<u64, RadixCiphertext>(0u64, source.ctx.params.num_blocks);
            let mut acc_noise = NoiseMetadata::fresh(source.ctx.max_noise_budget);
            for row in 0..rows {
                acc = source.bootstrap_accumulator_if_needed(acc, &mut acc_noise)?;
                acc = sk.add_parallelized(&acc, &source.data[row * cols + col]);
                acc_noise =
                    source.project_accumulation_noise(&acc_noise, &source.noise, FheOp::Add)?;
            }
            out.push(acc);
            out_noise = acc_noise;
        }

        Self::from_parts(
            out,
            TensorShape::from_2d(1, cols)?,
            out_noise,
            Arc::clone(&source.ctx),
        )
    }

    fn zero_ciphertext(&self) -> RadixCiphertext {
        self.ctx
            .server_key
            .create_trivial_radix::<u64, RadixCiphertext>(0u64, self.ctx.params.num_blocks)
    }

    fn bootstrap_accumulator_if_needed(
        &self,
        acc: RadixCiphertext,
        acc_noise: &mut NoiseMetadata,
    ) -> Result<RadixCiphertext, TensorError> {
        if acc_noise.level >= DEFAULT_OP_SAFE_THRESHOLD {
            let zero = self
                .ctx
                .server_key
                .create_trivial_radix::<u64, RadixCiphertext>(0u64, self.ctx.params.num_blocks);
            let refreshed = self.ctx.server_key.add_parallelized(&acc, &zero);
            acc_noise.reset();
            Ok(refreshed)
        } else {
            Ok(acc)
        }
    }

    fn elementwise_cpu_op<F>(&self, other: &Self, mut f: F, op: FheOp) -> Result<Self, TensorError>
    where
        F: FnMut(&tfhe::integer::ServerKey, &RadixCiphertext, &RadixCiphertext) -> RadixCiphertext,
    {
        let out_shape = TensorShape::broadcast_shape(&self.shape, &other.shape)?;
        let sk = Arc::clone(&self.ctx.server_key);
        let mut out = Vec::with_capacity(out_shape.elem_count());

        for flat in 0..out_shape.elem_count() {
            let left_idx = self.shape.broadcast_flat_index(&out_shape, flat)?;
            let right_idx = other.shape.broadcast_flat_index(&out_shape, flat)?;
            let result = f(&sk, &self.data[left_idx], &other.data[right_idx]);
            out.push(result);
        }

        let noise = self.binary_noise(other, op)?;
        Self::from_parts(out, out_shape, noise, Arc::clone(&self.ctx))
    }

    pub fn truncate_approximate(&self, frac_bits: u32) -> Result<Self, TensorError> {
        let sk = Arc::clone(&self.ctx.server_key);
        let out = self
            .data
            .par_iter()
            .map(|ct| {
                let mut shifted = sk.unchecked_scalar_right_shift_parallelized(ct, frac_bits);
                sk.full_propagate_parallelized(&mut shifted);
                shifted
            })
            .collect::<Vec<_>>();
        let noise = self.unary_noise(FheOp::Truncate)?;
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }
}

#[cfg(not(feature = "gpu"))]
impl FheTensorOps for EncryptedTensor {
    fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.elementwise_cpu_op(
            other,
            |sk, lhs, rhs| sk.add_parallelized(lhs, rhs),
            FheOp::Add,
        )
    }

    fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.elementwise_cpu_op(
            other,
            |sk, lhs, rhs| sk.sub_parallelized(lhs, rhs),
            FheOp::Sub,
        )
    }

    fn scalar_mul(&self, scalar: i64) -> Result<Self, TensorError> {
        if scalar < 0 {
            return Err(TensorError::UnsupportedOperation(
                "negative scalar multiplication is not supported on unsigned radix ciphertexts",
            ));
        }

        let scalar_u64 = scalar as u64;
        let out = self
            .data
            .iter()
            .map(|ct| self.ctx.server_key.scalar_mul_parallelized(ct, scalar_u64))
            .collect::<Vec<_>>();
        let noise = self.unary_noise(FheOp::ScalarMul(scalar_u64))?;
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    fn div_scalar(&self, scalar: i64) -> Result<Self, TensorError> {
        if scalar == 0 {
            return Err(TensorError::DivisionByZero);
        }
        if scalar < 0 {
            return Err(TensorError::UnsupportedOperation(
                "negative scalar division is not supported on unsigned radix ciphertexts",
            ));
        }

        let out = self
            .data
            .iter()
            .map(|ct| {
                self.ctx
                    .server_key
                    .scalar_div_parallelized(ct, scalar as u64)
            })
            .collect::<Vec<_>>();
        let mut noise = self.noise.clone();
        noise.reset();
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    fn matmul(&self, other: &Self) -> Result<Self, TensorError> {
        if self.shape.dims().len() != 2 || other.shape.dims().len() != 2 {
            return Err(TensorError::InvalidShape(
                "matmul requires two rank-2 tensors".to_string(),
            ));
        }

        let m = self.shape.rows();
        let k = self.shape.cols();
        let rhs_rows = other.shape.rows();
        let n = other.shape.cols();

        if k != rhs_rows {
            return Err(TensorError::MatmulMismatch {
                left: self.shape.dims().to_vec(),
                right: other.shape.dims().to_vec(),
            });
        }

        let lhs = self.refresh_for_heavy_op()?;
        let rhs = other.refresh_for_heavy_op()?;
        let sk = Arc::clone(&lhs.ctx.server_key);
        let mut product_noise = NoiseMetadata::merge(&lhs.noise, &rhs.noise);
        product_noise.budget = lhs.ctx.max_noise_budget;
        product_noise.apply(FheOp::CtCtMul)?;
        let mut out = Vec::with_capacity(m * n);

        for row in 0..m {
            for col in 0..n {
                let mut acc = sk
                    .create_trivial_radix::<u64, RadixCiphertext>(0u64, lhs.ctx.params.num_blocks);
                for depth in 0..k {
                    let lhs_ct = &lhs.data[row * k + depth];
                    let rhs_ct = &rhs.data[depth * n + col];
                    let product = sk.mul_parallelized(lhs_ct, rhs_ct);
                    // For the forward-path dot product we intentionally keep the raw
                    // multiplication results in the ciphertext carry buffer and defer the
                    // expensive WoPBS refresh until the final fused truncate+sigmoid stage.
                    acc = sk.unchecked_add_parallelized(&acc, &product);
                }
                // Flush the carry buffer into message bits before the fused WoPBS LUT.
                // This is a fast parallel carry propagation, not a full bootstrap.
                sk.full_propagate_parallelized(&mut acc);
                out.push(acc);
            }
        }

        let mut noise = NoiseMetadata::merge(&lhs.noise, &rhs.noise);
        noise.budget = lhs.ctx.max_noise_budget;
        noise.apply(FheOp::CtCtMul)?;
        Self::from_parts(
            out,
            TensorShape::from_2d(m, n)?,
            noise,
            Arc::clone(&lhs.ctx),
        )
    }

    fn truncate(&self, frac_bits: u32) -> Result<Self, TensorError> {
        let out = self
            .data
            .iter()
            .map(|ct| {
                self.ctx
                    .server_key
                    .scalar_right_shift_parallelized(ct, frac_bits)
            })
            .collect::<Vec<_>>();
        let noise = self.unary_noise(FheOp::Truncate)?;
        Self::from_parts(out, self.shape.clone(), noise, Arc::clone(&self.ctx))
    }

    fn noise_level(&self) -> NoiseMetadata {
        self.noise.clone()
    }

    fn shape(&self) -> &TensorShape {
        &self.shape
    }
}
