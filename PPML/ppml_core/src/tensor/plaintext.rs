use crate::tensor::noise::NoiseMetadata;
use crate::tensor::ops::{FheTensorOps, TensorError};
use crate::tensor::shape::TensorShape;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaintextTensor {
    data: Vec<i64>,
    shape: TensorShape,
}

impl PlaintextTensor {
    pub fn new(data: Vec<i64>, shape: TensorShape) -> Result<Self, TensorError> {
        if data.len() != shape.elem_count() {
            return Err(TensorError::InvalidShape(format!(
                "tensor data length {} does not match shape {:?}",
                data.len(),
                shape.dims()
            )));
        }
        Ok(Self { data, shape })
    }

    pub fn from_vec2(data: Vec<Vec<i64>>) -> Result<Self, TensorError> {
        let rows = data.len();
        let cols = data.first().map_or(0, Vec::len);
        if rows == 0 || cols == 0 {
            return Err(TensorError::InvalidShape(
                "from_vec2 requires a non-empty matrix".to_string(),
            ));
        }
        if data.iter().any(|row| row.len() != cols) {
            return Err(TensorError::InvalidShape(
                "all rows must have the same length".to_string(),
            ));
        }

        let flat = data.into_iter().flatten().collect();
        Self::new(flat, TensorShape::from_2d(rows, cols)?)
    }

    pub fn zeros(rows: usize, cols: usize) -> Result<Self, TensorError> {
        let shape = TensorShape::from_2d(rows, cols)?;
        Self::new(vec![0; shape.elem_count()], shape)
    }

    pub fn map<F>(&self, mut f: F) -> Result<Self, TensorError>
    where
        F: FnMut(i64) -> i64,
    {
        let data = self.data.iter().copied().map(&mut f).collect();
        Self::new(data, self.shape.clone())
    }

    pub fn transpose(&self) -> Result<Self, TensorError> {
        let out_shape = self.shape.transpose_2d()?;
        let rows = self.shape.rows();
        let cols = self.shape.cols();
        let mut out = vec![0; self.data.len()];

        for row in 0..rows {
            for col in 0..cols {
                out[col * rows + row] = self.data[row * cols + col];
            }
        }

        Self::new(out, out_shape)
    }

    pub fn sum_axis0(&self) -> Result<Self, TensorError> {
        if self.shape.dims().len() != 2 {
            return Err(TensorError::InvalidShape(
                "sum_axis0 requires a rank-2 tensor".to_string(),
            ));
        }

        let rows = self.shape.rows();
        let cols = self.shape.cols();
        let mut sums = vec![0; cols];

        for row in 0..rows {
            for col in 0..cols {
                sums[col] += self.data[row * cols + col];
            }
        }

        Self::new(sums, TensorShape::from_2d(1, cols)?)
    }

    pub fn data(&self) -> &[i64] {
        &self.data
    }

    fn elementwise_op<F>(&self, other: &Self, mut f: F) -> Result<Self, TensorError>
    where
        F: FnMut(i64, i64) -> i64,
    {
        let out_shape = TensorShape::broadcast_shape(&self.shape, &other.shape)?;
        let mut out = Vec::with_capacity(out_shape.elem_count());

        for flat in 0..out_shape.elem_count() {
            let left_idx = self.shape.broadcast_flat_index(&out_shape, flat)?;
            let right_idx = other.shape.broadcast_flat_index(&out_shape, flat)?;
            out.push(f(self.data[left_idx], other.data[right_idx]));
        }

        Self::new(out, out_shape)
    }
}

impl FheTensorOps for PlaintextTensor {
    fn add(&self, other: &Self) -> Result<Self, TensorError> {
        self.elementwise_op(other, |a, b| a + b)
    }

    fn sub(&self, other: &Self) -> Result<Self, TensorError> {
        self.elementwise_op(other, |a, b| a - b)
    }

    fn scalar_mul(&self, scalar: i64) -> Result<Self, TensorError> {
        self.map(|value| value * scalar)
    }

    fn div_scalar(&self, scalar: i64) -> Result<Self, TensorError> {
        if scalar == 0 {
            return Err(TensorError::DivisionByZero);
        }
        self.map(|value| value / scalar)
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

        let mut out = vec![0; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut acc = 0_i64;
                for depth in 0..k {
                    acc += self.data[row * k + depth] * other.data[depth * n + col];
                }
                out[row * n + col] = acc;
            }
        }

        Self::new(out, TensorShape::from_2d(m, n)?)
    }

    fn truncate(&self, frac_bits: u32) -> Result<Self, TensorError> {
        let shift = frac_bits as i64;
        self.map(|value| {
            if shift == 0 {
                value
            } else if value >= 0 {
                (value + (1_i64 << (shift - 1))) >> shift
            } else {
                -(((-value) + (1_i64 << (shift - 1))) >> shift)
            }
        })
    }

    fn noise_level(&self) -> NoiseMetadata {
        NoiseMetadata::fresh(0)
    }

    fn shape(&self) -> &TensorShape {
        &self.shape
    }
}
