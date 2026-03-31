use serde::{Deserialize, Serialize};

use crate::tensor::ops::TensorError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl TensorShape {
    pub fn new(dims: Vec<usize>) -> Result<Self, TensorError> {
        if dims.is_empty() {
            return Err(TensorError::InvalidShape(
                "tensor shape must have at least one dimension".to_string(),
            ));
        }
        if dims.iter().any(|&dim| dim == 0) {
            return Err(TensorError::InvalidShape(
                "tensor dimensions must be non-zero".to_string(),
            ));
        }

        let mut strides = vec![1; dims.len()];
        for idx in (0..dims.len() - 1).rev() {
            strides[idx] = strides[idx + 1] * dims[idx + 1];
        }

        Ok(Self { dims, strides })
    }

    pub fn from_2d(rows: usize, cols: usize) -> Result<Self, TensorError> {
        Self::new(vec![rows, cols])
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn rows(&self) -> usize {
        if self.ndim() == 1 {
            1
        } else {
            self.dims[0]
        }
    }

    pub fn cols(&self) -> usize {
        if self.ndim() == 1 {
            self.dims[0]
        } else {
            self.dims[1]
        }
    }

    pub fn index_to_flat(&self, index: &[usize]) -> Result<usize, TensorError> {
        if index.len() != self.ndim() {
            return Err(TensorError::InvalidShape(format!(
                "expected {} indices, got {}",
                self.ndim(),
                index.len()
            )));
        }

        let mut flat = 0;
        for (axis, (&value, &dim)) in index.iter().zip(self.dims.iter()).enumerate() {
            if value >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    axis,
                    index: value,
                    dim,
                });
            }
            flat += value * self.strides[axis];
        }
        Ok(flat)
    }

    pub fn flat_to_index(&self, mut flat: usize) -> Vec<usize> {
        let mut index = vec![0; self.ndim()];
        for (axis, stride) in self.strides.iter().enumerate() {
            index[axis] = flat / stride;
            flat %= stride;
        }
        index
    }

    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.ndim() != 2 {
            return Err(TensorError::InvalidShape(
                "transpose_2d requires a rank-2 tensor".to_string(),
            ));
        }
        Self::from_2d(self.cols(), self.rows())
    }

    pub fn broadcast_shape(a: &Self, b: &Self) -> Result<Self, TensorError> {
        let max_rank = a.ndim().max(b.ndim());
        let mut out_dims = Vec::with_capacity(max_rank);

        for axis in 0..max_rank {
            let a_dim = a.dim_from_right(axis);
            let b_dim = b.dim_from_right(axis);
            let out = if a_dim == b_dim {
                a_dim
            } else if a_dim == 1 {
                b_dim
            } else if b_dim == 1 {
                a_dim
            } else {
                return Err(TensorError::BroadcastMismatch {
                    left: a.dims.clone(),
                    right: b.dims.clone(),
                });
            };
            out_dims.push(out);
        }

        out_dims.reverse();
        Self::new(out_dims)
    }

    pub fn broadcast_flat_index(
        &self,
        target_shape: &TensorShape,
        target_flat_index: usize,
    ) -> Result<usize, TensorError> {
        if self.ndim() > target_shape.ndim() {
            return Err(TensorError::BroadcastMismatch {
                left: self.dims.clone(),
                right: target_shape.dims.clone(),
            });
        }

        let target_index = target_shape.flat_to_index(target_flat_index);
        let rank_diff = target_shape.ndim() - self.ndim();
        let mut source_index = vec![0; self.ndim()];

        for axis in 0..self.ndim() {
            let source_dim = self.dims[axis];
            let target_axis = axis + rank_diff;
            let target_value = target_index[target_axis];
            source_index[axis] = if source_dim == 1 { 0 } else { target_value };
        }

        self.index_to_flat(&source_index)
    }

    fn dim_from_right(&self, axis_from_right: usize) -> usize {
        if axis_from_right >= self.ndim() {
            1
        } else {
            self.dims[self.ndim() - 1 - axis_from_right]
        }
    }
}
