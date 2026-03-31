use std::sync::Arc;

use tfhe::integer::RadixClientKey;

use crate::context::FheContext;
use crate::quantization::config::QuantConfig;
use crate::tensor::{EncryptedTensor, FheTensorOps, PlaintextTensor, TensorError, TensorShape};

#[derive(Clone, Debug)]
pub struct Quantizer {
    pub config: QuantConfig,
}

impl Quantizer {
    pub fn new(config: QuantConfig) -> Self {
        Self { config }
    }

    pub fn quantize_scalar(&self, value: f64) -> i64 {
        self.config.quantize(value)
    }

    pub fn dequantize_scalar(&self, value: i64) -> f64 {
        self.config.dequantize(value)
    }

    pub fn quantize_matrix(&self, matrix: &[Vec<f64>]) -> Result<PlaintextTensor, TensorError> {
        let quantized = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .map(|value| self.config.quantize(*value))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        PlaintextTensor::from_vec2(quantized)
    }

    pub fn truncate_after_mul(
        &self,
        tensor: PlaintextTensor,
    ) -> Result<PlaintextTensor, TensorError> {
        tensor.truncate(self.config.frac_bits)
    }

    pub fn sigmoid(&self, value: i64) -> i64 {
        let x = self.config.dequantize(value);
        self.config.quantize(1.0 / (1.0 + (-x).exp()))
    }

    pub fn encrypt_tensor(
        &self,
        matrix: &[Vec<f64>],
        client_key: &RadixClientKey,
        ctx: Arc<FheContext>,
    ) -> Result<EncryptedTensor, TensorError> {
        let shape = TensorShape::from_2d(matrix.len(), matrix.first().map_or(0, Vec::len))?;
        if matrix.is_empty() || matrix[0].is_empty() {
            return Err(TensorError::InvalidShape(
                "encrypt_tensor requires a non-empty matrix".to_string(),
            ));
        }
        let quantized = matrix
            .iter()
            .flat_map(|row| row.iter().map(|value| self.config.quantize(*value)))
            .collect::<Vec<_>>();
        self.encrypt_quantized(&quantized, shape, client_key, ctx)
    }

    pub fn encrypt_quantized(
        &self,
        values: &[i64],
        shape: TensorShape,
        client_key: &RadixClientKey,
        ctx: Arc<FheContext>,
    ) -> Result<EncryptedTensor, TensorError> {
        if values.len() != shape.elem_count() {
            return Err(TensorError::InvalidShape(format!(
                "quantized value length {} does not match shape {:?}",
                values.len(),
                shape.dims()
            )));
        }

        let data = values
            .iter()
            .map(|value| {
                let encoded = encode_signed(*value, ctx.params.total_bits);
                client_key.encrypt(encoded)
            })
            .collect::<Vec<_>>();

        EncryptedTensor::new(data, shape, ctx)
    }

    pub fn decrypt_quantized(
        &self,
        ciphertext: &tfhe::integer::RadixCiphertext,
        client_key: &RadixClientKey,
    ) -> i64 {
        let raw: u64 = client_key.decrypt(ciphertext);
        decode_signed(raw, self.config.total_bits)
    }
}

fn encode_signed(value: i64, total_bits: u32) -> u64 {
    let modulus = 1_i128 << total_bits;
    if value < 0 {
        (modulus + value as i128) as u64
    } else {
        value as u64
    }
}

fn decode_signed(value: u64, total_bits: u32) -> i64 {
    let sign_bit = 1_u64 << (total_bits - 1);
    if value & sign_bit == 0 {
        value as i64
    } else {
        (value as i128 - (1_i128 << total_bits)) as i64
    }
}
