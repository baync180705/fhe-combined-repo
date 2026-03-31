use std::sync::Arc;

use tfhe::integer::wopbs::WopbsKey;
use tfhe::integer::{RadixCiphertext, ServerKey};

use crate::quantization::config::QuantConfig;

#[derive(Clone)]
pub struct LutEngine {
    pub server_key: Arc<ServerKey>,
    pub wopbs_key: Arc<WopbsKey>,
    pub config: QuantConfig,
}

impl std::fmt::Debug for LutEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LutEngine")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl LutEngine {
    pub fn new(server_key: Arc<ServerKey>, wopbs_key: Arc<WopbsKey>, config: QuantConfig) -> Self {
        Self {
            server_key,
            wopbs_key,
            config,
        }
    }

    pub fn from_client_key(
        client_key: &tfhe::integer::RadixClientKey,
        wopbs_key: Arc<WopbsKey>,
        config: QuantConfig,
    ) -> Self {
        Self::new(
            Arc::new(ServerKey::new_radix_server_key(client_key.as_ref())),
            wopbs_key,
            config,
        )
    }

    pub fn sigmoid(&self, ct: &RadixCiphertext) -> RadixCiphertext {
        let scale = self.config.scale as f64;
        let ct_wopbs = self
            .wopbs_key
            .keyswitch_to_wopbs_params(&self.server_key, ct);
        let lut_fn = |q: u64| -> u64 {
            let q_signed = decode_signed_lut(q, self.config.total_bits) as f64;
            let x = q_signed / scale;
            let s = 1.0 / (1.0 + (-x).exp());
            ((s * scale).round() as i64).clamp(0, self.config.q_max) as u64
        };
        let result_wopbs = self.wopbs_key.wopbs(
            &ct_wopbs,
            &self.wopbs_key.generate_lut_radix(&ct_wopbs, lut_fn),
        );
        self.wopbs_key.keyswitch_to_pbs_params(&result_wopbs)
    }

    pub fn sigmoid_prime(&self, ct: &RadixCiphertext) -> RadixCiphertext {
        let scale = self.config.scale as f64;
        let ct_wopbs = self
            .wopbs_key
            .keyswitch_to_wopbs_params(&self.server_key, ct);
        let lut_fn = |q: u64| -> u64 {
            let q_signed = decode_signed_lut(q, self.config.total_bits) as f64;
            let x = q_signed / scale;
            let s = 1.0 / (1.0 + (-x).exp());
            let sp = s * (1.0 - s);
            ((sp * scale).round() as i64).clamp(0, self.config.q_max) as u64
        };
        let result_wopbs = self.wopbs_key.wopbs(
            &ct_wopbs,
            &self.wopbs_key.generate_lut_radix(&ct_wopbs, lut_fn),
        );
        self.wopbs_key.keyswitch_to_pbs_params(&result_wopbs)
    }

    pub fn truncate_then_sigmoid(&self, ct: &RadixCiphertext) -> RadixCiphertext {
        let scale_single = self.config.scale as f64;
        let scale_double = scale_single * scale_single;
        let ct_wopbs = self
            .wopbs_key
            .keyswitch_to_wopbs_params(&self.server_key, ct);
        let lut_fn = |q: u64| -> u64 {
            let q_signed = if q >= 32 { (q as i64) - 64 } else { q as i64 };
            let x_float = q_signed as f64 / scale_double;
            let sig = 1.0 / (1.0 + (-x_float).exp());
            (sig * scale_single).round().clamp(0.0, 8.0) as u64
        };
        let result_wopbs = self.wopbs_key.wopbs(
            &ct_wopbs,
            &self.wopbs_key.generate_lut_radix(&ct_wopbs, lut_fn),
        );
        self.wopbs_key.keyswitch_to_pbs_params(&result_wopbs)
    }

    pub fn fused_truncate_and_sigmoid(&self, ct: &RadixCiphertext) -> RadixCiphertext {
        self.truncate_then_sigmoid(ct)
    }
}

fn decode_signed_lut(value: u64, total_bits: u32) -> i64 {
    let sign_bit = 1_u64 << (total_bits - 1);
    let mask = (1_u64 << total_bits) - 1;
    let truncated = value & mask;
    if truncated & sign_bit == 0 {
        truncated as i64
    } else {
        (truncated as i128 - (1_i128 << total_bits)) as i64
    }
}
