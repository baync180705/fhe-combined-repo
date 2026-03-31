#[derive(Clone, Debug)]
pub struct QuantConfig {
    pub frac_bits: u32,
    pub total_bits: u32,
    pub scale: i64,
    pub q_min: i64,
    pub q_max: i64,
}

impl QuantConfig {
    pub fn q6f3() -> Self {
        Self {
            frac_bits: 3,
            total_bits: 6,
            scale: 1_i64 << 3,
            q_min: -32,
            q_max: 31,
        }
    }

    pub fn q16f8() -> Self {
        Self::q6f3()
    }

    pub fn quantize(&self, value: f64) -> i64 {
        (value * self.scale as f64)
            .round()
            .clamp(self.q_min as f64, self.q_max as f64) as i64
    }

    pub fn dequantize(&self, value: i64) -> f64 {
        value as f64 / self.scale as f64
    }

    pub fn clamp_quantized(&self, value: i64) -> i64 {
        value.clamp(self.q_min, self.q_max)
    }
}
