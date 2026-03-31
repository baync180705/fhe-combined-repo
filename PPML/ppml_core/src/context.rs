use std::fs;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
#[cfg(feature = "gpu")]
use tfhe::core_crypto::gpu::vec::GpuIndex;
#[cfg(feature = "gpu")]
use tfhe::core_crypto::gpu::{get_number_of_gpus, CudaStreams};
#[cfg(feature = "gpu")]
use tfhe::integer::gpu::CudaServerKey;
use tfhe::integer::wopbs::WopbsKey;
use tfhe::integer::{ClientKey, RadixClientKey, ServerKey};
use tfhe::shortint::parameters::parameters_wopbs_message_carry::LEGACY_WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS;
use tfhe::shortint::parameters::PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128;

pub use crate::quantization::config::QuantConfig;

#[derive(Serialize, Deserialize)]
struct CachedFheKeys {
    client_key: RadixClientKey,
    server_key: ServerKey,
    wopbs_key: WopbsKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu,
}

#[derive(Clone, Debug)]
pub struct RadixParams {
    pub num_blocks: usize,
    pub message_bits: u32,
    pub total_bits: u32,
}

impl RadixParams {
    pub fn q6() -> Self {
        Self {
            num_blocks: 3,
            message_bits: 2,
            total_bits: 6,
        }
    }

    pub fn q16() -> Self {
        Self::q6()
    }
}

#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct FheContext {
    pub server_key: Arc<CudaServerKey>,
    pub wopbs_key: Arc<WopbsKey>,
    pub streams: Arc<CudaStreams>,
    pub params: RadixParams,
    pub max_noise_budget: u32,
    pub quant_config: QuantConfig,
    pub backend: ComputeBackend,
}

#[cfg(not(feature = "gpu"))]
#[derive(Clone)]
pub struct FheContext {
    pub server_key: Arc<ServerKey>,
    pub wopbs_key: Arc<WopbsKey>,
    pub params: RadixParams,
    pub max_noise_budget: u32,
    pub quant_config: QuantConfig,
    pub backend: ComputeBackend,
}

impl std::fmt::Debug for FheContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FheContext")
            .field("backend", &self.backend)
            .field("params", &self.params)
            .field("max_noise_budget", &self.max_noise_budget)
            .field("quant_config", &self.quant_config)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "gpu")]
impl FheContext {
    // GPU-enabled builds keep the CUDA server key path available for heavy training workloads.
    pub fn new(
        server_key: CudaServerKey,
        wopbs_key: WopbsKey,
        streams: Arc<CudaStreams>,
        params: RadixParams,
        max_noise_budget: u32,
        quant_config: QuantConfig,
    ) -> Self {
        Self {
            server_key: Arc::new(server_key),
            wopbs_key: Arc::new(wopbs_key),
            streams,
            params,
            max_noise_budget,
            quant_config,
            backend: ComputeBackend::Gpu,
        }
    }

    pub fn radix_q16f8(
        server_key: CudaServerKey,
        wopbs_key: WopbsKey,
        streams: Arc<CudaStreams>,
    ) -> Self {
        Self::new(
            server_key,
            wopbs_key,
            streams,
            RadixParams::q16(),
            4,
            QuantConfig::q16f8(),
        )
    }

    pub fn streams(&self) -> &CudaStreams {
        self.streams.as_ref()
    }

    pub fn backend_name(&self) -> &'static str {
        "gpu"
    }

    // In a GPU build, this entry point intentionally initializes the CUDA backend for training.
    // CPU-only inference should use the default non-gpu build to avoid CUDA startup overhead.
    pub fn generate_keys_q16f8() -> (RadixClientKey, Arc<Self>) {
        let params = RadixParams::q16();
        let streams = default_cuda_streams();
        let client_key = ClientKey::new(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128);
        let radix_client_key = RadixClientKey::from((client_key.clone(), params.num_blocks));
        let server_key = CudaServerKey::new(&client_key, streams.as_ref());
        let host_server_key = ServerKey::new_radix_server_key(&client_key);
        let wopbs_key = WopbsKey::new_wopbs_key(
            &radix_client_key,
            &host_server_key,
            &LEGACY_WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        );
        let ctx = Arc::new(Self::new(
            server_key,
            wopbs_key,
            streams,
            params,
            4,
            QuantConfig::q16f8(),
        ));
        (radix_client_key, ctx)
    }

    pub fn load_or_generate(
        cache_path: &Path,
    ) -> Result<(RadixClientKey, Arc<Self>), std::io::Error> {
        let params = RadixParams::q16();
        let quant_config = QuantConfig::q16f8();
        let streams = default_cuda_streams();

        if cache_path.exists() {
            let bytes = fs::read(cache_path)?;
            let cached: CachedFheKeys = bincode::deserialize(&bytes).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("failed to deserialize cached FHE keys: {error}"),
                )
            })?;
            let cuda_server_key = CudaServerKey::new(cached.client_key.as_ref(), streams.as_ref());
            let ctx = Arc::new(Self::new(
                cuda_server_key,
                cached.wopbs_key,
                streams,
                params,
                4,
                quant_config,
            ));
            return Ok((cached.client_key, ctx));
        }

        let client_key = ClientKey::new(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128);
        let radix_client_key = RadixClientKey::from((client_key.clone(), params.num_blocks));
        let host_server_key = ServerKey::new_radix_server_key(&client_key);
        let wopbs_key = WopbsKey::new_wopbs_key(
            &radix_client_key,
            &host_server_key,
            &LEGACY_WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        );
        let cache_blob = CachedFheKeys {
            client_key: radix_client_key.clone(),
            server_key: host_server_key,
            wopbs_key: wopbs_key.clone(),
        };
        let serialized = bincode::serialize(&cache_blob).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to serialize generated FHE keys: {error}"),
            )
        })?;
        fs::write(cache_path, serialized)?;

        let cuda_server_key = CudaServerKey::new(&client_key, streams.as_ref());
        let ctx = Arc::new(Self::new(
            cuda_server_key,
            wopbs_key,
            streams,
            params,
            4,
            quant_config,
        ));
        Ok((radix_client_key, ctx))
    }
}

#[cfg(not(feature = "gpu"))]
impl FheContext {
    // CPU-only builds use the standard tfhe-rs ServerKey path. This is the default build mode
    // for local testing and for single-vector inference where GPU setup would add overhead.
    pub fn new(
        server_key: ServerKey,
        wopbs_key: WopbsKey,
        params: RadixParams,
        max_noise_budget: u32,
        quant_config: QuantConfig,
    ) -> Self {
        Self {
            server_key: Arc::new(server_key),
            wopbs_key: Arc::new(wopbs_key),
            params,
            max_noise_budget,
            quant_config,
            backend: ComputeBackend::Cpu,
        }
    }

    pub fn radix_q16f8(server_key: ServerKey, wopbs_key: WopbsKey) -> Self {
        Self::new(
            server_key,
            wopbs_key,
            RadixParams::q16(),
            4,
            QuantConfig::q16f8(),
        )
    }

    pub fn backend_name(&self) -> &'static str {
        "cpu"
    }

    pub fn generate_keys_q16f8() -> (RadixClientKey, Arc<Self>) {
        let params = RadixParams::q16();
        let client_key = ClientKey::new(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128);
        let radix_client_key = RadixClientKey::from((client_key.clone(), params.num_blocks));
        let server_key = ServerKey::new_radix_server_key(&client_key);
        let wopbs_key = WopbsKey::new_wopbs_key(
            &radix_client_key,
            &server_key,
            &LEGACY_WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        );
        let ctx = Arc::new(Self::new(
            server_key,
            wopbs_key,
            params,
            4,
            QuantConfig::q16f8(),
        ));
        (radix_client_key, ctx)
    }

    pub fn load_or_generate(
        cache_path: &Path,
    ) -> Result<(RadixClientKey, Arc<Self>), std::io::Error> {
        let params = RadixParams::q16();
        let quant_config = QuantConfig::q16f8();

        if cache_path.exists() {
            let bytes = fs::read(cache_path)?;
            let cached: CachedFheKeys = bincode::deserialize(&bytes).map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("failed to deserialize cached FHE keys: {error}"),
                )
            })?;
            let ctx = Arc::new(Self::new(
                cached.server_key,
                cached.wopbs_key,
                params,
                4,
                quant_config,
            ));
            return Ok((cached.client_key, ctx));
        }

        let client_key = ClientKey::new(PARAM_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M128);
        let radix_client_key = RadixClientKey::from((client_key.clone(), params.num_blocks));
        let server_key = ServerKey::new_radix_server_key(&client_key);
        let wopbs_key = WopbsKey::new_wopbs_key(
            &radix_client_key,
            &server_key,
            &LEGACY_WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        );
        let cache_blob = CachedFheKeys {
            client_key: radix_client_key.clone(),
            server_key: server_key.clone(),
            wopbs_key: wopbs_key.clone(),
        };
        let serialized = bincode::serialize(&cache_blob).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("failed to serialize generated FHE keys: {error}"),
            )
        })?;
        fs::write(cache_path, serialized)?;

        let ctx = Arc::new(Self::new(server_key, wopbs_key, params, 4, quant_config));
        Ok((radix_client_key, ctx))
    }
}

#[cfg(feature = "gpu")]
fn default_cuda_streams() -> Arc<CudaStreams> {
    let gpu_count = get_number_of_gpus();
    assert!(
        gpu_count > 0,
        "tfhe CUDA backend requested, but no CUDA GPU was detected"
    );

    let gpu_index = std::env::var("TFHE_RS_GPU_INDEX")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(0);

    assert!(
        gpu_index < gpu_count,
        "requested CUDA device {gpu_index}, but only {gpu_count} GPU(s) were detected"
    );

    Arc::new(CudaStreams::new_single_gpu(GpuIndex::new(gpu_index)))
}
