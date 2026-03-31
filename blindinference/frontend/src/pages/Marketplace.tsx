import { useEffect, useState } from 'react';
import { formatUnits } from 'ethers';
import { Card, Button } from '../components/UI';
import { Database, Cpu, ArrowRight, Wallet } from 'lucide-react';
import { motion } from 'motion/react';
import { Model } from '../services/fheService';
import { useWeb3 } from '../hooks/useWeb3';

type RegistryModel = Model & {
  inferenceFeeRaw: bigint;
  ipfsHash: string;
};

const FALLBACK_MODEL_ID = BigInt(import.meta.env.VITE_DEFAULT_MODEL_ID ?? '1');

export default function Marketplace({ onSelectModel }: { onSelectModel: (model: Model) => void }) {
  const { address, role, connect, contracts, paymentTokenName, paymentTokenDecimals } = useWeb3();
  const [models, setModels] = useState<RegistryModel[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let isActive = true;

    async function loadModels() {
      if (!contracts.modelRegistry) {
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        let modelIds: bigint[] = [];

        try {
          const modelCount = (await contracts.modelRegistry.modelCount()) as bigint;
          if (modelCount > 0n) {
            modelIds = Array.from({ length: Number(modelCount) }, (_, index) => BigInt(index + 1));
          }
        } catch {
          modelIds = [FALLBACK_MODEL_ID];
        }

        const nextModels: RegistryModel[] = [];

        for (const modelId of modelIds) {
          try {
            const [inferenceFee, labAddress, ipfsHash] = await Promise.all([
              contracts.modelRegistry.getInferenceFee(modelId) as Promise<bigint>,
              contracts.modelRegistry.getLabWallet(modelId) as Promise<string>,
              contracts.modelRegistry.getIpfsHash(modelId) as Promise<string>,
            ]);

            const formattedFee = Number(formatUnits(inferenceFee, paymentTokenDecimals));
            nextModels.push({
              id: `MOD-${modelId.toString().padStart(3, '0')}`,
              modelId,
              name: ipfsHash && ipfsHash !== '' ? ipfsHash : `Encrypted Model #${modelId.toString()}`,
              price: formattedFee,
              labAddress,
              accuracy: 'Private',
              inferenceFeeRaw: inferenceFee,
              ipfsHash,
            });
          } catch {
            continue;
          }
        }

        if (isActive) {
          setModels(nextModels);
        }
      } catch (loadError) {
        if (isActive) {
          setError(loadError instanceof Error ? loadError.message : 'Failed to load models from registry.');
        }
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadModels();

    return () => {
      isActive = false;
    };
  }, [contracts.modelRegistry]);

  return (
    <div className="space-y-12">
      <div className="flex flex-col md:flex-row items-end justify-between gap-6">
        <div className="space-y-4">
          <div className="text-[10px] font-bold uppercase tracking-[0.4em] text-[var(--accent-cyan)]">Registry // v2.0</div>
          <h1 className="text-5xl font-black tracking-tighter uppercase">Model <span className="neon-text">Marketplace</span></h1>
          <p className="text-[var(--text-muted)] max-w-md">
            {role === 'ai_lab'
              ? 'Inspect the live marketplace from the supply side and preview how Data Sources will discover and consume registered encrypted models.'
              : 'Browse registered AI labs and their encrypted models. Data Sources pay only when they request private inference.'}
          </p>
        </div>
        <div className="flex gap-4">
          <div className="flex items-center gap-3 px-6 py-3 bg-white/5 rounded-full border border-white/10">
            <Database className="w-4 h-4 text-[var(--accent-cyan)]" />
            <span className="text-xs font-mono font-bold">{models.length} DISCOVERED MODELS</span>
          </div>
        </div>
      </div>

      {!address && (
        <Card className="flex items-center justify-between gap-4">
          <div>
            <div className="font-bold">Connect your wallet to read the private marketplace registry</div>
            <div className="text-sm text-[var(--text-muted)]">The current frontend reads registered models through your injected Sepolia wallet.</div>
          </div>
          <Button onClick={() => void connect()}>
            <Wallet className="w-4 h-4 mr-2" />
            Connect Wallet
          </Button>
        </Card>
      )}

      {isLoading && (
        <Card className="text-sm text-[var(--text-muted)]">Loading models from ModelRegistry...</Card>
      )}

      {error && (
        <Card className="text-sm text-rose-300 border-rose-500/30 bg-rose-500/10">{error}</Card>
      )}

      {!isLoading && models.length > 0 && (
        <div className="space-y-4">
          <div className="grid grid-cols-5 px-8 py-4 text-[10px] font-bold uppercase tracking-[0.2em] text-white/30">
            <div className="col-span-2">Model Identifier</div>
            <div>Inference Fee</div>
            <div>AI Lab</div>
            <div className="text-right">Action</div>
          </div>

          {models.map((model, i) => (
            <motion.div
              key={(model.modelId?.toString() ?? model.id) || `market-model-${i}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08 }}
            >
              <Card className="grid grid-cols-5 items-center hover:bg-white/[0.04] transition-all py-6 group cursor-pointer">
                <div className="col-span-2 flex items-center gap-6">
                  <div className="w-12 h-12 rounded-2xl bg-white/5 flex items-center justify-center border border-white/10 group-hover:border-[var(--accent-cyan)]/50 transition-colors">
                    <Cpu className="w-6 h-6 text-[var(--accent-cyan)]" />
                  </div>
                  <div>
                    <div className="font-bold text-lg group-hover:text-[var(--accent-cyan)] transition-colors">{model.name}</div>
                    <div className="text-[10px] font-mono text-white/30 uppercase tracking-widest">
                      {model.id} // on-chain id {model.modelId?.toString()}
                    </div>
                  </div>
                </div>
                <div className="font-mono text-[var(--accent-cyan)] font-bold">{formatUnits(model.inferenceFeeRaw, paymentTokenDecimals)} {paymentTokenName}</div>
                <div className="text-[10px] font-mono text-white/40 uppercase tracking-widest">{model.labAddress}</div>
                <div className="text-right">
                  <Button variant="outline" size="sm" onClick={() => onSelectModel(model)}>
                    {role === 'ai_lab' ? 'Preview Flow' : 'Request Inference'} <ArrowRight className="ml-2 w-3 h-3" />
                  </Button>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
      )}

      {!isLoading && !error && address && models.length === 0 && (
        <Card className="text-sm text-[var(--text-muted)]">No active models were found in the current registry yet.</Card>
      )}
    </div>
  );
}
