import { useEffect, useState } from 'react';
import { Card, Button } from '../components/UI';
import { useWeb3 } from '../hooks/useWeb3';
import { downloadDatasetArtifact, getDatasetCatalog, type DatasetManifest } from '../services/workspaceService';
import {
  Database,
  Download,
  FileDigit,
  Loader2,
  Network,
  ShieldAlert,
  ShieldCheck,
  Sigma,
  TableProperties,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';

function truncateAddress(address: string) {
  return address.length > 14 ? `${address.slice(0, 10)}...${address.slice(-4)}` : address;
}

function shortHash(hash?: string | null) {
  if (!hash) {
    return 'Pending';
  }

  return `${hash.slice(0, 10)}...${hash.slice(-6)}`;
}

export default function LabDatasetsWorkspace({ onOpenModels }: { onOpenModels?: (datasetId: string) => void }) {
  const { role, jwt, paymentTokenName } = useWeb3();
  const [datasets, setDatasets] = useState<DatasetManifest[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeDownloadId, setActiveDownloadId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (!jwt || role !== 'ai_lab') {
      return;
    }

    let isActive = true;

    async function loadCatalog() {
      setIsLoading(true);
      setError(null);
      try {
        const nextDatasets = await getDatasetCatalog(jwt);
        if (!isActive) {
          return;
        }
        setDatasets(nextDatasets);
      } catch (loadError) {
        if (!isActive) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : 'Failed to load encrypted dataset catalog');
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadCatalog();

    return () => {
      isActive = false;
    };
  }, [jwt, role]);

  if (role !== 'ai_lab') {
    return (
      <div className="mx-auto flex max-w-3xl flex-col items-center justify-center py-24 text-center">
        <div className="mb-6 rounded-full border border-rose-500/20 bg-rose-500/10 p-5">
          <ShieldAlert className="h-10 w-10 text-rose-300" />
        </div>
        <h1 className="text-3xl font-black uppercase tracking-tight text-rose-200">AI Lab Access Only</h1>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-[var(--text-muted)]">
          The datasets workspace is reserved for AI Lab wallets so they can discover and download PPML-compatible encrypted training artifacts.
        </p>
      </div>
    );
  }

  const handleDownload = async (dataset: DatasetManifest) => {
    if (!jwt) {
      setError('Authenticate your AI lab wallet before downloading datasets.');
      return;
    }

    setActiveDownloadId(dataset.dataset_id);
    setError(null);
    setSuccess(null);

    try {
      await downloadDatasetArtifact(dataset.file_id, dataset.filename, jwt);
      setSuccess(`Downloaded ${dataset.filename}`);
    } catch (downloadError) {
      setError(downloadError instanceof Error ? downloadError.message : 'Failed to download dataset');
    } finally {
      setActiveDownloadId(null);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center gap-4">
        <div className="rounded-xl bg-[var(--accent-cyan)]/10 p-3">
          <Database className="h-8 w-8 text-[var(--accent-cyan)]" />
        </div>
        <div>
          <h1 className="text-3xl font-bold tracking-tight neon-text">Encrypted Datasets</h1>
          <p className="text-[var(--text-muted)]">
            Browse PPML-compatible encrypted datasets, inspect tensor metadata, and download artifacts for offline training.
          </p>
        </div>
      </div>

      <div className="grid gap-8 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <div className="flex items-center justify-between">
            <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
              Dataset Catalog
            </div>
            <div className="text-xs font-mono text-white/40">
              {isLoading ? 'syncing...' : `${datasets.length} artifacts`}
            </div>
          </div>

          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {datasets.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-white/10 p-6 text-sm text-[var(--text-muted)] md:col-span-2">
                No encrypted datasets are available yet. Data sources need to upload CSV datasets first.
              </div>
            ) : (
              datasets.map((dataset) => (
                <div key={dataset.dataset_id} className="rounded-3xl border border-white/10 bg-white/[0.02] p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-bold text-white">{dataset.original_filename ?? dataset.filename}</div>
                      <div className="mt-1 text-xs uppercase tracking-widest text-[var(--accent-cyan)]">
                        {dataset.artifact_type ?? 'ppml_encrypted_dataset'}
                      </div>
                    </div>
                    <div className="rounded-full bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      {dataset.status}
                    </div>
                  </div>

                  <div className="mt-5 grid gap-3 text-xs text-[var(--text-muted)]">
                    <div className="flex items-center gap-2">
                      <TableProperties className="h-4 w-4 text-[var(--accent-cyan)]" />
                      <span>{dataset.row_count ?? 0} rows // {dataset.feature_count ?? 0} features // {dataset.label_count ?? 0} label columns</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Sigma className="h-4 w-4 text-[var(--accent-cyan)]" />
                      <span>Q{dataset.quantization?.total_bits ?? '?'}f{dataset.quantization?.frac_bits ?? '?'} // scale {dataset.quantization?.scale ?? '?'}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <FileDigit className="h-4 w-4 text-[var(--accent-cyan)]" />
                      <span>Owner {truncateAddress(dataset.owner_address)}</span>
                    </div>
                  </div>

                  <div className="mt-5 rounded-2xl border border-white/10 bg-black/10 p-4 text-xs text-white/45">
                    <div className="font-mono">SHA {shortHash(dataset.artifact_sha256)}</div>
                    <div className="mt-2">Label {dataset.label_name ?? dataset.label_column_index ?? 'n/a'}</div>
                    <div className="mt-1">
                      Feature tensor {dataset.tensor_artifacts?.features.rows ?? 0}x{dataset.tensor_artifacts?.features.cols ?? 0}
                    </div>
                    <div className="mt-1">
                      Label tensor {dataset.tensor_artifacts?.labels.rows ?? 0}x{dataset.tensor_artifacts?.labels.cols ?? 0}
                    </div>
                  </div>

                  <div className="mt-4 rounded-2xl border border-white/10 bg-white/[0.02] p-4">
                    <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      <Network className="h-4 w-4" />
                      Trained Models
                    </div>
                    <div className="mt-3 text-xs text-[var(--text-muted)]">
                      {dataset.linked_model_count ?? 0} models linked to this dataset
                    </div>
                    <div className="mt-3 space-y-2">
                      {(dataset.linked_models ?? []).length === 0 ? (
                        <div className="text-xs text-white/40">
                          No AI lab has uploaded a trained model for this dataset yet.
                        </div>
                      ) : (
                        (dataset.linked_models ?? []).slice(0, 4).map((model) => (
                          <div
                            key={model.model_id}
                            className="rounded-xl border border-white/10 bg-black/10 px-3 py-2"
                          >
                            <div className="text-xs font-bold text-white">{model.name}</div>
                            <div className="mt-1 text-[10px] uppercase tracking-widest text-white/45">
                              {truncateAddress(model.lab_address)} {model.price_bfhe ? `// ${model.price_bfhe} ${paymentTokenName}` : ''} {model.on_chain_model_id ? `// On-chain ${model.on_chain_model_id}` : ''}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>

                  <div className="mt-5 flex items-center justify-between gap-4">
                    <div className="text-xs text-[var(--text-muted)]">
                      {dataset.notes ?? 'No additional dataset notes were provided.'}
                    </div>
                    <div className="flex items-center gap-2">
                      {onOpenModels && (
                        <Button
                          type="button"
                          size="sm"
                          variant="secondary"
                          onClick={() => onOpenModels(dataset.dataset_id)}
                        >
                          Use In Models
                        </Button>
                      )}
                      <Button
                        type="button"
                        size="sm"
                        isLoading={activeDownloadId === dataset.dataset_id}
                        onClick={() => void handleDownload(dataset)}
                      >
                        <Download className="mr-2 h-4 w-4" />
                        Download
                      </Button>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        <div className="space-y-6">
          <Card className="bg-[var(--bg-secondary)]/30 border-dashed">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">Training Hand-Off</h3>
            <div className="mt-4 space-y-3 text-sm text-[var(--text-muted)]">
              <p>Each artifact is already encrypted into PPML-native `EncryptedTensor` bytes.</p>
              <p>Download the JSON artifact, then feed the feature and label tensors into your PPML training pipeline.</p>
              <p>The dataset metadata preserves row count, quantization profile, and label provenance for reproducible training.</p>
            </div>
          </Card>

          <AnimatePresence>
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-lg border border-white/10 bg-white/[0.03] p-4 text-sm text-white/70"
              >
                <Loader2 className="mr-2 inline h-4 w-4 animate-spin" />
                Loading dataset catalog...
              </motion.div>
            )}
            {success && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-200"
              >
                <ShieldCheck className="mr-2 inline h-4 w-4" />
                {success}
              </motion.div>
            )}
            {error && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200"
              >
                <ShieldAlert className="mr-2 inline h-4 w-4" />
                {error}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
