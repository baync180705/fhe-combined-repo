import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from 'react';
import { Card, Button, Input } from '../components/UI';
import { useWeb3 } from '../hooks/useWeb3';
import {
  getDatasetCatalog,
  getLabModelArtifacts,
  publishModelRecord,
  type DatasetManifest,
  type TrainedModelRecord,
} from '../services/workspaceService';
import {
  Beaker,
  FileUp,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Network,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import {
  parseUploadedWeightsArtifact,
  registerEncryptedModel,
  toPriceUnits,
  type UploadedWeightsArtifact,
} from '../services/fheService';

function truncateAddress(address: string) {
  return address.length > 14 ? `${address.slice(0, 10)}...${address.slice(-4)}` : address;
}

export default function LabModelsWorkspace({ initialDatasetId }: { initialDatasetId?: string | null }) {
  const {
    address,
    role,
    jwt,
    connect,
    fhenixClient,
    ensureFhenixClient,
    contracts,
    isInitializingFhe,
    paymentTokenName,
  } = useWeb3();
  const [datasets, setDatasets] = useState<DatasetManifest[]>([]);
  const [models, setModels] = useState<TrainedModelRecord[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [price, setPrice] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [parsedArtifact, setParsedArtifact] = useState<UploadedWeightsArtifact | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isPublishing, setIsPublishing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (!address || !jwt || role !== 'ai_lab') {
      return;
    }

    let isActive = true;

    async function loadWorkspace() {
      setIsLoading(true);
      setError(null);
      try {
        const [nextDatasets, nextModels] = await Promise.all([
          getDatasetCatalog(jwt),
          getLabModelArtifacts(address, jwt),
        ]);

        if (!isActive) {
          return;
        }

        setDatasets(nextDatasets);
        setModels(nextModels);
        if (nextDatasets.length > 0 && selectedDatasetId === '') {
          setSelectedDatasetId(nextDatasets[0].dataset_id);
        }
      } catch (loadError) {
        if (!isActive) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : 'Failed to load model workspace');
      } finally {
        if (isActive) {
          setIsLoading(false);
        }
      }
    }

    void loadWorkspace();

    return () => {
      isActive = false;
    };
  }, [address, jwt, role, selectedDatasetId]);

  useEffect(() => {
    if (initialDatasetId) {
      setSelectedDatasetId(initialDatasetId);
    }
  }, [initialDatasetId]);

  const datasetMap = useMemo(
    () => new Map(datasets.map((dataset) => [dataset.dataset_id, dataset])),
    [datasets],
  );

  const selectedDataset = selectedDatasetId === '' ? null : datasetMap.get(selectedDatasetId) ?? null;

  if (role !== 'ai_lab') {
    return (
      <div className="mx-auto flex max-w-3xl flex-col items-center justify-center py-24 text-center">
        <div className="mb-6 rounded-full border border-rose-500/20 bg-rose-500/10 p-5">
          <ShieldAlert className="h-10 w-10 text-rose-300" />
        </div>
        <h1 className="text-3xl font-black uppercase tracking-tight text-rose-200">AI Lab Access Only</h1>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-[var(--text-muted)]">
          The models workspace is reserved for AI Lab wallets so they can publish encrypted model artifacts with explicit dataset lineage.
        </p>
      </div>
    );
  }

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setFile(nextFile);
    setParsedArtifact(null);
    setError(null);
    setSuccess(null);

    if (!nextFile) {
      return;
    }

    try {
      const artifact = await parseUploadedWeightsArtifact(nextFile);
      setParsedArtifact(artifact);
      if (name.trim() === '' && artifact.name) {
        setName(artifact.name);
      }
    } catch (parseError) {
      setFile(null);
      setParsedArtifact(null);
      setError(
        parseError instanceof Error
          ? parseError.message
          : 'Failed to read the weights JSON artifact',
      );
      event.target.value = '';
    }
  };

  const handlePublish = async (event: FormEvent) => {
    event.preventDefault();
    if (!jwt) {
      setError('Authenticate your AI lab wallet before publishing a model.');
      return;
    }
    if (selectedDatasetId === '') {
      setError('Select the dataset used to train this model.');
      return;
    }
    if (!file || !parsedArtifact) {
      setError('Choose a valid weights-and-bias JSON artifact before publishing.');
      return;
    }
    if (name.trim() === '') {
      setError('Provide a model name.');
      return;
    }

    setIsPublishing(true);
    setError(null);
    setSuccess(null);

    try {
      let activeClient = fhenixClient;
      let activeRegistry = contracts.modelRegistry;
      let activeAddress = address;

      if (!activeAddress) {
        const session = await connect();
        activeClient = session?.fhenixClient ?? activeClient;
        activeRegistry = session?.contracts.modelRegistry ?? activeRegistry;
        activeAddress = session?.address ?? activeAddress;
      }

      if (!activeClient) {
        activeClient = await ensureFhenixClient();
      }

      if (!activeRegistry) {
        throw new Error('ModelRegistry contract is not configured');
      }

      if (!activeAddress) {
        throw new Error('AI lab wallet is not connected');
      }

      const labRecord = await activeRegistry.aiLabs(activeAddress);
      const isLabRegistered =
        typeof labRecord?.isRegistered === 'boolean'
          ? labRecord.isRegistered
          : Boolean(Array.isArray(labRecord) ? labRecord[1] : false);

      if (!isLabRegistered) {
        throw new Error('Activate this AI Lab on-chain in the Lab workspace before publishing models.');
      }

      const modelName = name.trim();
      const registryReference = parsedArtifact.name?.trim() || modelName;
      const { modelId, receipt } = await registerEncryptedModel({
        client: activeClient,
        modelRegistry: activeRegistry,
        weights: parsedArtifact.weights,
        bias: parsedArtifact.bias,
        pricePerQuery: toPriceUnits(price),
        registryReference,
      });

      const record = await publishModelRecord(jwt, {
        dataset_id: selectedDatasetId,
        name: modelName,
        description,
        price_bfhe: price || undefined,
        status: 'published_on_chain',
        on_chain_model_id: modelId.toString(),
        tx_hash: receipt.hash,
        original_filename: parsedArtifact.originalFilename,
        artifact_sha256: parsedArtifact.artifactSha256,
        metadata_uri: parsedArtifact.metadataUri,
        registry_reference: registryReference,
        weight_count: parsedArtifact.weights.length,
        feature_names: parsedArtifact.features,
        scale: parsedArtifact.scale,
      });

      setModels((prev) => [record, ...prev]);
      setFile(null);
      setParsedArtifact(null);
      setName('');
      setDescription('');
      setPrice('');
      setSuccess(
        `Registered ${record.name} on-chain as model ${record.on_chain_model_id} and linked it to its training dataset.`,
      );
    } catch (publishError) {
      setError(
        publishError instanceof Error ? publishError.message : 'Failed to publish encrypted model',
      );
    } finally {
      setIsPublishing(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center gap-4">
        <div className="rounded-xl bg-[var(--accent-cyan)]/10 p-3">
          <Beaker className="h-8 w-8 text-[var(--accent-cyan)]" />
        </div>
        <div>
          <h1 className="text-3xl font-bold tracking-tight neon-text">Model Upload</h1>
          <p className="text-[var(--text-muted)]">
            Publish encrypted model artifacts and make their training provenance explicit by linking each upload to a dataset.
          </p>
        </div>
      </div>

      <div className="grid gap-8 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <Card>
            <form onSubmit={handlePublish} className="space-y-6">
              <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                <Sparkles className="h-4 w-4" />
                Publish Encrypted Model
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                    Training Dataset
                  </label>
                  <select
                    value={selectedDatasetId}
                    onChange={(event) => setSelectedDatasetId(event.target.value)}
                    className="w-full rounded-full border border-white/10 bg-white/5 px-6 py-3 text-sm text-white focus:border-[var(--accent-cyan)] focus:outline-none"
                    required
                  >
                    <option value="" disabled>
                      Select dataset
                    </option>
                    {datasets.map((dataset) => (
                      <option key={dataset.dataset_id} value={dataset.dataset_id} className="bg-slate-900 text-white">
                        {(dataset.original_filename ?? dataset.filename)} | {dataset.row_count ?? 0} rows
                      </option>
                    ))}
                  </select>
                </div>

                <Input
                  label="Model Name"
                  type="text"
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="Encrypted Diabetes Risk Model"
                  required
                />

                <Input
                  label={`Price (${paymentTokenName})`}
                  type="text"
                  value={price}
                  onChange={(event) => setPrice(event.target.value)}
                  placeholder="0.5"
                />
              </div>

              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                  Description
                </label>
                <textarea
                  className="min-h-28 w-full rounded-3xl border border-white/10 bg-white/5 px-6 py-4 text-sm text-white focus:border-[var(--accent-cyan)] focus:outline-none"
                  value={description}
                  onChange={(event) => setDescription(event.target.value)}
                  placeholder="Summarize what this encrypted model predicts and what dataset it was trained from."
                />
              </div>

              <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.02] p-6">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      Weights And Bias JSON
                    </div>
                    <div className="mt-2 text-sm text-[var(--text-muted)]">
                      Upload a JSON artifact containing quantized unsigned `weights` and `bias`. The frontend encrypts them with CoFHE and registers the model on-chain.
                    </div>
                  </div>
                  <label className="inline-flex cursor-pointer items-center rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-bold uppercase tracking-widest text-white transition-colors hover:border-[var(--accent-cyan)]/40">
                    <FileUp className="mr-2 h-4 w-4" />
                    {file ? file.name : 'Choose File'}
                    <input
                      type="file"
                      accept=".json,application/json"
                      className="hidden"
                      onChange={(event) => void handleFileChange(event)}
                    />
                  </label>
                </div>
              </div>

              {parsedArtifact && (
                <div className="rounded-3xl border border-white/10 bg-black/10 p-6">
                  <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                    <Network className="h-4 w-4" />
                    Parsed Artifact
                  </div>
                  <div className="mt-4 grid gap-3 text-sm text-[var(--text-muted)] md:grid-cols-2">
                    <div>Weights {parsedArtifact.weights.length}</div>
                    <div>Bias {parsedArtifact.bias}</div>
                    <div>Scale {parsedArtifact.scale ?? 'n/a'}</div>
                    <div>SHA {parsedArtifact.artifactSha256.slice(0, 12)}...{parsedArtifact.artifactSha256.slice(-8)}</div>
                  </div>
                  <div className="mt-4 text-xs text-white/45">
                    {parsedArtifact.features.slice(0, 6).join(' / ')}
                    {parsedArtifact.features.length > 6 ? ' / ...' : ''}
                  </div>
                </div>
              )}

              <div className="flex items-center justify-between border-t border-white/5 pt-4">
                <div className="text-xs text-[var(--text-muted)]">
                  The JSON never goes to the contract directly. The browser encrypts each parameter with CoFHE, submits the ciphertext handles on-chain, and then saves only provenance metadata to the backend.
                </div>
                <Button
                  type="submit"
                  isLoading={isPublishing || isInitializingFhe}
                  disabled={datasets.length === 0 || !parsedArtifact}
                >
                  {isPublishing
                    ? 'Publishing...'
                    : isInitializingFhe
                      ? 'Initializing FHE...'
                      : 'Encrypt And Publish'}
                </Button>
              </div>
            </form>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="bg-[var(--bg-secondary)]/30 border-dashed">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">Dataset Requirement</h3>
            <div className="mt-4 space-y-3 text-sm text-[var(--text-muted)]">
              <p>Each published model must reference exactly one dataset in this Wave 1 flow.</p>
              <p>The JSON artifact must contain quantized unsigned integers so the contract can store them as encrypted `euint32` parameters.</p>
              <p>After the tx succeeds, the backend persists only provenance metadata and the resulting on-chain model id.</p>
            </div>
          </Card>

          <Card className="bg-[var(--bg-secondary)]/30 border-dashed">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">Selected Dataset</h3>
            <div className="mt-4 space-y-2 text-sm text-[var(--text-muted)]">
              <p>{selectedDataset ? (selectedDataset.original_filename ?? selectedDataset.filename) : 'No dataset selected yet.'}</p>
              {selectedDataset && (
                <p>
                  {selectedDataset.row_count ?? 0} rows // {selectedDataset.feature_count ?? 0} features // Q
                  {selectedDataset.quantization?.total_bits ?? '?'}f{selectedDataset.quantization?.frac_bits ?? '?'}
                </p>
              )}
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
                Loading model workspace...
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

      <Card>
        <div className="flex items-center justify-between">
          <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
            Uploaded Models
          </div>
          <div className="text-xs font-mono text-white/40">{models.length} records</div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-2">
          {models.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-white/10 p-6 text-sm text-[var(--text-muted)] md:col-span-2">
              No encrypted models uploaded yet.
            </div>
          ) : (
            models.map((model) => {
              const dataset = datasetMap.get(model.dataset_id);
              return (
                <div key={model.model_id} className="rounded-3xl border border-white/10 bg-white/[0.02] p-6">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-bold text-white">{model.name}</div>
                      <div className="mt-1 text-xs uppercase tracking-widest text-[var(--accent-cyan)]">
                        {model.status}
                      </div>
                    </div>
                    <div className="rounded-full bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      {model.on_chain_model_id ? `On-chain ${model.on_chain_model_id}` : 'Metadata Only'}
                    </div>
                  </div>

                  <div className="mt-4 text-sm text-[var(--text-muted)]">
                    {model.description ?? 'No model description provided.'}
                  </div>

                  <div className="mt-4 grid gap-2 text-xs text-white/45">
                    <div>Dataset {(dataset?.original_filename ?? dataset?.filename) ?? model.dataset_id}</div>
                    <div>Owner {truncateAddress(model.lab_address)}</div>
                    <div>Source {model.original_filename ?? model.filename}</div>
                    <div>Weights {model.weight_count ?? 'n/a'} {model.scale ? `// scale ${model.scale}` : ''}</div>
                    <div>{model.price_bfhe ? `${model.price_bfhe} ${paymentTokenName}` : 'No price metadata'} {model.tx_hash ? `// Tx ${model.tx_hash.slice(0, 10)}...` : ''}</div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </Card>
    </div>
  );
}
