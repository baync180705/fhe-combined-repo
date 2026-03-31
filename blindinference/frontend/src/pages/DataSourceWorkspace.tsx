import { FormEvent, useEffect, useState } from 'react';
import { Card, Button, Input } from '../components/UI';
import { useWeb3 } from '../hooks/useWeb3';
import {
  encryptAndUploadDataset,
  getOutgoingDatasets,
  getOutgoingSubmissions,
  type DatasetManifest,
  type SubmissionRecord,
} from '../services/workspaceService';
import {
  Activity,
  Database,
  FileUp,
  Hash,
  Loader2,
  ShieldAlert,
  ShieldCheck,
  TableProperties,
  Upload,
} from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';

function formatShortHash(value?: string | null) {
  if (!value) {
    return 'Pending';
  }

  return `${value.slice(0, 12)}...${value.slice(-8)}`;
}

export default function DataSourceWorkspace() {
  const { address, role, jwt, paymentTokenName } = useWeb3();
  const [labelColumn, setLabelColumn] = useState('last');
  const [hasHeader, setHasHeader] = useState(true);
  const [notes, setNotes] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<DatasetManifest[]>([]);
  const [submissions, setSubmissions] = useState<SubmissionRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!address || !jwt || role !== 'data_source') {
      return;
    }

    let isActive = true;

    async function loadWorkspace() {
      setIsLoading(true);
      try {
        const [nextDatasets, nextSubmissions] = await Promise.all([
          getOutgoingDatasets(address, jwt),
          getOutgoingSubmissions(address, jwt),
        ]);

        if (!isActive) {
          return;
        }

        setDatasets(nextDatasets);
        setSubmissions(nextSubmissions);
      } catch (loadError) {
        if (!isActive) {
          return;
        }
        setError(loadError instanceof Error ? loadError.message : 'Failed to load upload workspace');
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
  }, [address, jwt, role]);

  if (role !== 'data_source') {
    return (
      <div className="mx-auto flex max-w-3xl flex-col items-center justify-center py-24 text-center">
        <div className="mb-6 rounded-full border border-rose-500/20 bg-rose-500/10 p-5">
          <ShieldAlert className="h-10 w-10 text-rose-300" />
        </div>
        <h1 className="text-3xl font-black uppercase tracking-tight text-rose-200">Data Source Access Only</h1>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-[var(--text-muted)]">
          This workspace is reserved for Data Source wallets. Use it to upload CSV datasets, let the backend encrypt them into PPML-compatible tensors, and track private inference requests.
        </p>
      </div>
    );
  }

  const handleUpload = async (event: FormEvent) => {
    event.preventDefault();
    if (!address || !jwt) {
      setError('Authenticate your wallet before uploading datasets.');
      return;
    }
    if (!file) {
      setError('Select a dataset file before uploading.');
      return;
    }
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setError('Upload a CSV dataset so the backend can split features and labels correctly.');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadSuccess(null);

    try {
      const manifest = await encryptAndUploadDataset(jwt, {
        file,
        label_column: labelColumn,
        has_header: hasHeader,
        notes,
      });

      setDatasets((prev) => [manifest, ...prev]);
      setFile(null);
      setNotes('');
      setUploadSuccess(`Dataset encrypted into a PPML-compatible artifact with ${manifest.row_count ?? 0} rows.`);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Failed to upload dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center gap-4">
        <div className="rounded-xl bg-[var(--accent-cyan)]/10 p-3">
          <Database className="h-8 w-8 text-[var(--accent-cyan)]" />
        </div>
        <div>
          <h1 className="text-3xl font-bold tracking-tight neon-text">Dataset Upload</h1>
          <p className="text-[var(--text-muted)]">
            Upload CSV datasets and let the backend encrypt them into PPML-compatible TFHE radix tensors for downstream lab training.
          </p>
        </div>
      </div>

      <div className="grid gap-8 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <Card>
            <form onSubmit={handleUpload} className="space-y-6">
              <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                <Upload className="h-4 w-4" />
                Dataset Encryption Intake
              </div>

              <div className="grid gap-6 md:grid-cols-2">
                <Input
                  label="Label Column"
                  type="text"
                  value={labelColumn}
                  onChange={(event) => setLabelColumn(event.target.value)}
                  placeholder="last, 0, outcome"
                  required
                />
                <div className="space-y-2">
                  <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                    CSV Header
                  </label>
                  <label className="flex h-[46px] items-center gap-3 rounded-full border border-white/10 bg-white/5 px-5 text-sm text-white">
                    <input
                      type="checkbox"
                      checked={hasHeader}
                      onChange={(event) => setHasHeader(event.target.checked)}
                      className="h-4 w-4 accent-[var(--accent-cyan)]"
                    />
                    First row contains column names
                  </label>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)]">
                  Notes
                </label>
                <textarea
                  className="min-h-28 w-full rounded-3xl border border-white/10 bg-white/5 px-6 py-4 text-sm text-white focus:border-[var(--accent-cyan)] focus:outline-none"
                  value={notes}
                  onChange={(event) => setNotes(event.target.value)}
                  placeholder="Optional metadata about the dataset schema, cohort, or intended training context."
                />
              </div>

              <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.02] p-6">
                <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                  <div>
                    <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      CSV Dataset
                    </div>
                    <div className="mt-2 text-sm text-[var(--text-muted)]">
                      Upload a UTF-8 CSV file. The backend will parse it, split features and labels, and store a PPML-native encrypted dataset artifact in GridFS.
                    </div>
                  </div>
                  <label className="inline-flex cursor-pointer items-center rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-bold uppercase tracking-widest text-white transition-colors hover:border-[var(--accent-cyan)]/40">
                    <FileUp className="mr-2 h-4 w-4" />
                    {file ? file.name : 'Choose File'}
                    <input
                      type="file"
                      accept=".csv,text/csv"
                      className="hidden"
                      onChange={(event) => setFile(event.target.files?.[0] ?? null)}
                    />
                  </label>
                </div>
              </div>

              <div className="flex items-center justify-between border-t border-white/5 pt-4">
                <div className="text-xs text-[var(--text-muted)]">
                  The backend encrypts with PPML-compatible TFHE radix keys, then stores only the encrypted artifact and metadata in Mongo/GridFS.
                </div>
                <Button type="submit" isLoading={isUploading}>
                  Encrypt And Upload
                </Button>
              </div>
            </form>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="bg-[var(--bg-secondary)]/30 border-dashed">
            <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--text-muted)]">Upload Rules</h3>
            <div className="mt-4 space-y-3 text-sm text-[var(--text-muted)]">
              <div className="flex items-start gap-3">
                <TableProperties className="mt-0.5 h-4 w-4 text-[var(--accent-cyan)]" />
                <span>CSV rows must be numeric and rectangular.</span>
              </div>
              <div className="flex items-start gap-3">
                <Hash className="mt-0.5 h-4 w-4 text-[var(--accent-cyan)]" />
                <span>The label column can be the last column, a zero-based index, or a named header.</span>
              </div>
              <div className="flex items-start gap-3">
                <Database className="mt-0.5 h-4 w-4 text-[var(--accent-cyan)]" />
                <span>Encrypted artifacts are published to the shared AI-lab dataset catalog after upload.</span>
              </div>
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
                Loading upload workspace...
              </motion.div>
            )}
            {uploadSuccess && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4 text-sm text-emerald-200"
              >
                <ShieldCheck className="mr-2 inline h-4 w-4" />
                {uploadSuccess}
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

      <div className="grid gap-8 xl:grid-cols-2">
        <Card>
          <div className="flex items-center justify-between">
            <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
              Uploaded Datasets
            </div>
            <div className="text-xs font-mono text-white/40">{datasets.length} records</div>
          </div>

          <div className="mt-6 space-y-4">
            {datasets.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-white/10 p-6 text-sm text-[var(--text-muted)]">
                No datasets uploaded yet.
              </div>
            ) : (
              datasets.map((dataset) => (
                <div key={dataset.dataset_id} className="rounded-2xl border border-white/10 bg-white/[0.02] p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-bold">{dataset.original_filename ?? dataset.filename}</div>
                      <div className="mt-1 text-xs text-[var(--text-muted)]">
                        {dataset.row_count ?? 0} rows // {dataset.feature_count ?? 0} features // label {dataset.label_name ?? dataset.label_column_index ?? 'n/a'}
                      </div>
                    </div>
                    <div className="rounded-full bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      {dataset.status}
                    </div>
                  </div>
                  <div className="mt-4 grid gap-2 text-xs text-white/40 md:grid-cols-2">
                    <div className="font-mono">File ID {dataset.file_id}</div>
                    <div className="font-mono">SHA {formatShortHash(dataset.artifact_sha256)}</div>
                    <div>Scheme {dataset.encryption_scheme ?? 'tfhe-rs-radix'}</div>
                    <div>Q{dataset.quantization?.total_bits ?? '?'}f{dataset.quantization?.frac_bits ?? '?'}</div>
                  </div>
                  <div className="mt-4 rounded-2xl border border-white/10 bg-black/10 p-4">
                    <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      Linked Models
                    </div>
                    <div className="mt-2 text-xs text-[var(--text-muted)]">
                      {dataset.linked_model_count ?? 0} trained models are linked to this dataset
                    </div>
                    <div className="mt-3 space-y-2">
                      {(dataset.linked_models ?? []).length === 0 ? (
                        <div className="text-xs text-white/40">
                          No AI lab has uploaded a trained model for this dataset yet.
                        </div>
                      ) : (
                        (dataset.linked_models ?? []).slice(0, 3).map((model) => (
                          <div key={model.model_id} className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2">
                            <div className="text-xs font-bold text-white">{model.name}</div>
                            <div className="mt-1 text-[10px] uppercase tracking-widest text-white/45">
                              {model.price_bfhe ? `${model.price_bfhe} ${paymentTokenName}` : 'Unpriced'} {model.on_chain_model_id ? `// On-chain ${model.on_chain_model_id}` : ''}
                            </div>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
              Inference Requests
            </div>
            <div className="text-xs font-mono text-white/40">{submissions.length} tracked</div>
          </div>

          <div className="mt-6 space-y-4">
            {submissions.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-white/10 p-6 text-sm text-[var(--text-muted)]">
                No inference requests tracked yet. Run blind inference from the marketplace flow to populate this list.
              </div>
            ) : (
              submissions.map((submission) => (
                <div key={submission.submission_id} className="rounded-2xl border border-white/10 bg-white/[0.02] p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-bold">Request {submission.request_id}</div>
                      <div className="mt-1 text-xs text-[var(--text-muted)]">
                        Model {submission.model_id} // Lab {submission.lab_address}
                      </div>
                    </div>
                    <div className="rounded-full bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      {submission.status}
                    </div>
                  </div>
                  {submission.plaintext_result && (
                    <div className="mt-4 flex items-center gap-2 text-sm text-emerald-300">
                      <Activity className="h-4 w-4" />
                      Result {submission.plaintext_result}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
