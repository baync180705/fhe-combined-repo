import { useEffect, useState } from 'react';
import { formatUnits } from 'ethers';
import { Card, Button, Input } from '../components/UI';
import { Beaker, Loader2, Shield, ShieldAlert, Sparkles, Wallet } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useWeb3 } from '../hooks/useWeb3';
import { getUserProfile } from '../services/profileService';
import { isIpfsUri } from '../services/ipfsProfileService';
import { getIncomingDatasets, getIncomingSubmissions, type DatasetManifest, type SubmissionRecord } from '../services/workspaceService';

export default function LabDashboard() {
  const { address, role, jwt, connect, contracts, paymentTokenName, paymentTokenDecimals } = useWeb3();
  const [profileURI, setProfileURI] = useState('');
  const [isActivatingLab, setIsActivatingLab] = useState(false);
  const [isCheckingLabStatus, setIsCheckingLabStatus] = useState(false);
  const [isLabRegisteredOnChain, setIsLabRegisteredOnChain] = useState(false);
  const [labActivationTxHash, setLabActivationTxHash] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [incomingDatasets, setIncomingDatasets] = useState<DatasetManifest[]>([]);
  const [incomingSubmissions, setIncomingSubmissions] = useState<SubmissionRecord[]>([]);
  const [labModels, setLabModels] = useState<Array<{ modelId: bigint; name: string; fee: bigint; ipfsHash: string }>>([]);
  const [isLoadingOperations, setIsLoadingOperations] = useState(false);

  useEffect(() => {
    if (!address || !jwt || role !== 'ai_lab') {
      return;
    }

    let isActive = true;

    async function hydrateProfile() {
      try {
        const profile = await getUserProfile(address, jwt);
        if (!isActive) {
          return;
        }

        if (isIpfsUri(profile.profile_uri)) {
          setProfileURI(profile.profile_uri);
        } else {
          setProfileURI('');
        }
      } catch (loadError) {
        if (!isActive) {
          return;
        }
        console.error('Failed to load AI lab profile:', loadError);
        setProfileURI('');
      }
    }

    void hydrateProfile();

    return () => {
      isActive = false;
    };
  }, [address, jwt, role]);

  useEffect(() => {
    if (!address || role !== 'ai_lab' || !contracts.modelRegistry) {
      setIsLabRegisteredOnChain(false);
      return;
    }

    let isActive = true;

    async function loadLabStatus() {
      setIsCheckingLabStatus(true);
      try {
        const labRecord = await contracts.modelRegistry?.aiLabs(address);
        const onChainRegistered =
          typeof labRecord?.isRegistered === 'boolean'
            ? labRecord.isRegistered
            : Boolean(Array.isArray(labRecord) ? labRecord[1] : false);

        const onChainProfileUri =
          typeof labRecord?.profileURI === 'string'
            ? labRecord.profileURI
            : typeof labRecord?.[0] === 'string'
              ? labRecord[0]
              : '';

        if (!isActive) {
          return;
        }

        setIsLabRegisteredOnChain(onChainRegistered);
        if (isIpfsUri(onChainProfileUri)) {
          setProfileURI(onChainProfileUri);
        }
      } catch (statusError) {
        if (!isActive) {
          return;
        }
        console.error('Failed to read on-chain AI lab status:', statusError);
      } finally {
        if (isActive) {
          setIsCheckingLabStatus(false);
        }
      }
    }

    void loadLabStatus();

    return () => {
      isActive = false;
    };
  }, [address, role, contracts.modelRegistry]);

  useEffect(() => {
    if (!address || !jwt || role !== 'ai_lab') {
      return;
    }

    let isActive = true;

    async function loadOperationsMetadata() {
      setIsLoadingOperations(true);
      try {
        const [datasets, submissions] = await Promise.all([
          getIncomingDatasets(address, jwt),
          getIncomingSubmissions(address, jwt),
        ]);

        if (!isActive) {
          return;
        }

        setIncomingDatasets(datasets);
        setIncomingSubmissions(submissions);
      } catch (operationsError) {
        if (!isActive) {
          return;
        }
        console.error('Failed to load incoming AI lab metadata:', operationsError);
      } finally {
        if (isActive) {
          setIsLoadingOperations(false);
        }
      }
    }

    void loadOperationsMetadata();

    return () => {
      isActive = false;
    };
  }, [address, jwt, role]);

  useEffect(() => {
    if (!address || role !== 'ai_lab' || !contracts.modelRegistry) {
      setLabModels([]);
      return;
    }

    let isActive = true;

    async function loadLabModels() {
      try {
        const modelCount = (await contracts.modelRegistry?.modelCount()) as bigint;
        const models: Array<{ modelId: bigint; name: string; fee: bigint; ipfsHash: string }> = [];

        for (let index = 1n; index <= modelCount; index += 1n) {
          try {
            const [labWallet, ipfsHash, fee] = await Promise.all([
              contracts.modelRegistry?.getLabWallet(index) as Promise<string>,
              contracts.modelRegistry?.getIpfsHash(index) as Promise<string>,
              contracts.modelRegistry?.getInferenceFee(index) as Promise<bigint>,
            ]);

            if (labWallet.toLowerCase() === address.toLowerCase()) {
              models.push({
                modelId: index,
                name: ipfsHash || `Encrypted Model #${index.toString()}`,
                fee,
                ipfsHash,
              });
            }
          } catch {
            continue;
          }
        }

        if (isActive) {
          setLabModels(models);
        }
      } catch (modelError) {
        if (isActive) {
          console.error('Failed to load AI lab models:', modelError);
        }
      }
    }

    void loadLabModels();

    return () => {
      isActive = false;
    };
  }, [address, role, contracts.modelRegistry]);

  if (role !== 'ai_lab') {
    return (
      <div className="mx-auto flex max-w-3xl flex-col items-center justify-center py-24 text-center">
        <div className="mb-6 rounded-full border border-rose-500/20 bg-rose-500/10 p-5">
          <ShieldAlert className="h-10 w-10 text-rose-300" />
        </div>
        <h1 className="text-3xl font-black uppercase tracking-tight text-rose-200">AI Lab Access Only</h1>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-[var(--text-muted)]">
          The Lab Dashboard is the supply-side workspace for registered AI Labs. Choose the AI Lab role from the onboarding shell to manage profiles, register encrypted models, and set inference pricing.
        </p>
      </div>
    );
  }

  const handleActivateLab = async () => {
    setError(null);
    setLabActivationTxHash(null);
    setIsActivatingLab(true);

    try {
      let activeRegistry = contracts.modelRegistry;
      let activeAddress = address;

      if (!activeAddress) {
        const session = await connect();
        activeRegistry = session?.contracts.modelRegistry ?? activeRegistry;
        activeAddress = session?.address ?? activeAddress;
      }

      if (!activeRegistry) {
        throw new Error('ModelRegistry contract is not configured');
      }

      if (!activeAddress) {
        throw new Error('AI lab wallet is not connected');
      }

      const trimmedProfileUri = profileURI.trim();
      if (trimmedProfileUri === '') {
        throw new Error('Save your profile to IPFS in the Profile workspace before on-chain activation.');
      }

      const labRecord = await activeRegistry.aiLabs(activeAddress);
      const alreadyRegistered =
        typeof labRecord?.isRegistered === 'boolean'
          ? labRecord.isRegistered
          : Boolean(Array.isArray(labRecord) ? labRecord[1] : false);

      if (!alreadyRegistered) {
        const activationTx = await activeRegistry.registerLab(trimmedProfileUri);
        const receipt = await activationTx.wait();
        setLabActivationTxHash(receipt?.hash ?? activationTx.hash ?? null);
      }

      setIsLabRegisteredOnChain(true);
    } catch (activationError) {
      setError(
        activationError instanceof Error
          ? activationError.message
          : 'Failed to activate AI lab on-chain.',
      );
    } finally {
      setIsActivatingLab(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex items-center gap-4 mb-8">
        <div className="p-3 bg-[var(--accent-cyan)]/10 rounded-xl">
          <Beaker className="w-8 h-8 text-[var(--accent-cyan)]" />
        </div>
        <div>
          <h1 className="text-3xl font-bold tracking-tight neon-text">Lab Dashboard</h1>
          <p className="text-[var(--text-muted)]">Encrypt and register a quantized logistic-regression model with the Fhenix network public key.</p>
        </div>
      </div>

      <Card className={isLabRegisteredOnChain ? 'border-emerald-500/20 bg-emerald-500/5' : 'border-amber-500/20 bg-amber-500/5'}>
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
              <Wallet className="h-4 w-4" />
              On-Chain AI Lab Activation
            </div>
            <h2 className="text-2xl font-black uppercase tracking-tight">
              {isLabRegisteredOnChain ? 'AI Lab Registered On Fhenix' : 'Activate Your AI Lab Identity'}
            </h2>

            <div className="text-xs font-mono text-white/50">
              Wallet: {address ?? 'Disconnected'}
            </div>
            {labActivationTxHash && (
              <div className="text-xs font-mono text-emerald-300 break-all">
                Activation Tx: {labActivationTxHash}
              </div>
            )}
          </div>

          <div className="w-full max-w-xl space-y-4">
            <Input
              label="AI Lab Profile URI"
              type="text"
              value={profileURI}
              placeholder="ipfs://..."
              readOnly
              disabled={isActivatingLab}
              required
            />
            <div className="flex items-center justify-between gap-4">
              <div className="text-xs text-[var(--text-muted)]">
                {isCheckingLabStatus
                  ? 'Checking on-chain registration state...'
                  : isLabRegisteredOnChain
                    ? 'This wallet can now register encrypted models.'
                    : 'Save your profile first, then complete this transaction once per AI Lab wallet.'}
              </div>
              <Button
                type="button"
                onClick={handleActivateLab}
                isLoading={isActivatingLab}
                disabled={isLabRegisteredOnChain || isCheckingLabStatus || profileURI.trim() === ''}
              >
                <Sparkles className="mr-2 h-4 w-4" />
                {isLabRegisteredOnChain ? 'Activated' : 'Activate On-Chain'}
              </Button>
            </div>
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        <div className="md:col-span-2">
          <Card>
            <div className="space-y-8">
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                  <Shield className="w-4 h-4" />
                  AI Lab Publish Flow
                </div>

                <div className="rounded-2xl border border-white/10 bg-white/[0.01] p-6 space-y-4">
                  <p className="text-lg font-bold">How this lab workspace is used</p>
                  <p className="text-sm text-[var(--text-muted)]">
                    Activate your AI lab identity on-chain once, then use the <span className="text-white">Datasets</span> tab to download encrypted datasets and the <span className="text-white">Models</span> tab to upload a weights-and-bias JSON, encrypt it with CoFHE in the browser, and register the resulting model on-chain.
                  </p>

                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="rounded-xl border border-white/10 bg-[var(--bg-secondary)]/40 p-4">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">1 // Activate</div>
                      <div className="mt-2 text-sm text-white">Bind this wallet as an AI lab on-chain with your profile URI.</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-[var(--bg-secondary)]/40 p-4">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">2 // Train</div>
                      <div className="mt-2 text-sm text-white">Download an encrypted dataset artifact and complete off-chain training.</div>
                    </div>
                    <div className="rounded-xl border border-white/10 bg-[var(--bg-secondary)]/40 p-4">
                      <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">3 // Publish</div>
                      <div className="mt-2 text-sm text-white">Publish the trained weights from the Models workspace with pricing and dataset provenance.</div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="space-y-4 pt-6 border-t border-white/5">
                <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                  <Sparkles className="w-4 h-4" />
                  Profile URI
                </div>
                <div className="max-w-xl rounded-3xl border border-white/10 bg-white/5 px-6 py-4 text-sm text-white/75 break-all">
                  {profileURI || 'No IPFS profile URI yet. Save your profile in the Profile workspace first.'}
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  This URI is only used for the one-time on-chain lab activation step above. The actual model publish flow now lives in the Models tab.
                </div>
              </div>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="bg-[var(--bg-secondary)]/30 border-dashed">
            <h3 className="text-xs font-bold uppercase tracking-widest mb-4 text-[var(--text-muted)]">Publish Guide</h3>
            <div className="space-y-4 text-xs text-[var(--text-muted)] leading-relaxed">
              <p>Model JSONs should carry quantized unsigned integer `weights` and `bias` values that fit into `uint32`.</p>
              <p>The browser encrypts those parameters with CoFHE before calling `ModelRegistry.registerModel(...)`.</p>
              <p>After the tx succeeds, the backend records the dataset linkage and on-chain model id for provenance.</p>
            </div>
          </Card>

          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-lg"
              >
                <div className="flex items-center gap-2 text-sm font-bold text-rose-300">
                  <Loader2 className="w-4 h-4" />
                  Lab Activation Failed
                </div>
                <div className="mt-2 text-[10px] text-rose-200/80">{error}</div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      <div className="grid gap-8 xl:grid-cols-2">
        <Card>
          <div className="flex items-center justify-between">
            <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
              Registered Models
            </div>
            <div className="text-xs font-mono text-white/40">{labModels.length} active</div>
          </div>

          <div className="mt-6 space-y-4">
            {labModels.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-white/10 p-6 text-sm text-[var(--text-muted)]">
                No models registered for this AI Lab wallet yet.
              </div>
            ) : (
              labModels.map((model) => (
                <div key={model.modelId.toString()} className="rounded-2xl border border-white/10 bg-white/[0.02] p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-bold">{model.name}</div>
                      <div className="mt-1 text-xs text-[var(--text-muted)]">Model ID {model.modelId.toString()}</div>
                    </div>
                    <div className="rounded-full bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      {formatUnits(model.fee, paymentTokenDecimals)} {paymentTokenName}
                    </div>
                  </div>
                  <div className="mt-4 text-xs font-mono text-white/40">
                    {model.ipfsHash || 'No profile URI supplied'}
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>

        <Card>
          <div className="flex items-center justify-between">
            <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
              Incoming Datasets
            </div>
            <div className="text-xs font-mono text-white/40">
              {isLoadingOperations ? 'syncing...' : `${incomingDatasets.length} manifests`}
            </div>
          </div>

          <div className="mt-6 space-y-4">
            {incomingDatasets.length === 0 ? (
              <div className="rounded-2xl border border-dashed border-white/10 p-6 text-sm text-[var(--text-muted)]">
                No incoming dataset manifests have been assigned to this AI Lab yet.
              </div>
            ) : (
              incomingDatasets.map((dataset) => (
                <div key={dataset.dataset_id} className="rounded-2xl border border-white/10 bg-white/[0.02] p-5">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <div className="font-bold">{dataset.filename}</div>
                      <div className="mt-1 text-xs text-[var(--text-muted)]">
                        Owner {dataset.owner_address} {dataset.model_id ? `// Model ${dataset.model_id}` : ''}
                      </div>
                    </div>
                    <div className="rounded-full bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                      {dataset.status}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card>
      </div>

      <Card>
        <div className="flex items-center justify-between">
          <div className="text-sm font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
            Incoming Inference Activity
          </div>
          <div className="text-xs font-mono text-white/40">
            {isLoadingOperations ? 'syncing...' : `${incomingSubmissions.length} requests`}
          </div>
        </div>

        <div className="mt-6 space-y-4">
          {incomingSubmissions.length === 0 ? (
            <div className="rounded-2xl border border-dashed border-white/10 p-6 text-sm text-[var(--text-muted)]">
              No inference requests have been mirrored to the AI Lab metadata queue yet.
            </div>
          ) : (
            incomingSubmissions.map((submission) => (
              <div key={submission.submission_id} className="rounded-2xl border border-white/10 bg-white/[0.02] p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="font-bold">Request {submission.request_id}</div>
                    <div className="mt-1 text-xs text-[var(--text-muted)]">
                      Data Source {submission.owner_address} // Model {submission.model_id}
                    </div>
                  </div>
                  <div className="rounded-full bg-white/10 px-3 py-1 text-[10px] font-bold uppercase tracking-widest text-[var(--accent-cyan)]">
                    {submission.status}
                  </div>
                </div>
                {submission.tx_hash && (
                  <div className="mt-4 text-xs font-mono text-white/40 break-all">
                    Tx {submission.tx_hash}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  );
}
