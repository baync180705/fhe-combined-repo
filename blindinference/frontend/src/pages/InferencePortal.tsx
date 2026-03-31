import React, { useEffect, useRef, useState, FormEvent } from 'react';
import { formatUnits } from 'ethers';
import { Model, submitInference, unsealInferenceResult } from '../services/fheService';
import { Card, Button, Input } from '../components/UI';
import { motion, AnimatePresence } from 'motion/react';
import { Shield, Terminal, Lock, Unlock, Activity, CheckCircle2 } from 'lucide-react';
import { useWeb3 } from '../hooks/useWeb3';
import { upsertSubmission } from '../services/workspaceService';

const DEFAULT_MODEL_ID = BigInt(import.meta.env.VITE_DEFAULT_MODEL_ID ?? '1');

type PortalStatus =
  | 'idle'
  | 'approving'
  | 'encrypting'
  | 'submitting'
  | 'sealed'
  | 'decrypting'
  | 'complete';

export default function InferencePortal({ model }: { model: Model }) {
  const [patientInputs, setPatientInputs] = useState({
    glucose: '120',
    bmi: '28',
    age: '45',
  });
  const [modelId, setModelId] = useState(
    (model.modelId ?? DEFAULT_MODEL_ID).toString(),
  );
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<PortalStatus>('idle');
  const [requestId, setRequestId] = useState<string | null>(null);
  const [sealedHandle, setSealedHandle] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<string | null>(null);
  const [isDecrypting, setIsDecrypting] = useState(false);
  const [inferenceFee, setInferenceFee] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const {
    address,
    role,
    jwt,
    connect,
    provider,
    fhenixClient,
    ensureFhenixClient,
    contracts,
    paymentTokenAddress,
    paymentTokenName,
    paymentTokenDecimals,
    inferenceEngineAddress,
  } = useWeb3();

  useEffect(() => {
    setModelId((model.modelId ?? DEFAULT_MODEL_ID).toString());
  }, [model.modelId]);

  const addLog = (msg: string) => {
    setLogs((prev) => [...prev, `[${new globalThis.Date().toLocaleTimeString()}] ${msg}`]);
  };

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const handleInference = async (e: FormEvent) => {
    e.preventDefault();
    setLogs([]);
    setInferenceResult(null);
    setSealedHandle(null);
    setRequestId(null);
    setInferenceFee(null);
    addLog(`Initiating blind inference with ${model.name}...`);

    try {
      let activeClient = fhenixClient;
      let activeBlindInference = contracts.blindInference;
      let activeRegistry = contracts.modelRegistry;
      let activePaymentToken = contracts.paymentToken;
      let activeAddress = address;

      if (!activeAddress) {
        addLog('Wallet not connected. Requesting connection...');
        const session = await connect();
        activeClient = session?.fhenixClient ?? activeClient;
        activeBlindInference = session?.contracts.blindInference ?? activeBlindInference;
        activeRegistry = session?.contracts.modelRegistry ?? activeRegistry;
        activePaymentToken = session?.contracts.paymentToken ?? activePaymentToken;
        activeAddress = session?.address ?? activeAddress;
      }

      if (!activeClient) {
        addLog('Initializing Fhenix client on demand...');
        activeClient = await ensureFhenixClient();
      }

      if (!activeBlindInference || !activeRegistry || !activePaymentToken) {
        throw new Error('BlindInference, ModelRegistry, or payment token contract is not configured');
      }

      if (!inferenceEngineAddress) {
        throw new Error('Inference engine address is not configured');
      }

      const selectedModelId = BigInt(modelId);
      const featureVector = [
        Number(patientInputs.glucose),
        Number(patientInputs.bmi),
        Number(patientInputs.age),
      ];
      const featureNames = ['Glucose', 'BMI', 'Age'];

      featureVector.forEach((value, index) => {
        if (!Number.isFinite(value)) {
          throw new Error(`Invalid numeric value for ${featureNames[index]}`);
        }
      });

      const fee = (await activeRegistry.getInferenceFee(selectedModelId)) as bigint;
      const formattedFee = formatUnits(fee, paymentTokenDecimals);
      setInferenceFee(formattedFee);
      addLog(`Fetched inference fee for model ${selectedModelId.toString()}: ${formattedFee} ${paymentTokenName}`);

      const allowance = (await activePaymentToken.allowance(activeAddress, inferenceEngineAddress)) as bigint;

      if (allowance < fee) {
        setStatus('approving');
        addLog(`Allowance is insufficient. Setting approval to the exact required amount: ${formattedFee} ${paymentTokenName}.`);
        const approvalTx = await activePaymentToken.approve(inferenceEngineAddress, fee);
        addLog(`Approval submitted: ${approvalTx.hash}`);
        await approvalTx.wait();
        addLog(`Token approval confirmed at ${formattedFee} ${paymentTokenName}.`);
      } else {
        addLog(`Existing ${paymentTokenName} allowance is sufficient.`);
      }

      setStatus('encrypting');
      addLog('Encrypting Glucose, BMI, and Age with the Fhenix network public key...');

      setStatus('submitting');
      addLog('Submitting encrypted inference request to the toll bridge...');
      const submission = await submitInference({
        client: activeClient,
        blindInference: activeBlindInference,
        modelId: selectedModelId,
        inputs: featureVector,
      });

      setRequestId(submission.requestId.toString());
      addLog(`Inference submitted successfully. Request ID: ${submission.requestId.toString()}`);
      addLog(`Prediction transaction confirmed: ${submission.receipt.hash}`);

      if (jwt && role === 'data_source') {
        await upsertSubmission(jwt, {
          request_id: submission.requestId.toString(),
          model_id: selectedModelId.toString(),
          lab_address: model.labAddress,
          tx_hash: submission.receipt.hash,
          status: 'submitted',
        });
      }

      const encryptedResult = await activeBlindInference.getResult(submission.requestId);
      const normalizedHandle = encryptedResult.toString();
      setSealedHandle(normalizedHandle);
      addLog('Encrypted score handle fetched from BlindInference.');
      addLog('The result remains sealed until the hospital signs a viewing permit.');

      if (jwt && role === 'data_source') {
        await upsertSubmission(jwt, {
          request_id: submission.requestId.toString(),
          model_id: selectedModelId.toString(),
          lab_address: model.labAddress,
          tx_hash: submission.receipt.hash,
          status: 'sealed',
          result_handle: normalizedHandle,
        });
      }

      setStatus('sealed');
    } catch (err) {
      addLog(`ERROR: ${err instanceof Error ? err.message : 'Inference failed.'}`);
      setStatus('idle');
    }
  };

  const handleDecrypt = async () => {
    if (!fhenixClient || !provider || !requestId) {
      addLog('ERROR: Connect a wallet and run an inference before decrypting.');
      setStatus('sealed');
      return;
    }

    setStatus('decrypting');
    setIsDecrypting(true);
    addLog('Prompting wallet for an EIP-712 permit to unseal the result...');

    try {
      if (!contracts.inferenceEngine || !inferenceEngineAddress) {
        throw new Error('Inference engine contract is not configured');
      }

      const encryptedResult = await contracts.inferenceEngine.getResult(BigInt(requestId));
      const normalizedHandle = encryptedResult.toString();
      setSealedHandle(normalizedHandle);
      addLog(`Encrypted result handle fetched for request ${requestId}.`);
      addLog('Permit ready. Requesting plaintext from the Fhenix threshold network...');
      const decrypted = await unsealInferenceResult(
        fhenixClient,
        provider,
        inferenceEngineAddress,
        encryptedResult,
      );
      setInferenceResult(decrypted.toString());
      setStatus('complete');
      addLog(`Decryption successful. Plaintext score: ${decrypted.toString()}`);

      if (jwt && role === 'data_source') {
        await upsertSubmission(jwt, {
          request_id: requestId,
          model_id: modelId,
          lab_address: model.labAddress,
          status: 'complete',
          result_handle: normalizedHandle,
          plaintext_result: decrypted.toString(),
        });
      }
    } catch (error) {
      addLog(`ERROR: ${error instanceof Error ? error.message : 'Decryption failed.'}`);
      setStatus('sealed');
    } finally {
      setIsDecrypting(false);
    }
  };

  const isWorking =
    status === 'approving' || status === 'encrypting' || status === 'submitting';

  const actionLabel =
    status === 'approving'
      ? 'Approving Tokens...'
      : status === 'encrypting'
        ? 'Encrypting Data...'
        : status === 'submitting'
          ? 'Submitting Inference...'
          : 'Run Blind Inference';

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-in fade-in duration-500">
      <div className="lg:col-span-2 space-y-6">
        <div className="flex items-center gap-4 mb-8">
          <div className="p-3 bg-[var(--accent-cyan)]/10 rounded-xl">
            <Shield className="w-8 h-8 text-[var(--accent-cyan)]" />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight neon-text">Inference Portal</h1>
            <p className="text-[var(--text-muted)]">
              {role === 'ai_lab'
                ? 'Preview the buyer-side blind inference flow your customers will experience when they submit encrypted inputs.'
                : `Data Sources approve ${paymentTokenName} spend, encrypt sensitive features locally, and only the requester can unseal the final score.`}
            </p>
          </div>
        </div>

        <Card className="relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4">
            <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 rounded-full border border-emerald-500/20">
              <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] font-bold text-emerald-500 uppercase tracking-widest">Pay-Per-Inference Active</span>
            </div>
          </div>

          <form onSubmit={handleInference} className="space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <Input
                label="On-Chain Model ID"
                type="number"
                placeholder="1"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                required
                disabled={status !== 'idle'}
              />
              <div className="rounded-xl border border-white/10 bg-[var(--bg-secondary)]/40 px-4 py-3">
                <div className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Spender</div>
                <div className="mt-2 text-sm text-[var(--text-main)] break-all">{inferenceEngineAddress ?? 'Not configured'}</div>
              </div>
              <Input
                label="Glucose"
                type="number"
                placeholder="120"
                value={patientInputs.glucose}
                onChange={(e) => setPatientInputs({ ...patientInputs, glucose: e.target.value })}
                required
                disabled={status !== 'idle'}
              />
              <Input
                label="BMI"
                type="number"
                placeholder="28"
                value={patientInputs.bmi}
                onChange={(e) => setPatientInputs({ ...patientInputs, bmi: e.target.value })}
                required
                disabled={status !== 'idle'}
              />
              <div className="col-span-2">
                <Input
                  label="Age"
                  type="number"
                  placeholder="45"
                  value={patientInputs.age}
                  onChange={(e) => setPatientInputs({ ...patientInputs, age: e.target.value })}
                  required
                  disabled={status !== 'idle'}
                />
              </div>
            </div>

            <div className="rounded-xl border border-white/10 bg-[var(--bg-secondary)]/30 px-4 py-3 text-sm text-[var(--text-muted)]">
              <div>Selected model: {model.name}</div>
              <div>Payment token: {paymentTokenAddress ? `${paymentTokenName} (${paymentTokenAddress})` : `${paymentTokenName} (Not configured)`}</div>
              <div>Inference fee: {inferenceFee ? `${inferenceFee} ${paymentTokenName}` : 'Will be fetched before submit'}</div>
            </div>

            <div className="pt-4 flex items-center justify-between border-t border-[var(--bg-secondary)]">
              <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
                <Lock className="w-3 h-3" />
                Plaintext never leaves the data source browser. Approval and inference are signed as separate wallet actions.
              </div>
              <Button type="submit" isLoading={isWorking} disabled={status !== 'idle'}>
                {actionLabel}
              </Button>
            </div>
          </form>
        </Card>

        <AnimatePresence>
          {(status === 'approving' || status === 'encrypting' || status === 'submitting') && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center justify-center py-12 space-y-4"
            >
              <motion.div
                animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                transition={{ repeat: Infinity, duration: 2 }}
                className="w-24 h-24 rounded-full border-4 border-[var(--accent-cyan)] flex items-center justify-center"
              >
                <Activity className="w-12 h-12 text-[var(--accent-cyan)]" />
              </motion.div>
              <h3 className="text-xl font-bold neon-text">
                {status === 'approving'
                  ? `Authorizing ${paymentTokenName} Spend`
                  : status === 'encrypting'
                    ? 'Encrypting Patient Data'
                    : 'Submitting Inference'}
              </h3>
              <p className="text-sm text-[var(--text-muted)]">
                {status === 'approving'
                  ? 'Waiting for ERC-20 approve() confirmation...'
                  : status === 'encrypting'
                    ? 'Preparing encrypted inputs with the Fhenix network public key...'
                    : 'Executing encrypted dot-product math on-chain...'}
              </p>
            </motion.div>
          )}

          {(sealedHandle || inferenceResult || requestId) && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-4"
            >
              <Card className="bg-[var(--bg-secondary)]/50 border-[var(--accent-cyan)]/30">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-5 h-5 text-[var(--status-success)]" />
                    <h3 className="font-bold">Encrypted Score Ready</h3>
                  </div>
                  <div className="text-[10px] font-mono text-[var(--text-muted)]">
                    REQUEST {requestId ?? 'pending'}
                  </div>
                </div>

                {!inferenceResult ? (
                  <div className="flex flex-col items-center py-6 space-y-4">
                    {sealedHandle && (
                      <div className="p-4 bg-[var(--bg-secondary)] rounded-lg border border-slate-800 w-full font-mono text-xs break-all text-[var(--text-muted)]">
                        {sealedHandle}
                      </div>
                    )}
                    <Button variant="outline" onClick={handleDecrypt} isLoading={isDecrypting}>
                      <Unlock className="w-4 h-4 mr-2" />
                      {isDecrypting ? 'Decrypting Result...' : 'Decrypt Result'}
                    </Button>
                  </div>
                ) : (
                  <div className="text-center py-8 space-y-2">
                    <div className="text-sm uppercase tracking-widest text-[var(--text-muted)] font-bold">Plaintext Logistic Regression Score</div>
                    <div className="text-4xl font-black neon-text uppercase tracking-tighter">{inferenceResult}</div>
                  </div>
                )}
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="space-y-6">
        <Card className="h-full flex flex-col min-h-[500px]">
          <div className="flex items-center gap-2 mb-4 border-b border-[var(--bg-secondary)] pb-4">
            <Terminal className="w-4 h-4 text-[var(--accent-cyan)]" />
            <h3 className="text-xs font-bold uppercase tracking-widest">Live Privacy Ledger</h3>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 font-mono text-[10px]">
            {logs.length === 0 && (
              <div className="text-[var(--text-muted)] italic opacity-50">Waiting for process initiation...</div>
            )}
            {logs.map((log, i) => (
              <div key={i} className="text-[var(--text-muted)] border-l-2 border-[var(--bg-secondary)] pl-2 py-1">
                {log}
              </div>
            ))}
            <div ref={logEndRef} />
          </div>
        </Card>
      </div>
    </div>
  );
}
