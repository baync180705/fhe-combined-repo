import type { CofheClient, EncryptedUint32Input } from '@cofhe/sdk';
import type { BrowserProvider, Contract, ContractTransactionReceipt } from 'ethers';
import { parseUnits } from 'ethers';

const PAYMENT_TOKEN_DECIMALS = Number(import.meta.env.VITE_PAYMENT_TOKEN_DECIMALS ?? '6');

type LegacyFhenixClientLike = {
  unseal: (contractAddress: string, ciphertext: string, account: string) => bigint;
};

export interface Model {
  id: string;
  modelId?: bigint;
  name: string;
  price: number;
  labAddress: string;
  accuracy: string;
}

export interface MockLogisticRegressionModel {
  name: string;
  scale: number;
  features: string[];
  weights: number[];
  bias: number;
  ipfsHash: string;
}

export interface UploadedWeightsArtifact {
  name?: string;
  scale?: number;
  features: string[];
  weights: number[];
  bias: number;
  metadataUri?: string;
  artifactSha256: string;
  originalFilename: string;
}

export interface RegisteredModelReceipt {
  modelId: bigint;
  receipt: ContractTransactionReceipt;
}

export interface InferenceSubmission {
  requestId: bigint;
  receipt: ContractTransactionReceipt;
}

export const MOCK_MODELS: Model[] = [
  { id: 'MOD-001', name: 'Diabetes Risk V2', price: 5.5, labAddress: '0x71C...3A1', accuracy: '94.2%' },
  { id: 'MOD-002', name: 'CardioScan Pro', price: 12.0, labAddress: '0x92B...4F2', accuracy: '91.8%' },
  { id: 'MOD-003', name: 'NeuroDetect Alpha', price: 25.0, labAddress: '0xA5E...9D0', accuracy: '89.5%' },
];

export type EncryptedInferenceInput = EncryptedUint32Input;

export function assertUint32(value: number): number {
  if (!Number.isFinite(value) || value < 0 || value > 0xffff_ffff) {
    throw new Error('Inference inputs must be finite uint32 values');
  }
  return Math.trunc(value);
}

export function createMockDiabetesModel(): MockLogisticRegressionModel {
  return {
    name: 'Diabetes-LogReg',
    scale: 1000,
    features: ['Glucose', 'BMI', 'Age'],
    weights: [150, 210, 55],
    bias: 500,
    ipfsHash: 'ipfs://diabetes-logreg-v1',
  };
}

export async function encryptUint32(
  client: CofheClient,
  value: number,
): Promise<EncryptedInferenceInput> {
  const { Encryptable } = await import('@cofhe/sdk');
  const [encryptedInput] = await client
    .encryptInputs([Encryptable.uint32(BigInt(assertUint32(value)))])
    .execute();
  return encryptedInput;
}

export async function encryptUint32Array(
  client: CofheClient,
  values: number[],
): Promise<EncryptedInferenceInput[]> {
  const { Encryptable } = await import('@cofhe/sdk');
  const encryptables = values.map((value) => Encryptable.uint32(BigInt(assertUint32(value))));
  return client.encryptInputs(encryptables).execute();
}

export async function encryptInferenceInput(
  client: CofheClient,
  value: number,
): Promise<EncryptedInferenceInput> {
  return encryptUint32(client, value);
}

export async function ensureSelfPermit(client: CofheClient) {
  const permit = await client.permits.getOrCreateSelfPermit();
  const permitHash = client.permits.getHash(permit);
  client.permits.selectActivePermit(permitHash);
  return permit;
}

export async function decryptInferenceResult(client: CofheClient, handle: bigint | string): Promise<bigint> {
  const { FheTypes } = await import('@cofhe/sdk');
  return client.decryptForView(handle, FheTypes.Uint32).execute();
}

export async function unsealInferenceResult(
  client: LegacyFhenixClientLike | CofheClient,
  provider: BrowserProvider,
  contractAddress: string,
  encryptedResult: bigint | string,
): Promise<bigint> {
  if ('unseal' in client && typeof client.unseal === 'function') {
    const { getPermit } = await import('fhenixjs-access-control');
    await getPermit(contractAddress, provider);
    const signer = await provider.getSigner();
    const account = await signer.getAddress();
    const normalizedCiphertext =
      typeof encryptedResult === 'string' ? encryptedResult : encryptedResult.toString();
    return client.unseal(contractAddress, normalizedCiphertext, account);
  }

  if ('decryptForView' in client && typeof client.decryptForView === 'function') {
    const { FheTypes } = await import('@cofhe/sdk');
    await client.permits.getOrCreateSelfPermit();
    return client.decryptForView(encryptedResult, FheTypes.Uint32).execute();
  }

  throw new Error('Connected FHE client does not support result unsealing');
}

export function toPriceUnits(price: string): bigint {
  const normalized = price.trim() === '' ? '0' : price.trim();
  return parseUnits(normalized, PAYMENT_TOKEN_DECIMALS);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function getOptionalString(value: unknown): string | undefined {
  if (typeof value !== 'string') {
    return undefined;
  }

  const trimmed = value.trim();
  return trimmed === '' ? undefined : trimmed;
}

function normalizeFeatureNames(value: unknown, expectedCount: number): string[] {
  if (!Array.isArray(value) || value.length !== expectedCount) {
    return Array.from({ length: expectedCount }, (_, index) => `feature_${index + 1}`);
  }

  const normalized = value.map((entry, index) => {
    if (typeof entry !== 'string' || entry.trim() === '') {
      return `feature_${index + 1}`;
    }
    return entry.trim();
  });

  return normalized;
}

async function sha256Hex(data: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return Array.from(new Uint8Array(digest))
    .map((value) => value.toString(16).padStart(2, '0'))
    .join('');
}

export async function parseUploadedWeightsArtifact(file: File): Promise<UploadedWeightsArtifact> {
  const buffer = await file.arrayBuffer();
  const text = new TextDecoder().decode(buffer);

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    throw new Error(
      error instanceof Error ? `Invalid JSON artifact: ${error.message}` : 'Invalid JSON artifact',
    );
  }

  if (!isPlainObject(parsed)) {
    throw new Error('Model artifact must be a JSON object containing weights and bias.');
  }

  if (!Array.isArray(parsed.weights) || parsed.weights.length === 0) {
    throw new Error('Model artifact must include a non-empty "weights" array.');
  }

  const weights = parsed.weights.map((value, index) => {
    if (typeof value !== 'number') {
      throw new Error(`Weight at index ${index} must be numeric.`);
    }
    return assertUint32(value);
  });

  if (typeof parsed.bias !== 'number') {
    throw new Error('Model artifact must include a numeric "bias" value.');
  }

  const bias = assertUint32(parsed.bias);

  const scale =
    typeof parsed.scale === 'number' && Number.isFinite(parsed.scale) && parsed.scale > 0
      ? parsed.scale
      : undefined;

  return {
    name: getOptionalString(parsed.name),
    scale,
    features: normalizeFeatureNames(parsed.features, weights.length),
    weights,
    bias,
    metadataUri: getOptionalString(parsed.metadataUri ?? parsed.metadata_uri),
    artifactSha256: await sha256Hex(buffer),
    originalFilename: file.name,
  };
}

function requireReceipt(txReceipt: ContractTransactionReceipt | null): ContractTransactionReceipt {
  if (!txReceipt) {
    throw new Error('Transaction receipt missing');
  }
  return txReceipt;
}

function findEventArg(
  contract: Contract,
  receipt: ContractTransactionReceipt,
  eventName: string,
  argName: string,
): bigint {
  for (const log of receipt.logs) {
    try {
      const parsed = contract.interface.parseLog({ topics: log.topics, data: log.data });
      if (parsed && parsed.name === eventName) {
        const value = parsed.args[argName];
        if (typeof value === 'bigint') {
          return value;
        }
      }
    } catch {
      continue;
    }
  }

  throw new Error(`Unable to find ${eventName}.${argName} in transaction logs`);
}

export async function registerMockModel(params: {
  client: CofheClient;
  modelRegistry: Contract;
  pricePerQuery: bigint;
}): Promise<RegisteredModelReceipt> {
  const mockModel = createMockDiabetesModel();
  const encryptedWeights = await encryptUint32Array(params.client, mockModel.weights);
  const encryptedBias = await encryptUint32(params.client, mockModel.bias);

  const tx = await params.modelRegistry.registerModel(
    encryptedWeights,
    encryptedBias,
    params.pricePerQuery,
    mockModel.ipfsHash,
  );
  const receipt = requireReceipt(await tx.wait());
  const modelId = findEventArg(params.modelRegistry, receipt, 'ModelRegistered', 'modelId');

  return { modelId, receipt };
}

export async function registerEncryptedModel(params: {
  client: CofheClient;
  modelRegistry: Contract;
  weights: number[];
  bias: number;
  pricePerQuery: bigint;
  registryReference: string;
}): Promise<RegisteredModelReceipt> {
  const encryptedWeights = await encryptUint32Array(params.client, params.weights);
  const encryptedBias = await encryptUint32(params.client, params.bias);

  const tx = await params.modelRegistry.registerModel(
    encryptedWeights,
    encryptedBias,
    params.pricePerQuery,
    params.registryReference,
  );
  const receipt = requireReceipt(await tx.wait());
  const modelId = findEventArg(params.modelRegistry, receipt, 'ModelRegistered', 'modelId');

  return { modelId, receipt };
}

export async function submitInference(params: {
  client: CofheClient;
  blindInference: Contract;
  modelId: bigint;
  inputs: number[];
}): Promise<InferenceSubmission> {
  const encryptedInputs = await encryptUint32Array(params.client, params.inputs);
  const inferenceTx = await params.blindInference.predict(params.modelId, encryptedInputs);
  const inferenceReceipt = requireReceipt(await inferenceTx.wait());
  const requestId = findEventArg(params.blindInference, inferenceReceipt, 'PredictionRequested', 'requestId');

  return {
    requestId,
    receipt: inferenceReceipt,
  };
}
