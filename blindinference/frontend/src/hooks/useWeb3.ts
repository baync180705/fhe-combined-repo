import { createContext, createElement, ReactNode, useContext, useEffect, useMemo, useState } from 'react';
import { BrowserProvider, Contract, JsonRpcSigner } from 'ethers';
import type { CofheClient } from '@cofhe/sdk';
import blindInferenceArtifact from '../../../fhenix_inference/artifacts/contracts/BlindInference.sol/BlindInference.json';
import inferenceEngineArtifact from '../../../fhenix_inference/artifacts/contracts/InferenceEngine.sol/InferenceEngine.json';
import modelRegistryArtifact from '../../../fhenix_inference/artifacts/contracts/ModelRegistry.sol/ModelRegistry.json';
import paymentEscrowArtifact from '../../../fhenix_inference/artifacts/contracts/PaymentEscrow.sol/PaymentEscrow.json';
import paymentTokenArtifact from '../contracts/abis/MockERC20.json';
import { AppRole, isAppRole } from '../lib/roles';

declare global {
  interface Window {
    ethereum?: {
      request(args: { method: string; params?: unknown[] }): Promise<unknown>;
    };
  }
}

type ContractArtifact = {
  abi: unknown[];
};

type Web3Contracts = {
  blindInference: Contract | null;
  inferenceEngine: Contract | null;
  modelRegistry: Contract | null;
  paymentEscrow: Contract | null;
  paymentToken: Contract | null;
};

type Web3ContextValue = {
  address: string | null;
  role: AppRole | null;
  jwt: string | null;
  provider: BrowserProvider | null;
  signer: JsonRpcSigner | null;
  fhenixClient: CofheClient | null;
  connectionError: string | null;
  contracts: Web3Contracts;
  paymentTokenAddress: string | null;
  paymentTokenName: string;
  paymentTokenDecimals: number;
  inferenceEngineAddress: string | null;
  isConnecting: boolean;
  isInitializingFhe: boolean;
  isAuthenticating: boolean;
  isRoleSelectionOpen: boolean;
  connect: () => Promise<{
    address: string | null;
    provider: BrowserProvider;
    signer: JsonRpcSigner;
    fhenixClient: CofheClient | null;
    contracts: Web3Contracts;
  }>;
  ensureFhenixClient: () => Promise<CofheClient>;
  selectRole: (role: AppRole) => Promise<void>;
  dismissRoleSelection: () => void;
  disconnect: () => void;
  abis: {
    blindInference: unknown[];
    inferenceEngine: unknown[];
    modelRegistry: unknown[];
    paymentEscrow: unknown[];
    paymentToken: unknown[];
  };
  artifacts: typeof contractArtifacts;
};

const contractArtifacts = {
  blindInference: blindInferenceArtifact as ContractArtifact,
  inferenceEngine: inferenceEngineArtifact as ContractArtifact,
  modelRegistry: modelRegistryArtifact as ContractArtifact,
  paymentEscrow: paymentEscrowArtifact as ContractArtifact,
  paymentToken: paymentTokenArtifact as ContractArtifact,
};

const blindInferenceAddress = import.meta.env.VITE_BLIND_INFERENCE_ADDRESS as string | undefined;
const inferenceEngineAddress =
  (import.meta.env.VITE_INFERENCE_ENGINE_ADDRESS as string | undefined) ?? blindInferenceAddress;
const modelRegistryAddress = import.meta.env.VITE_MODEL_REGISTRY_ADDRESS as string | undefined;
const paymentEscrowAddress = import.meta.env.VITE_PAYMENT_ESCROW_ADDRESS as string | undefined;
const paymentTokenAddress = import.meta.env.VITE_PAYMENT_TOKEN_ADDRESS as string | undefined;
const paymentTokenName = (import.meta.env.VITE_PAYMENT_TOKEN_NAME as string | undefined) ?? 'USDC';
const paymentTokenDecimals = Number(import.meta.env.VITE_PAYMENT_TOKEN_DECIMALS ?? '6');

const defaultContracts: Web3Contracts = {
  blindInference: null,
  inferenceEngine: null,
  modelRegistry: null,
  paymentEscrow: null,
  paymentToken: null,
};

const Web3Context = createContext<Web3ContextValue | null>(null);
const SEPOLIA_CHAIN_ID = 11155111n;
const SEPOLIA_CHAIN_ID_HEX = '0xaa36a7';
const ROLE_STORAGE_PREFIX = 'blindference.role';
const TOKEN_STORAGE_PREFIX = 'blindference.token';
const BACKEND_BASE_URL = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://127.0.0.1:8000';

function normalizeAddress(address: string) {
  return address.toLowerCase();
}

function roleStorageKey(address: string) {
  return `${ROLE_STORAGE_PREFIX}:${normalizeAddress(address)}`;
}

function tokenStorageKey(address: string) {
  return `${TOKEN_STORAGE_PREFIX}:${normalizeAddress(address)}`;
}

function getStoredRole(address: string | null): AppRole | null {
  if (!address) {
    return null;
  }

  const stored = localStorage.getItem(roleStorageKey(address));
  return isAppRole(stored) ? stored : null;
}

function persistRole(address: string, role: AppRole) {
  localStorage.setItem(roleStorageKey(address), role);
}

function getStoredToken(address: string | null): string | null {
  if (!address) {
    return null;
  }

  return localStorage.getItem(tokenStorageKey(address));
}

function persistToken(address: string, token: string) {
  localStorage.setItem(tokenStorageKey(address), token);
}

function clearStoredToken(address: string | null) {
  if (!address) {
    return;
  }

  localStorage.removeItem(tokenStorageKey(address));
}

type VerifyResponse = {
  access_token: string;
  token_type: string;
  role: AppRole;
};

async function ensureSepoliaNetwork(provider: BrowserProvider) {
  const network = await provider.getNetwork();
  if (network.chainId === SEPOLIA_CHAIN_ID) {
    return;
  }

  if (!window.ethereum) {
    throw new Error('MetaMask is required to switch to Sepolia');
  }

  try {
    await window.ethereum.request({
      method: 'wallet_switchEthereumChain',
      params: [{ chainId: SEPOLIA_CHAIN_ID_HEX }],
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : '';
    throw new Error(
      message.includes('4902')
        ? 'Sepolia is not available in MetaMask. Add the Sepolia network and try again.'
        : 'Wrong network detected. Switch MetaMask to Ethereum Sepolia and reconnect.',
    );
  }

  const refreshedNetwork = await provider.getNetwork();
  if (refreshedNetwork.chainId !== SEPOLIA_CHAIN_ID) {
    throw new Error('Wrong network detected. Switch MetaMask to Ethereum Sepolia and reconnect.');
  }
}

async function createFhenixClient(provider: BrowserProvider, signer: JsonRpcSigner) {
  await ensureSepoliaNetwork(provider);

  const [{ Ethers6Adapter }, { chains }, { createCofheClient, createCofheConfig }] = await Promise.all([
    import('@cofhe/sdk/adapters'),
    import('@cofhe/sdk/chains'),
    import('@cofhe/sdk/web'),
  ]);

  const config = createCofheConfig({
    supportedChains: [chains.sepolia],
    // Disable the SDK's iframe-backed IndexedDB cache. In our local/Vite setup
    // the cross-origin storage hub may fail to initialize, which breaks client
    // startup while rehydrating the persisted FHE keys store.
    fheKeyStorage: null,
  });
  const client = createCofheClient(config);
  const { publicClient, walletClient } = await Ethers6Adapter(provider, signer);
  await client.connect(publicClient, walletClient);

  return client;
}

function buildContracts(signer: JsonRpcSigner): Web3Contracts {
  return {
    blindInference: blindInferenceAddress
      ? new Contract(blindInferenceAddress, contractArtifacts.blindInference.abi, signer)
      : null,
    inferenceEngine: inferenceEngineAddress
      ? new Contract(inferenceEngineAddress, contractArtifacts.inferenceEngine.abi, signer)
      : null,
    modelRegistry: modelRegistryAddress
      ? new Contract(modelRegistryAddress, contractArtifacts.modelRegistry.abi, signer)
      : null,
    paymentEscrow: paymentEscrowAddress
      ? new Contract(paymentEscrowAddress, contractArtifacts.paymentEscrow.abi, signer)
      : null,
    paymentToken: paymentTokenAddress
      ? new Contract(paymentTokenAddress, contractArtifacts.paymentToken.abi, signer)
      : null,
  };
}

export function Web3Provider({ children }: { children: ReactNode }) {
  const [address, setAddress] = useState<string | null>(null);
  const [role, setRole] = useState<AppRole | null>(null);
  const [jwt, setJwt] = useState<string | null>(null);
  const [provider, setProvider] = useState<BrowserProvider | null>(null);
  const [signer, setSigner] = useState<JsonRpcSigner | null>(null);
  const [fhenixClient, setFhenixClient] = useState<CofheClient | null>(null);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [contracts, setContracts] = useState<Web3Contracts>(defaultContracts);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isInitializingFhe, setIsInitializingFhe] = useState(false);
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [isRoleSelectionOpen, setIsRoleSelectionOpen] = useState(false);

  async function authenticateWallet(
    nextAddress: string,
    nextSigner: JsonRpcSigner,
    nextRole: AppRole,
  ): Promise<VerifyResponse> {
    setIsAuthenticating(true);
    setConnectionError(null);

    try {
      const nonceResponse = await fetch(`${BACKEND_BASE_URL}/api/v1/auth/nonce`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          address: normalizeAddress(nextAddress),
          role: nextRole,
        }),
      });

      if (!nonceResponse.ok) {
        const errorPayload = await nonceResponse.text();
        throw new Error(errorPayload || 'Failed to request authentication nonce');
      }

      const noncePayload = (await nonceResponse.json()) as {
        nonce_id: string;
        message: string;
      };

      const signature = await nextSigner.signMessage(noncePayload.message);

      const verifyResponse = await fetch(`${BACKEND_BASE_URL}/api/v1/auth/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          address: normalizeAddress(nextAddress),
          role: nextRole,
          nonce_id: noncePayload.nonce_id,
          signature,
        }),
      });

      if (!verifyResponse.ok) {
        const errorPayload = await verifyResponse.text();
        throw new Error(errorPayload || 'Failed to verify wallet signature');
      }

      const verified = (await verifyResponse.json()) as VerifyResponse;
      persistToken(nextAddress, verified.access_token);
      setJwt(verified.access_token);
      setRole(verified.role);
      setIsRoleSelectionOpen(false);
      return verified;
    } catch (error) {
      clearStoredToken(nextAddress);
      setJwt(null);
      setConnectionError(error instanceof Error ? error.message : 'Wallet authentication failed.');
      throw error;
    } finally {
      setIsAuthenticating(false);
    }
  }

  async function initializeFhenixClient(nextProvider: BrowserProvider, nextSigner: JsonRpcSigner) {
    setIsInitializingFhe(true);
    try {
      const nextClient = await createFhenixClient(nextProvider, nextSigner);
      setFhenixClient(nextClient);
      setConnectionError(null);
      return nextClient;
    } catch (error) {
      console.error('Failed to initialize Fhenix client:', error);
      setFhenixClient(null);
      setConnectionError(
        error instanceof Error
          ? error.message
          : 'Wallet connected, but the Fhenix client failed to initialize.',
      );
      return null;
    } finally {
      setIsInitializingFhe(false);
    }
  }

  useEffect(() => {
    let isActive = true;

    async function hydrateAuthorizedAccount() {
      if (!window.ethereum) {
        return;
      }

      const accounts = (await window.ethereum.request({ method: 'eth_accounts' })) as string[];
      if (accounts.length === 0 || !isActive) {
        return;
      }

      let nextProvider = new BrowserProvider(window.ethereum);
      await ensureSepoliaNetwork(nextProvider);
      nextProvider = new BrowserProvider(window.ethereum);
      const nextSigner = await nextProvider.getSigner();
      const nextAddress = accounts[0];
      const nextRole = getStoredRole(nextAddress);

      if (!isActive) {
        return;
      }

      setAddress(nextAddress);
      setRole(nextRole);
      setJwt(getStoredToken(nextAddress));
      setIsRoleSelectionOpen(nextRole == null);
      setProvider(nextProvider);
      setSigner(nextSigner);
      setContracts(buildContracts(nextSigner));
    }

    hydrateAuthorizedAccount().catch((error) => {
      console.error('Failed to hydrate wallet state:', error);
    });

    return () => {
      isActive = false;
    };
  }, []);

  const connect = async () => {
    if (!window.ethereum) {
      throw new Error('No injected wallet found');
    }

    setIsConnecting(true);
    setConnectionError(null);
    try {
      const accounts = (await window.ethereum.request({ method: 'eth_requestAccounts' })) as string[];
      let nextProvider = new BrowserProvider(window.ethereum);
      await ensureSepoliaNetwork(nextProvider);
      nextProvider = new BrowserProvider(window.ethereum);
      const nextSigner = await nextProvider.getSigner();
      const nextContracts = buildContracts(nextSigner);
      const nextAddress = accounts[0] ?? null;
      const nextRole = getStoredRole(nextAddress);
      const nextToken = getStoredToken(nextAddress);

      setAddress(nextAddress);
      setRole(nextRole);
      setJwt(nextToken);
      setProvider(nextProvider);
      setSigner(nextSigner);
      setContracts(nextContracts);
      setIsRoleSelectionOpen(Boolean(nextAddress) && nextRole == null);

      if (nextAddress && nextRole && !nextToken) {
        await authenticateWallet(nextAddress, nextSigner, nextRole);
      }

      return {
        address: nextAddress,
        provider: nextProvider,
        signer: nextSigner,
        fhenixClient: null,
        contracts: nextContracts,
      };
    } finally {
      setIsConnecting(false);
    }
  };

  const ensureFhenixClient = async () => {
    if (fhenixClient) {
      return fhenixClient;
    }

    if (!provider || !signer) {
      throw new Error('Connect your wallet before initializing Fhenix');
    }

    const nextClient = await initializeFhenixClient(provider, signer);
    if (!nextClient) {
      throw new Error('Fhenix client could not be initialized');
    }

    return nextClient;
  };

  const selectRole = async (nextRole: AppRole) => {
    if (!address || !signer) {
      throw new Error('Connect your wallet before selecting a role');
    }

    persistRole(address, nextRole);
    setRole(nextRole);
    await authenticateWallet(address, signer, nextRole);
  };

  const dismissRoleSelection = () => {
    setIsRoleSelectionOpen(false);
  };

  const disconnect = () => {
    clearStoredToken(address);
    setAddress(null);
    setRole(null);
    setJwt(null);
    setProvider(null);
    setSigner(null);
    setFhenixClient(null);
    setConnectionError(null);
    setContracts(defaultContracts);
    setIsRoleSelectionOpen(false);
  };

  const value = useMemo<Web3ContextValue>(
    () => ({
      address,
      role,
      jwt,
      provider,
      signer,
      fhenixClient,
      connectionError,
      contracts,
      paymentTokenAddress: paymentTokenAddress ?? null,
      paymentTokenName,
      paymentTokenDecimals,
      inferenceEngineAddress: inferenceEngineAddress ?? null,
      isConnecting,
      isInitializingFhe,
      isAuthenticating,
      isRoleSelectionOpen,
      connect,
      ensureFhenixClient,
      selectRole,
      dismissRoleSelection,
      disconnect,
      abis: {
        blindInference: contractArtifacts.blindInference.abi,
        inferenceEngine: contractArtifacts.inferenceEngine.abi,
        modelRegistry: contractArtifacts.modelRegistry.abi,
        paymentEscrow: contractArtifacts.paymentEscrow.abi,
        paymentToken: contractArtifacts.paymentToken.abi,
      },
      artifacts: contractArtifacts,
    }),
    [
      address,
      role,
      jwt,
      provider,
      signer,
      fhenixClient,
      connectionError,
      contracts,
      isConnecting,
      isInitializingFhe,
      isAuthenticating,
      isRoleSelectionOpen,
    ],
  );

  return createElement(Web3Context.Provider, { value }, children);
}

export function useWeb3() {
  const context = useContext(Web3Context);
  if (!context) {
    throw new Error('useWeb3 must be used within a Web3Provider');
  }
  return context;
}
