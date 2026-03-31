import { AppRole } from '../lib/roles';

const PINATA_PIN_JSON_URL = 'https://api.pinata.cloud/pinning/pinJSONToIPFS';
const PINATA_JWT = import.meta.env.VITE_PINATA_JWT as string | undefined;
const IPFS_GATEWAY_BASE_URL = (
  (import.meta.env.VITE_IPFS_GATEWAY_BASE_URL as string | undefined) ?? 'https://gateway.pinata.cloud/ipfs'
).replace(/\/$/, '');

export type IpfsUserProfileDocument = {
  version: 1;
  address: string;
  role: AppRole;
  display_name: string | null;
  organization: string | null;
  bio: string | null;
  created_at: string;
  updated_at: string;
};

export function isIpfsUri(value: string | null | undefined): value is string {
  return typeof value === 'string' && value.startsWith('ipfs://');
}

export function buildIpfsGatewayUrl(uri: string) {
  if (uri.startsWith('ipfs://')) {
    const normalizedPath = uri.replace('ipfs://', '').replace(/^ipfs\//, '');
    return `${IPFS_GATEWAY_BASE_URL}/${normalizedPath}`;
  }

  return uri;
}

export async function uploadUserProfileToIpfs(document: IpfsUserProfileDocument): Promise<string> {
  if (!PINATA_JWT) {
    throw new Error('Missing VITE_PINATA_JWT. Configure an IPFS pinning token before saving profiles.');
  }

  const response = await fetch(PINATA_PIN_JSON_URL, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${PINATA_JWT}`,
    },
    body: JSON.stringify({
      pinataContent: document,
      pinataMetadata: {
        name: `blindference-profile-${document.address.toLowerCase()}`,
      },
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to upload profile to IPFS');
  }

  const payload = (await response.json()) as { IpfsHash?: string };
  if (!payload.IpfsHash) {
    throw new Error('IPFS pinning response did not include a CID');
  }

  return `ipfs://${payload.IpfsHash}`;
}

export async function fetchUserProfileFromIpfs(uri: string): Promise<IpfsUserProfileDocument> {
  const response = await fetch(buildIpfsGatewayUrl(uri), {
    method: 'GET',
    headers: {
      Accept: 'application/json',
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch profile from IPFS');
  }

  return response.json() as Promise<IpfsUserProfileDocument>;
}
