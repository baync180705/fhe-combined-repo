import { AppRole } from '../lib/roles';

const BACKEND_BASE_URL = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://127.0.0.1:8000';

export type UserProfile = {
  address: string;
  role: AppRole;
  display_name: string | null;
  organization: string | null;
  bio: string | null;
  profile_uri: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

export type UserProfileInput = {
  display_name: string;
  organization: string;
  bio: string;
  profile_uri: string;
};

function authHeaders(jwt: string) {
  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${jwt}`,
  };
}

export async function getUserProfile(address: string, jwt: string): Promise<UserProfile> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/profile/${address}`, {
    method: 'GET',
    headers: authHeaders(jwt),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch user profile');
  }

  return response.json() as Promise<UserProfile>;
}

export async function saveUserProfile(
  address: string,
  jwt: string,
  payload: UserProfileInput,
): Promise<UserProfile> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/profile/${address}`, {
    method: 'PUT',
    headers: authHeaders(jwt),
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to save user profile');
  }

  return response.json() as Promise<UserProfile>;
}
