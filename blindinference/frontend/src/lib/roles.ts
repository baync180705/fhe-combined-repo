export type AppRole = 'data_source' | 'ai_lab';

export type RoleDefinition = {
  label: string;
  badge: string;
  tagline: string;
  summary: string;
  capabilities: string[];
};

export const ROLE_DEFINITIONS: Record<AppRole, RoleDefinition> = {
  data_source: {
    label: 'Data Source',
    badge: 'SOURCE',
    tagline: 'Upload private inputs and request confidential AI decisions.',
    summary:
      'Best for hospitals, research partners, and institutions that want to keep raw data private while using external AI models.',
    capabilities: [
      'Browse available AI labs and models',
      'Run blind inference on encrypted inputs',
      'Keep sensitive data local until encrypted',
    ],
  },
  ai_lab: {
    label: 'AI Lab',
    badge: 'AI LAB',
    tagline: 'Register encrypted models and serve private inference to data owners.',
    summary:
      'Best for model builders who want to list encrypted models, set prices, and operate inside the confidential Fhenix workflow.',
    capabilities: [
      'Register an AI lab identity on-chain',
      'Upload and price encrypted models',
      'Manage the model supply side of the marketplace',
    ],
  },
};

export function isAppRole(value: string | null | undefined): value is AppRole {
  return value === 'data_source' || value === 'ai_lab';
}
