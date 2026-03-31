const BACKEND_BASE_URL = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? 'http://127.0.0.1:8000';

export type DatasetManifest = {
  dataset_id: string;
  file_id: string;
  filename: string;
  owner_address: string;
  lab_address?: string | null;
  model_id?: string | null;
  original_filename?: string | null;
  content_type?: string | null;
  notes?: string | null;
  status: string;
  visibility?: string;
  artifact_type?: string | null;
  encryption_scheme?: string | null;
  source_format?: string | null;
  artifact_sha256?: string | null;
  row_count?: number;
  feature_count?: number;
  label_count?: number;
  label_column_index?: number;
  feature_names?: string[] | null;
  label_name?: string | null;
  quantization?: {
    frac_bits: number;
    total_bits: number;
    scale: number;
    q_min: number;
    q_max: number;
  } | null;
  tensor_artifacts?: {
    features: {
      rows: number;
      cols: number;
      encrypted_byte_len: number;
    };
    labels: {
      rows: number;
      cols: number;
      encrypted_byte_len: number;
    };
  } | null;
  linked_model_count?: number;
  linked_models?: TrainedModelRecord[];
  created_at: string;
  updated_at: string;
};

export type TrainedModelRecord = {
  model_id: string;
  dataset_id: string;
  file_id?: string | null;
  lab_address: string;
  name: string;
  description?: string | null;
  price_bfhe?: string | null;
  status: string;
  artifact_type?: string | null;
  artifact_sha256?: string | null;
  content_type?: string | null;
  filename: string;
  original_filename?: string | null;
  on_chain_model_id?: string | null;
  tx_hash?: string | null;
  metadata_uri?: string | null;
  registry_reference?: string | null;
  weight_count?: number | null;
  feature_names?: string[] | null;
  scale?: number | null;
  created_at: string;
  updated_at: string;
};

export type SubmissionRecord = {
  submission_id: string;
  request_id: string;
  owner_address: string;
  lab_address: string;
  model_id: string;
  tx_hash?: string | null;
  status: string;
  result_handle?: string | null;
  plaintext_result?: string | null;
  created_at: string;
  updated_at: string;
};

function authHeaders(jwt: string, includeJson = true) {
  return {
    ...(includeJson ? { 'Content-Type': 'application/json' } : {}),
    Authorization: `Bearer ${jwt}`,
  };
}

export async function encryptAndUploadDataset(
  jwt: string,
  payload: {
    file: File;
    label_column?: string;
    has_header?: boolean;
    notes?: string;
  },
): Promise<DatasetManifest> {
  const formData = new FormData();
  formData.append('file', payload.file);
  formData.append('label_column', payload.label_column ?? 'last');
  formData.append('has_header', payload.has_header ? 'true' : 'false');
  if (payload.notes) {
    formData.append('notes', payload.notes);
  }

  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/datasets/encrypt-upload`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${jwt}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to encrypt and upload dataset');
  }

  return response.json() as Promise<DatasetManifest>;
}

export async function getOutgoingDatasets(address: string, jwt: string): Promise<DatasetManifest[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/datasets/outgoing/${address}`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch outgoing datasets');
  }

  return response.json() as Promise<DatasetManifest[]>;
}

export async function getIncomingDatasets(address: string, jwt: string): Promise<DatasetManifest[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/datasets/incoming/${address}`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch incoming datasets');
  }

  return response.json() as Promise<DatasetManifest[]>;
}

export async function getDatasetCatalog(jwt: string): Promise<DatasetManifest[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/datasets/catalog`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch dataset catalog');
  }

  return response.json() as Promise<DatasetManifest[]>;
}

export async function downloadDatasetArtifact(fileId: string, filename: string, jwt: string): Promise<void> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/dataset/download/${fileId}`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to download encrypted dataset artifact');
  }

  const blob = await response.blob();
  const objectUrl = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = objectUrl;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.URL.revokeObjectURL(objectUrl);
}

export async function uploadModelArtifact(
  jwt: string,
  payload: {
    file: File;
    dataset_id: string;
    name: string;
    description?: string;
    price_bfhe?: string;
    status?: string;
    on_chain_model_id?: string;
  },
): Promise<TrainedModelRecord> {
  const formData = new FormData();
  formData.append('file', payload.file);
  formData.append('dataset_id', payload.dataset_id);
  formData.append('name', payload.name);
  if (payload.description) {
    formData.append('description', payload.description);
  }
  if (payload.price_bfhe) {
    formData.append('price_bfhe', payload.price_bfhe);
  }
  if (payload.status) {
    formData.append('status', payload.status);
  }
  if (payload.on_chain_model_id) {
    formData.append('on_chain_model_id', payload.on_chain_model_id);
  }

  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/models/upload`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${jwt}`,
    },
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to upload encrypted model artifact');
  }

  return response.json() as Promise<TrainedModelRecord>;
}

export async function publishModelRecord(
  jwt: string,
  payload: {
    dataset_id: string;
    name: string;
    description?: string;
    price_bfhe?: string;
    status?: string;
    on_chain_model_id: string;
    tx_hash?: string;
    original_filename?: string;
    artifact_sha256?: string;
    metadata_uri?: string;
    registry_reference?: string;
    weight_count?: number;
    feature_names?: string[];
    scale?: number;
  },
): Promise<TrainedModelRecord> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/models/publish`, {
    method: 'POST',
    headers: authHeaders(jwt),
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to persist published model metadata');
  }

  return response.json() as Promise<TrainedModelRecord>;
}

export async function getLabModelArtifacts(address: string, jwt: string): Promise<TrainedModelRecord[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/models/by-lab/${address}`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch AI lab model artifacts');
  }

  return response.json() as Promise<TrainedModelRecord[]>;
}

export async function getModelCatalog(jwt: string): Promise<TrainedModelRecord[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/models/catalog`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch model catalog');
  }

  return response.json() as Promise<TrainedModelRecord[]>;
}

export async function upsertSubmission(
  jwt: string,
  payload: {
    request_id: string;
    model_id: string;
    lab_address: string;
    tx_hash?: string;
    status: string;
    result_handle?: string;
    plaintext_result?: string;
  },
): Promise<SubmissionRecord> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/submissions`, {
    method: 'POST',
    headers: authHeaders(jwt),
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to persist submission metadata');
  }

  return response.json() as Promise<SubmissionRecord>;
}

export async function getOutgoingSubmissions(address: string, jwt: string): Promise<SubmissionRecord[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/submissions/outgoing/${address}`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch outgoing submissions');
  }

  return response.json() as Promise<SubmissionRecord[]>;
}

export async function getIncomingSubmissions(address: string, jwt: string): Promise<SubmissionRecord[]> {
  const response = await fetch(`${BACKEND_BASE_URL}/api/v1/submissions/incoming/${address}`, {
    method: 'GET',
    headers: authHeaders(jwt, false),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || 'Failed to fetch incoming submissions');
  }

  return response.json() as Promise<SubmissionRecord[]>;
}
