import asyncio
import csv
import hashlib
import json
import logging
import os
import inspect
import re
import secrets
import time
import tempfile
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path

import jwt
import motor.motor_asyncio
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gridfs.errors import NoFile
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
from pydantic import BaseModel
from eth_account import Account
from eth_account.messages import encode_defunct

load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))

CHUNK_SIZE = 1024 * 1024
ADDRESS_PATTERN = re.compile(r"^0x[a-fA-F0-9]{40}$")
AUTH_NONCE_TTL_MINUTES = 5
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24
PPML_EXPORT_CANDIDATES = (
    Path(__file__).resolve().parents[2] / "ppml" / "model_export.json",
    Path(__file__).resolve().parents[2] / "PPML" / "model_export.json",
)
PPML_ROOT = Path(__file__).resolve().parents[2] / "PPML"
PPML_TRAIN_MANIFEST = PPML_ROOT / "ppml_train" / "Cargo.toml"
PPML_DATASET_KEY_CACHE = PPML_ROOT / ".cache" / "dataset_keys_q16f8.bin"
JWT_SECRET = os.getenv("JWT_SECRET", "blindference-dev-secret")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("blindinference")


class AuthNonceRequest(BaseModel):
    address: str
    role: str


class AuthVerifyRequest(BaseModel):
    address: str
    role: str
    nonce_id: str
    signature: str


class UserProfilePayload(BaseModel):
    display_name: str | None = None
    organization: str | None = None
    bio: str | None = None
    profile_uri: str | None = None


class DatasetManifestPayload(BaseModel):
    file_id: str
    filename: str
    lab_address: str | None = None
    model_id: str | None = None
    content_type: str | None = None
    notes: str | None = None


class SubmissionPayload(BaseModel):
    request_id: str
    model_id: str
    lab_address: str
    tx_hash: str | None = None
    status: str
    result_handle: str | None = None
    plaintext_result: str | None = None


class PublishedModelPayload(BaseModel):
    dataset_id: str
    name: str
    description: str | None = None
    price_bfhe: str | None = None
    status: str = "published_on_chain"
    on_chain_model_id: str
    tx_hash: str | None = None
    original_filename: str | None = None
    artifact_sha256: str | None = None
    metadata_uri: str | None = None
    registry_reference: str | None = None
    weight_count: int | None = None
    feature_names: list[str] | None = None
    scale: float | None = None


def serialize_linked_model(document: dict) -> dict:
    return {
        "model_id": str(document["_id"]),
        "dataset_id": document["dataset_id"],
        "file_id": document.get("file_id"),
        "lab_address": document["lab_address"],
        "name": document["name"],
        "description": document.get("description"),
        "price_bfhe": document.get("price_bfhe"),
        "status": document["status"],
        "artifact_type": document.get("artifact_type"),
        "artifact_sha256": document.get("artifact_sha256"),
        "content_type": document.get("content_type"),
        "filename": document["filename"],
        "original_filename": document.get("original_filename"),
        "on_chain_model_id": document.get("on_chain_model_id"),
        "tx_hash": document.get("tx_hash"),
        "metadata_uri": document.get("metadata_uri"),
        "registry_reference": document.get("registry_reference"),
        "weight_count": document.get("weight_count"),
        "feature_names": document.get("feature_names"),
        "scale": document.get("scale"),
        "created_at": document["created_at"],
        "updated_at": document["updated_at"],
    }


def serialize_dataset_manifest(document: dict, linked_models: list[dict]) -> dict:
    manifest = dict(document)
    manifest["dataset_id"] = str(manifest.pop("_id"))
    manifest["linked_models"] = [serialize_linked_model(model) for model in linked_models]
    manifest["linked_model_count"] = len(linked_models)
    return manifest


def normalize_address(address: str) -> str:
    if not ADDRESS_PATTERN.match(address):
        raise HTTPException(status_code=400, detail="invalid wallet address")
    return address.lower()


def validate_role(role: str) -> str:
    if role not in {"data_source", "ai_lab"}:
        raise HTTPException(status_code=400, detail="invalid role")
    return role


def build_auth_message(address: str, role: str, nonce: str, issued_at: datetime) -> str:
    issued_at_iso = issued_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return (
        "Blindference Authentication\n"
        f"Wallet: {address}\n"
        f"Role: {role}\n"
        f"Nonce: {nonce}\n"
        f"Issued At: {issued_at_iso}\n"
        "Sign this message to authenticate with Blindference."
    )


def create_access_token(address: str, role: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": address,
        "role": role,
        "exp": expires_at,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(authorization: str | None) -> dict:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")

    token = authorization.removeprefix("Bearer ").strip()
    if token == "":
        raise HTTPException(status_code=401, detail="missing bearer token")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception as error:
        raise HTTPException(status_code=401, detail="invalid bearer token") from error

    return payload


def require_subject_match(authorization: str | None, address: str) -> dict:
    payload = decode_access_token(authorization)
    if payload.get("sub") != address:
        raise HTTPException(status_code=403, detail="token subject does not match wallet address")
    return payload


def require_role(payload: dict, role: str) -> None:
    if payload.get("role") != role:
        raise HTTPException(status_code=403, detail=f"{role} role required")


def get_model_export_path() -> Path | None:
    for candidate in PPML_EXPORT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def get_gridfs_bucket(request: Request) -> AsyncIOMotorGridFSBucket:
    bucket = getattr(request.app.state, "fs", None)
    if bucket is None:
        raise HTTPException(status_code=503, detail="GridFS storage is not initialized")
    return bucket


async def maybe_await(result):
    if inspect.isawaitable(result):
        await result


def parse_label_column(label_column: str, header: list[str] | None, column_count: int) -> int:
    normalized = normalize_column_selector(label_column)
    if normalized == "":
        raise HTTPException(status_code=400, detail="label_column cannot be empty")

    if normalized == "last":
        return column_count - 1

    if header is not None:
        normalized_header = [normalize_column_selector(column_name) for column_name in header]
        if normalized in normalized_header:
            return normalized_header.index(normalized)

    try:
        label_index = int(normalized)
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="label_column must be 'last', a zero-based index, or a header name",
        ) from error

    if label_index < 0 or label_index >= column_count:
        raise HTTPException(status_code=400, detail="label_column is out of range")

    return label_index


def normalize_column_selector(value: str) -> str:
    trimmed = value.strip().lstrip("\ufeff")
    if len(trimmed) >= 2 and trimmed[0] == trimmed[-1] and trimmed[0] in {'"', "'"}:
        trimmed = trimmed[1:-1].strip()
    return trimmed.casefold()


def build_dataset_encryption_request(
    file_bytes: bytes,
    *,
    filename: str,
    label_column: str,
    has_header: bool,
) -> dict:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(status_code=400, detail="dataset upload must be valid UTF-8 CSV") from error

    reader = csv.reader(StringIO(text))
    rows = []
    for row in reader:
        normalized_row = [cell.strip() for cell in row]
        if any(cell != "" for cell in normalized_row):
            rows.append(normalized_row)

    if not rows:
        raise HTTPException(status_code=400, detail="dataset upload contained no rows")

    header = rows[0] if has_header else None
    data_rows = rows[1:] if has_header else rows
    if not data_rows:
        raise HTTPException(status_code=400, detail="dataset upload contained no data rows")

    column_count = len(data_rows[0])
    if column_count < 2:
        raise HTTPException(
            status_code=400,
            detail="dataset must contain at least one feature column and one label column",
        )

    label_index = parse_label_column(label_column, header, column_count)
    feature_names = None
    label_name = None
    if header is not None:
        feature_names = [name for index, name in enumerate(header) if index != label_index]
        label_name = header[label_index]

    feature_rows = []
    label_rows = []

    for row_number, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != column_count:
            raise HTTPException(
                status_code=400,
                detail=f"row {row_number} has {len(row)} columns, expected {column_count}",
            )

        try:
            numeric_row = [float(cell) for cell in row]
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail=f"row {row_number} contains a non-numeric value",
            ) from error

        feature_rows.append(
            [value for index, value in enumerate(numeric_row) if index != label_index]
        )
        label_rows.append([numeric_row[label_index]])

    return {
        "source_format": "csv",
        "label_column_index": label_index,
        "feature_rows": feature_rows,
        "label_rows": label_rows,
        "feature_names": feature_names,
        "label_name": label_name,
        "original_filename": filename,
    }


async def run_dataset_encryptor(request_payload: dict) -> tuple[bytes, dict]:
    key_cache_path = Path(os.getenv("PPML_DATASET_KEY_CACHE", str(PPML_DATASET_KEY_CACHE)))
    key_cache_path.parent.mkdir(parents=True, exist_ok=True)

    encryptor_binary = os.getenv("PPML_DATASET_ENCRYPTOR_BIN")

    with tempfile.TemporaryDirectory(prefix="blindinference-dataset-") as temp_dir:
        temp_path = Path(temp_dir)
        request_path = temp_path / "dataset_request.json"
        output_path = temp_path / "dataset_export.json"
        request_path.write_text(json.dumps(request_payload), encoding="utf-8")

        if encryptor_binary:
            command = [
                encryptor_binary,
                "--input",
                str(request_path),
                "--output",
                str(output_path),
                "--key-cache",
                str(key_cache_path),
            ]
        else:
            command = [
                "cargo",
                "run",
                "--manifest-path",
                str(Path(os.getenv("PPML_TRAIN_MANIFEST", str(PPML_TRAIN_MANIFEST)))),
                "--bin",
                "encrypt_dataset",
                "--",
                "--input",
                str(request_path),
                "--output",
                str(output_path),
                "--key-cache",
                str(key_cache_path),
            ]

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PPML_ROOT),
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.error(
                "dataset encryption failed",
                extra={
                    "stdout": stdout.decode("utf-8", errors="ignore"),
                    "stderr": stderr.decode("utf-8", errors="ignore"),
                },
            )
            raise HTTPException(
                status_code=500,
                detail="failed to encrypt dataset into PPML-compatible artifact",
            )

        try:
            artifact_bytes = output_path.read_bytes()
            artifact_json = json.loads(artifact_bytes.decode("utf-8"))
        except Exception as error:
            raise HTTPException(
                status_code=500,
                detail="dataset encryptor produced an unreadable artifact",
            ) from error

    return artifact_bytes, artifact_json


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting BlindInference API")
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI must be set before starting the backend")

    mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
    database = mongo_client["blindference_db"]
    gridfs_bucket = AsyncIOMotorGridFSBucket(database)

    app.state.mongo_client = mongo_client
    app.state.db = database
    app.state.fs = gridfs_bucket

    try:
        yield
    finally:
        mongo_client.close()
        logger.info("Shutting down BlindInference API")

app = FastAPI(
    title="BlindInference API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic request/response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"completed_in={process_time:.2f}ms status={response.status_code}"
    )
    return response

@app.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok"}

# Example root endpoint
@app.get("/", tags=["system"])
async def root():
    return {"message": "BlindInference backend is running"}


@app.post("/api/v1/auth/nonce", tags=["auth"])
async def create_auth_nonce(request: AuthNonceRequest):
    address = normalize_address(request.address)
    role = validate_role(request.role)
    issued_at = datetime.now(timezone.utc)
    nonce = secrets.token_urlsafe(24)
    message = build_auth_message(address, role, nonce, issued_at)

    auth_record = {
        "address": address,
        "role": role,
        "nonce": nonce,
        "issued_at": issued_at,
        "expires_at": issued_at + timedelta(minutes=AUTH_NONCE_TTL_MINUTES),
        "used": False,
    }

    result = await app.state.db.auth_nonces.insert_one(auth_record)

    return {
        "nonce_id": str(result.inserted_id),
        "message": message,
        "expires_in_seconds": AUTH_NONCE_TTL_MINUTES * 60,
    }


@app.post("/api/v1/auth/verify", tags=["auth"])
async def verify_auth_signature(request: AuthVerifyRequest):
    address = normalize_address(request.address)
    role = validate_role(request.role)

    try:
        nonce_object_id = ObjectId(request.nonce_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid nonce_id") from error

    nonce_record = await app.state.db.auth_nonces.find_one({"_id": nonce_object_id})
    if nonce_record is None:
        raise HTTPException(status_code=404, detail="auth nonce not found")

    if nonce_record.get("used"):
        raise HTTPException(status_code=409, detail="auth nonce already used")

    now = datetime.now(timezone.utc)
    expires_at = nonce_record["expires_at"]
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if now > expires_at:
        raise HTTPException(status_code=410, detail="auth nonce expired")

    if nonce_record["address"] != address:
        raise HTTPException(status_code=400, detail="nonce address mismatch")
    if nonce_record["role"] != role:
        raise HTTPException(status_code=400, detail="nonce role mismatch")

    message = build_auth_message(
        nonce_record["address"],
        nonce_record["role"],
        nonce_record["nonce"],
        nonce_record["issued_at"].replace(tzinfo=timezone.utc)
        if nonce_record["issued_at"].tzinfo is None
        else nonce_record["issued_at"],
    )

    try:
        recovered_address = Account.recover_message(
            encode_defunct(text=message),
            signature=request.signature,
        ).lower()
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid signature payload") from error

    if recovered_address != address:
        raise HTTPException(status_code=401, detail="signature does not match wallet address")

    await app.state.db.auth_nonces.update_one(
        {"_id": nonce_object_id},
        {"$set": {"used": True, "verified_at": now}},
    )

    await app.state.db.users.update_one(
        {"address": address},
        {
            "$set": {
                "address": address,
                "role": role,
                "last_authenticated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    token = create_access_token(address, role)
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": role,
    }


@app.get("/api/v1/profile/{address}", tags=["profile"])
async def get_user_profile(address: str, authorization: str | None = Header(default=None)):
    normalized_address = normalize_address(address)
    token_payload = require_subject_match(authorization, normalized_address)

    user = await app.state.db.users.find_one({"address": normalized_address})
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")

    profile = await app.state.db.user_profiles.find_one({"address": normalized_address}) or {}
    return {
        "address": normalized_address,
        "role": user.get("role", token_payload.get("role")),
        "display_name": profile.get("display_name"),
        "organization": profile.get("organization"),
        "bio": profile.get("bio"),
        "profile_uri": profile.get("profile_uri"),
        "created_at": profile.get("created_at"),
        "updated_at": profile.get("updated_at"),
    }


@app.put("/api/v1/profile/{address}", tags=["profile"])
async def update_user_profile(
    address: str,
    payload: UserProfilePayload,
    authorization: str | None = Header(default=None),
):
    normalized_address = normalize_address(address)
    token_payload = require_subject_match(authorization, normalized_address)
    now = datetime.now(timezone.utc)
    profile_uri = payload.profile_uri.strip() if payload.profile_uri else None

    if profile_uri is None:
        raise HTTPException(status_code=400, detail="profile_uri is required")

    sanitized_profile = {
        "display_name": None,
        "organization": None,
        "bio": None,
        "profile_uri": profile_uri,
    }

    await app.state.db.users.update_one(
        {"address": normalized_address},
        {
            "$set": {
                "address": normalized_address,
                "role": token_payload.get("role"),
                "last_authenticated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    await app.state.db.user_profiles.update_one(
        {"address": normalized_address},
        {
            "$set": {
                **sanitized_profile,
                "updated_at": now,
            },
            "$setOnInsert": {
                "address": normalized_address,
                "created_at": now,
            },
        },
        upsert=True,
    )

    return {
        "address": normalized_address,
        "role": token_payload.get("role"),
        **sanitized_profile,
        "updated_at": now,
    }


@app.post("/api/v1/datasets/manifest", tags=["datasets"])
async def create_dataset_manifest(
    payload: DatasetManifestPayload,
    authorization: str | None = Header(default=None),
):
    token_payload = decode_access_token(authorization)
    require_role(token_payload, "data_source")
    owner_address = normalize_address(token_payload["sub"])
    lab_address = normalize_address(payload.lab_address) if payload.lab_address else None
    now = datetime.now(timezone.utc)

    manifest = {
        "file_id": payload.file_id,
        "filename": payload.filename.strip(),
        "owner_address": owner_address,
        "lab_address": lab_address,
        "model_id": payload.model_id.strip() if payload.model_id else None,
        "content_type": payload.content_type.strip() if payload.content_type else None,
        "notes": payload.notes.strip() if payload.notes else None,
        "status": "uploaded",
        "visibility": "restricted" if lab_address else "ai_labs",
        "created_at": now,
        "updated_at": now,
    }

    result = await app.state.db.dataset_manifests.insert_one(manifest)
    return {
        "dataset_id": str(result.inserted_id),
        **manifest,
    }


@app.get("/api/v1/datasets/outgoing/{address}", tags=["datasets"])
async def list_outgoing_datasets(address: str, authorization: str | None = Header(default=None)):
    normalized_address = normalize_address(address)
    token_payload = require_subject_match(authorization, normalized_address)
    require_role(token_payload, "data_source")

    cursor = app.state.db.dataset_manifests.find(
        {"owner_address": normalized_address},
    ).sort("created_at", -1)
    manifests = await cursor.to_list(length=100)
    serialized_manifests = []
    for manifest in manifests:
        dataset_id = str(manifest["_id"])
        linked_model_cursor = app.state.db.model_artifacts.find(
            {"dataset_id": dataset_id},
        ).sort("created_at", -1)
        linked_models = await linked_model_cursor.to_list(length=50)
        serialized_manifests.append(serialize_dataset_manifest(manifest, linked_models))
    return serialized_manifests


@app.get("/api/v1/datasets/incoming/{address}", tags=["datasets"])
async def list_incoming_datasets(address: str, authorization: str | None = Header(default=None)):
    normalized_address = normalize_address(address)
    token_payload = require_subject_match(authorization, normalized_address)
    require_role(token_payload, "ai_lab")

    cursor = app.state.db.dataset_manifests.find(
        {
            "$or": [
                {"lab_address": normalized_address},
                {"visibility": "ai_labs"},
            ]
        },
    ).sort("created_at", -1)
    manifests = await cursor.to_list(length=100)
    serialized_manifests = []
    for manifest in manifests:
        dataset_id = str(manifest["_id"])
        linked_model_cursor = app.state.db.model_artifacts.find(
            {"dataset_id": dataset_id},
        ).sort("created_at", -1)
        linked_models = await linked_model_cursor.to_list(length=50)
        serialized_manifests.append(serialize_dataset_manifest(manifest, linked_models))
    return serialized_manifests


@app.get("/api/v1/datasets/catalog", tags=["datasets"])
async def list_dataset_catalog(authorization: str | None = Header(default=None)):
    token_payload = decode_access_token(authorization)
    role = token_payload.get("role")
    if role not in {"data_source", "ai_lab"}:
        raise HTTPException(status_code=403, detail="authenticated role required")

    cursor = app.state.db.dataset_manifests.find(
        {"visibility": "ai_labs"},
    ).sort("created_at", -1)
    manifests = await cursor.to_list(length=100)
    serialized_manifests = []
    for manifest in manifests:
        dataset_id = str(manifest["_id"])
        linked_model_cursor = app.state.db.model_artifacts.find(
            {"dataset_id": dataset_id},
        ).sort("created_at", -1)
        linked_models = await linked_model_cursor.to_list(length=50)
        serialized_manifests.append(serialize_dataset_manifest(manifest, linked_models))
    return serialized_manifests


@app.post("/api/v1/submissions", tags=["submissions"])
async def upsert_submission(
    payload: SubmissionPayload,
    authorization: str | None = Header(default=None),
):
    token_payload = decode_access_token(authorization)
    require_role(token_payload, "data_source")
    owner_address = normalize_address(token_payload["sub"])
    lab_address = normalize_address(payload.lab_address)
    now = datetime.now(timezone.utc)

    update = {
        "$set": {
            "owner_address": owner_address,
            "lab_address": lab_address,
            "model_id": payload.model_id.strip(),
            "tx_hash": payload.tx_hash.strip() if payload.tx_hash else None,
            "status": payload.status.strip(),
            "result_handle": payload.result_handle.strip() if payload.result_handle else None,
            "plaintext_result": payload.plaintext_result.strip() if payload.plaintext_result else None,
            "updated_at": now,
        },
        "$setOnInsert": {
            "request_id": payload.request_id.strip(),
            "created_at": now,
        },
    }

    await app.state.db.inference_submissions.update_one(
        {"request_id": payload.request_id.strip(), "owner_address": owner_address},
        update,
        upsert=True,
    )

    document = await app.state.db.inference_submissions.find_one(
        {"request_id": payload.request_id.strip(), "owner_address": owner_address}
    )
    if document is None:
        raise HTTPException(status_code=500, detail="failed to persist submission metadata")

    document["submission_id"] = str(document.pop("_id"))
    return document


@app.get("/api/v1/submissions/outgoing/{address}", tags=["submissions"])
async def list_outgoing_submissions(address: str, authorization: str | None = Header(default=None)):
    normalized_address = normalize_address(address)
    token_payload = require_subject_match(authorization, normalized_address)
    require_role(token_payload, "data_source")

    cursor = app.state.db.inference_submissions.find(
        {"owner_address": normalized_address},
    ).sort("created_at", -1)
    submissions = await cursor.to_list(length=100)
    for submission in submissions:
        submission["submission_id"] = str(submission.pop("_id"))
    return submissions


@app.get("/api/v1/submissions/incoming/{address}", tags=["submissions"])
async def list_incoming_submissions(address: str, authorization: str | None = Header(default=None)):
    normalized_address = normalize_address(address)
    token_payload = require_subject_match(authorization, normalized_address)
    require_role(token_payload, "ai_lab")

    cursor = app.state.db.inference_submissions.find(
        {"lab_address": normalized_address},
    ).sort("created_at", -1)
    submissions = await cursor.to_list(length=100)
    for submission in submissions:
        submission["submission_id"] = str(submission.pop("_id"))
    return submissions


@app.post("/api/v1/dataset/upload", tags=["dataset"])
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    fs = get_gridfs_bucket(request)

    upload_stream = fs.open_upload_stream(
        file.filename or "encrypted_dataset.bin",
        chunk_size_bytes=CHUNK_SIZE,
        metadata={
            "content_type": file.content_type or "application/octet-stream",
        },
    )

    try:
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            await upload_stream.write(chunk)
    except Exception as error:
        await maybe_await(upload_stream.abort())
        raise HTTPException(status_code=500, detail=f"failed to stream dataset into GridFS: {error}") from error
    finally:
        await file.close()

    await maybe_await(upload_stream.close())
    return {"file_id": str(upload_stream._id)}


@app.post("/api/v1/datasets/encrypt-upload", tags=["datasets"])
async def encrypt_and_upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    label_column: str = Form(default="last"),
    has_header: bool = Form(default=False),
    notes: str | None = Form(default=None),
    authorization: str | None = Header(default=None),
):
    token_payload = decode_access_token(authorization)
    require_role(token_payload, "data_source")
    owner_address = normalize_address(token_payload["sub"])

    try:
        file_bytes = await file.read()
        request_payload = build_dataset_encryption_request(
            file_bytes,
            filename=file.filename or "dataset.csv",
            label_column=label_column,
            has_header=has_header,
        )
        artifact_bytes, artifact_json = await run_dataset_encryptor(request_payload)
    finally:
        await file.close()

    fs = get_gridfs_bucket(request)
    now = datetime.now(timezone.utc)
    artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()
    artifact_filename = f"{Path(file.filename or 'dataset').stem}_ppml_dataset.json"

    upload_stream = fs.open_upload_stream(
        artifact_filename,
        chunk_size_bytes=CHUNK_SIZE,
        metadata={
            "content_type": "application/json",
            "artifact_type": artifact_json.get("artifact_type"),
            "owner_address": owner_address,
            "sha256": artifact_hash,
        },
    )

    try:
        await upload_stream.write(artifact_bytes)
    except Exception as error:
        await maybe_await(upload_stream.abort())
        raise HTTPException(status_code=500, detail=f"failed to persist encrypted dataset: {error}") from error

    await maybe_await(upload_stream.close())

    metadata = artifact_json["metadata"]
    encrypted_tensors = artifact_json["encrypted_tensors"]
    manifest = {
        "file_id": str(upload_stream._id),
        "filename": artifact_filename,
        "original_filename": file.filename or "dataset.csv",
        "owner_address": owner_address,
        "lab_address": None,
        "model_id": None,
        "content_type": "application/json",
        "notes": notes.strip() if notes else None,
        "status": "encrypted",
        "visibility": "ai_labs",
        "artifact_type": artifact_json["artifact_type"],
        "encryption_scheme": metadata["encryption_scheme"],
        "source_format": metadata["source_format"],
        "artifact_sha256": artifact_hash,
        "row_count": metadata["row_count"],
        "feature_count": metadata["feature_count"],
        "label_count": metadata["label_count"],
        "label_column_index": metadata["label_column_index"],
        "feature_names": metadata.get("feature_names"),
        "label_name": metadata.get("label_name"),
        "quantization": metadata["quantization"],
        "tensor_artifacts": {
            "features": {
                "rows": encrypted_tensors["features"]["rows"],
                "cols": encrypted_tensors["features"]["cols"],
                "encrypted_byte_len": len(encrypted_tensors["features"]["bytes"]),
            },
            "labels": {
                "rows": encrypted_tensors["labels"]["rows"],
                "cols": encrypted_tensors["labels"]["cols"],
                "encrypted_byte_len": len(encrypted_tensors["labels"]["bytes"]),
            },
        },
        "created_at": now,
        "updated_at": now,
    }

    result = await app.state.db.dataset_manifests.insert_one(manifest)
    inserted_manifest = await app.state.db.dataset_manifests.find_one({"_id": result.inserted_id})
    if inserted_manifest is None:
        raise HTTPException(status_code=500, detail="failed to persist dataset manifest")

    return serialize_dataset_manifest(inserted_manifest, [])


@app.get("/api/v1/dataset/download/{file_id}", tags=["dataset"])
async def download_dataset(
    file_id: str,
    request: Request,
    authorization: str | None = Header(default=None),
):
    fs = get_gridfs_bucket(request)
    token_payload = decode_access_token(authorization)
    requester_address = normalize_address(token_payload["sub"])
    requester_role = token_payload.get("role")

    try:
        object_id = ObjectId(file_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid GridFS file_id") from error

    file_document = await app.state.db.fs.files.find_one({"_id": object_id})
    if file_document is None:
        raise HTTPException(status_code=404, detail="dataset not found")

    metadata = file_document.get("metadata", {}) or {}
    artifact_type = metadata.get("artifact_type")
    owner_address = metadata.get("owner_address")
    lab_address = metadata.get("lab_address")

    if artifact_type == "ppml_encrypted_dataset":
        allowed = requester_role == "ai_lab" or owner_address == requester_address
        if not allowed:
            raise HTTPException(status_code=403, detail="not authorized to download this dataset artifact")
    elif artifact_type == "ppml_encrypted_model":
        if requester_role != "ai_lab" or lab_address != requester_address:
            raise HTTPException(status_code=403, detail="not authorized to download this model artifact")

    try:
        download_stream = await fs.open_download_stream(object_id)
    except NoFile as error:
        raise HTTPException(status_code=404, detail="dataset not found") from error

    async def iter_chunks():
        try:
            while True:
                chunk = await download_stream.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
        finally:
            await maybe_await(download_stream.close())

    return StreamingResponse(
        iter_chunks(),
        media_type=metadata.get("content_type") or "application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{file_document.get("filename", f"{file_id}.bin")}"'
        },
    )


@app.post("/api/v1/models/upload", tags=["models"])
async def upload_model_artifact(
    request: Request,
    file: UploadFile = File(...),
    dataset_id: str = Form(...),
    name: str = Form(...),
    description: str | None = Form(default=None),
    price_bfhe: str | None = Form(default=None),
    status: str = Form(default="uploaded"),
    on_chain_model_id: str | None = Form(default=None),
    authorization: str | None = Header(default=None),
):
    token_payload = decode_access_token(authorization)
    require_role(token_payload, "ai_lab")
    lab_address = normalize_address(token_payload["sub"])

    try:
        dataset_object_id = ObjectId(dataset_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid dataset_id") from error

    dataset = await app.state.db.dataset_manifests.find_one({"_id": dataset_object_id})
    if dataset is None:
        raise HTTPException(status_code=404, detail="linked dataset not found")
    if dataset.get("visibility") != "ai_labs" and dataset.get("lab_address") != lab_address:
        raise HTTPException(status_code=403, detail="dataset is not available to this AI lab")
    if dataset.get("artifact_type") != "ppml_encrypted_dataset":
        raise HTTPException(status_code=400, detail="linked dataset is not a PPML-compatible dataset artifact")

    trimmed_name = name.strip()
    if trimmed_name == "":
        raise HTTPException(status_code=400, detail="model name is required")

    trimmed_status = status.strip() or "uploaded"
    trimmed_description = description.strip() if description else None
    trimmed_price = price_bfhe.strip() if price_bfhe else None
    trimmed_on_chain_model_id = on_chain_model_id.strip() if on_chain_model_id else None

    file_bytes = await file.read()
    await file.close()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="model artifact file is empty")

    artifact_hash = hashlib.sha256(file_bytes).hexdigest()
    fs = get_gridfs_bucket(request)
    now = datetime.now(timezone.utc)

    upload_stream = fs.open_upload_stream(
        file.filename or "encrypted_model.bin",
        chunk_size_bytes=CHUNK_SIZE,
        metadata={
            "content_type": file.content_type or "application/octet-stream",
            "artifact_type": "ppml_encrypted_model",
            "lab_address": lab_address,
            "dataset_id": dataset_id,
            "sha256": artifact_hash,
        },
    )

    try:
        await upload_stream.write(file_bytes)
    except Exception as error:
        await maybe_await(upload_stream.abort())
        raise HTTPException(status_code=500, detail=f"failed to persist model artifact: {error}") from error

    await maybe_await(upload_stream.close())

    document = {
        "dataset_id": dataset_id,
        "file_id": str(upload_stream._id),
        "lab_address": lab_address,
        "name": trimmed_name,
        "description": trimmed_description,
        "price_bfhe": trimmed_price,
        "status": trimmed_status,
        "artifact_type": "ppml_encrypted_model",
        "artifact_sha256": artifact_hash,
        "content_type": file.content_type or "application/octet-stream",
        "filename": file.filename or "encrypted_model.bin",
        "original_filename": file.filename or "encrypted_model.bin",
        "on_chain_model_id": trimmed_on_chain_model_id,
        "created_at": now,
        "updated_at": now,
    }
    result = await app.state.db.model_artifacts.insert_one(document)

    await app.state.db.dataset_manifests.update_one(
        {"_id": dataset_object_id},
        {"$set": {"updated_at": now}},
    )
    inserted_model = await app.state.db.model_artifacts.find_one({"_id": result.inserted_id})
    if inserted_model is None:
        raise HTTPException(status_code=500, detail="failed to persist model artifact metadata")

    return serialize_linked_model(inserted_model)


@app.post("/api/v1/models/publish", tags=["models"])
async def publish_model_record(
    payload: PublishedModelPayload,
    authorization: str | None = Header(default=None),
):
    token_payload = decode_access_token(authorization)
    require_role(token_payload, "ai_lab")
    lab_address = normalize_address(token_payload["sub"])

    try:
        dataset_object_id = ObjectId(payload.dataset_id)
    except Exception as error:
        raise HTTPException(status_code=400, detail="invalid dataset_id") from error

    dataset = await app.state.db.dataset_manifests.find_one({"_id": dataset_object_id})
    if dataset is None:
        raise HTTPException(status_code=404, detail="linked dataset not found")
    if dataset.get("visibility") != "ai_labs" and dataset.get("lab_address") != lab_address:
        raise HTTPException(status_code=403, detail="dataset is not available to this AI lab")
    if dataset.get("artifact_type") != "ppml_encrypted_dataset":
        raise HTTPException(status_code=400, detail="linked dataset is not a PPML-compatible dataset artifact")

    trimmed_name = payload.name.strip()
    if trimmed_name == "":
        raise HTTPException(status_code=400, detail="model name is required")

    trimmed_status = payload.status.strip() or "published_on_chain"
    trimmed_description = payload.description.strip() if payload.description else None
    trimmed_price = payload.price_bfhe.strip() if payload.price_bfhe else None
    trimmed_on_chain_model_id = payload.on_chain_model_id.strip()
    if trimmed_on_chain_model_id == "":
        raise HTTPException(status_code=400, detail="on_chain_model_id is required")

    existing_model = await app.state.db.model_artifacts.find_one(
        {
            "lab_address": lab_address,
            "on_chain_model_id": trimmed_on_chain_model_id,
        }
    )
    if existing_model is not None:
        raise HTTPException(status_code=409, detail="this on-chain model id is already linked to a published model record")

    registry_reference = payload.registry_reference.strip() if payload.registry_reference else trimmed_name
    metadata_uri = payload.metadata_uri.strip() if payload.metadata_uri else None
    original_filename = payload.original_filename.strip() if payload.original_filename else None
    artifact_sha256 = payload.artifact_sha256.strip() if payload.artifact_sha256 else None
    tx_hash = payload.tx_hash.strip() if payload.tx_hash else None

    feature_names = None
    if payload.feature_names:
        feature_names = [feature_name.strip() for feature_name in payload.feature_names if feature_name.strip() != ""]

    now = datetime.now(timezone.utc)
    document = {
        "dataset_id": payload.dataset_id,
        "file_id": None,
        "lab_address": lab_address,
        "name": trimmed_name,
        "description": trimmed_description,
        "price_bfhe": trimmed_price,
        "status": trimmed_status,
        "artifact_type": "cofhe_frontend_published_model",
        "artifact_sha256": artifact_sha256,
        "content_type": "application/json",
        "filename": original_filename or f"{trimmed_name}.json",
        "original_filename": original_filename,
        "on_chain_model_id": trimmed_on_chain_model_id,
        "tx_hash": tx_hash,
        "metadata_uri": metadata_uri,
        "registry_reference": registry_reference,
        "weight_count": payload.weight_count,
        "feature_names": feature_names,
        "scale": payload.scale,
        "created_at": now,
        "updated_at": now,
    }

    result = await app.state.db.model_artifacts.insert_one(document)
    await app.state.db.dataset_manifests.update_one(
        {"_id": dataset_object_id},
        {"$set": {"updated_at": now}},
    )
    inserted_model = await app.state.db.model_artifacts.find_one({"_id": result.inserted_id})
    if inserted_model is None:
        raise HTTPException(status_code=500, detail="failed to persist published model metadata")

    return serialize_linked_model(inserted_model)


@app.get("/api/v1/models/by-lab/{address}", tags=["models"])
async def list_models_by_lab(address: str, authorization: str | None = Header(default=None)):
    normalized_address = normalize_address(address)
    token_payload = require_subject_match(authorization, normalized_address)
    require_role(token_payload, "ai_lab")

    cursor = app.state.db.model_artifacts.find(
        {"lab_address": normalized_address},
    ).sort("created_at", -1)
    models = await cursor.to_list(length=100)
    return [serialize_linked_model(model) for model in models]


@app.get("/api/v1/models/catalog", tags=["models"])
async def list_model_catalog(authorization: str | None = Header(default=None)):
    token_payload = decode_access_token(authorization)
    role = token_payload.get("role")
    if role not in {"data_source", "ai_lab"}:
        raise HTTPException(status_code=403, detail="authenticated role required")

    cursor = app.state.db.model_artifacts.find({}).sort("created_at", -1)
    models = await cursor.to_list(length=200)
    return [serialize_linked_model(model) for model in models]


@app.get("/api/v1/models/by-dataset/{dataset_id}", tags=["models"])
async def list_models_by_dataset(dataset_id: str, authorization: str | None = Header(default=None)):
    decode_access_token(authorization)

    cursor = app.state.db.model_artifacts.find(
        {"dataset_id": dataset_id},
    ).sort("created_at", -1)
    models = await cursor.to_list(length=100)
    return [serialize_linked_model(model) for model in models]


@app.get("/api/v1/model/status", tags=["model"])
async def model_status():
    model_export_path = get_model_export_path()
    if model_export_path is None:
        return {"status": "processing"}

    with model_export_path.open("r", encoding="utf-8") as export_file:
        return json.load(export_file)
