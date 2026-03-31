FROM rust:1.81-slim-bookworm AS ppml-builder

WORKDIR /build

COPY PPML ./PPML

RUN cargo build \
    --manifest-path PPML/ppml_train/Cargo.toml \
    --release \
    --bin encrypt_dataset \
    --bin verify_dataset


FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PPML_DATASET_ENCRYPTOR_BIN=/usr/local/bin/encrypt_dataset \
    PPML_DATASET_KEY_CACHE=/app/PPML/.cache/dataset_keys_q16f8.bin

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgcc-s1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY blindinference/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY blindinference ./blindinference
COPY PPML ./PPML

COPY --from=ppml-builder /build/PPML/target/release/encrypt_dataset /usr/local/bin/encrypt_dataset
COPY --from=ppml-builder /build/PPML/target/release/verify_dataset /usr/local/bin/verify_dataset

RUN mkdir -p /app/PPML/.cache

WORKDIR /app/blindinference/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
