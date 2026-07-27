"""Sube a Google Cloud Storage los archivos de datos que la app descarga al arrancar.

Reutiliza la configuración de `.streamlit/secrets.toml` (bloques `[gcs]`,
`[gcs.dataset_blobs]` y `[gcp_service_account]`), la misma que consume `app.py`,
para no duplicar el nombre del bucket ni las credenciales.

Uso:
    python upload_to_gcs.py              # sube la BBDD y los 3 datasets
    python upload_to_gcs.py --db         # solo la base de datos
    python upload_to_gcs.py ud_spanish_gsd ancora_dep_2_0_es   # datasets concretos
    python upload_to_gcs.py --force      # re-sube aunque el objeto ya exista en el bucket
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from graph_centrality_store import DATABASE_PATH, DATASET_DEFINITIONS, REPO_ROOT

GCS_SECTION = "gcs"
GCP_SERVICE_ACCOUNT_SECTION = "gcp_service_account"


def require_secret_section(name: str) -> dict:
    if name not in st.secrets:
        sys.exit(
            f"Falta el bloque [{name}] en .streamlit/secrets.toml. "
            "Complétalo antes de subir los archivos."
        )
    return st.secrets[name]


def build_client():
    from google.cloud import storage

    if GCP_SERVICE_ACCOUNT_SECTION in st.secrets:
        from google.oauth2 import service_account

        info = dict(st.secrets[GCP_SERVICE_ACCOUNT_SECTION])
        credentials = service_account.Credentials.from_service_account_info(info)
        return storage.Client(credentials=credentials, project=info.get("project_id"))

    # Sin credenciales explícitas: Application Default Credentials.
    return storage.Client()


def dataset_archive_path(dataset_id: str) -> Path:
    # Los datasets se empaquetan como <graph_dir>.zip junto al directorio de grafos.
    graph_dir = DATASET_DEFINITIONS[dataset_id]["graph_dir"]
    return graph_dir.with_suffix(".zip")


def upload_blob(bucket, local_path: Path, blob_name: str, force: bool) -> None:
    if not local_path.exists():
        print(f"  ✗ No existe {local_path.name}; se omite.")
        return

    blob = bucket.blob(blob_name)
    if not force and blob.exists():
        print(f"  = {blob_name} ya existe en el bucket (usa --force para sobrescribir); se omite.")
        return

    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"  ↑ Subiendo {local_path.name} → gs://{bucket.name}/{blob_name} ({size_mb:.1f} MB)...")
    blob.upload_from_filename(str(local_path))
    print(f"    ✓ Listo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sube los datos de la app a GCS.")
    parser.add_argument(
        "datasets",
        nargs="*",
        help="IDs de datasets a subir (por defecto todos). Opciones: "
        + ", ".join(DATASET_DEFINITIONS),
    )
    parser.add_argument("--db", action="store_true", help="Subir solo la base de datos SQLite.")
    parser.add_argument("--force", action="store_true", help="Sobrescribir objetos existentes.")
    args = parser.parse_args()

    gcs_config = require_secret_section(GCS_SECTION)
    bucket_name = gcs_config.get("bucket")
    if not bucket_name or "REEMPLAZAR" in str(bucket_name):
        sys.exit("Configura `bucket` en el bloque [gcs] de .streamlit/secrets.toml.")

    db_blob = gcs_config.get("db_blob", DATABASE_PATH.name)
    dataset_blobs = dict(gcs_config.get("dataset_blobs", {}))

    client = build_client()
    bucket = client.bucket(bucket_name)
    print(f"Bucket destino: gs://{bucket_name}\n")

    # Decidir qué subir.
    upload_db = args.db or not args.datasets
    if args.datasets:
        selected_datasets = args.datasets
        upload_db = args.db  # con datasets explícitos, la BBDD solo si se pide con --db
        unknown = [d for d in selected_datasets if d not in DATASET_DEFINITIONS]
        if unknown:
            sys.exit(f"Datasets desconocidos: {', '.join(unknown)}")
    elif args.db:
        selected_datasets = []
    else:
        selected_datasets = list(DATASET_DEFINITIONS)

    if upload_db:
        print("Base de datos:")
        upload_blob(bucket, DATABASE_PATH, db_blob, args.force)
        print()

    if selected_datasets:
        print("Datasets:")
        for dataset_id in selected_datasets:
            blob_name = dataset_blobs.get(dataset_id, f"{dataset_id}.zip")
            upload_blob(bucket, dataset_archive_path(dataset_id), blob_name, args.force)
        print()

    print("Terminado.")


if __name__ == "__main__":
    main()
