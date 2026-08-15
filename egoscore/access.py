"""EgoVerse data access, standalone.

Deliberately does not import ``egomimic`` — that pulls torch/lightning and a
GPU-shaped dependency tree we do not need to read zarr episodes. We talk to the
same Postgres episode table and the same R2 bucket directly.

Credentials follow the documented public path from the EgoVerse README: the
read-only IAM keys published there resolve the R2 and Postgres secrets out of
AWS Secrets Manager.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import boto3

REGION = "us-east-2"
BUCKET = "rldb"

DB_SECRETS = ["rds/appdb/appuser", "rds/appdb/appuser-readonly"]
R2_SECRETS = ["r2/rldb/credentials", "r2/rldb/public/credentials"]

REPO_ROOT = Path(__file__).resolve().parent.parent

# The EgoVerse README publishes read-only IAM keys for public dataset access. We look
# for them in a few conventional spots rather than vendoring them into this repo.
CRED_SEARCH_PATHS = [
    REPO_ROOT / ".egoverse_aws_credentials",
    REPO_ROOT.parent / "egoverse" / ".egoverse_aws_credentials",
    Path.home() / ".egoverse_aws_credentials",
]


def _bootstrap_aws_env() -> None:
    """Point boto3 at an EgoVerse credentials file if the user has no ambient creds.

    Also disables botocore's default request checksums: recent botocore sends CRC32
    on every request, and Cloudflare R2 rejects those with an opaque HTTP 400.
    """
    os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
    os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
    if not os.environ.get("AWS_SHARED_CREDENTIALS_FILE"):
        for path in CRED_SEARCH_PATHS:
            if path.exists():
                os.environ["AWS_SHARED_CREDENTIALS_FILE"] = str(path)
                break
    os.environ.setdefault("AWS_DEFAULT_REGION", REGION)


def _first_resolvable(client, names: list[str]) -> dict:
    """Return the first secret in ``names`` we are allowed to read.

    The admin secret is tried before the public read-only one, mirroring
    setup_secret.sh: a user with fuller credentials should get the fuller secret.
    """
    errors = {}
    for name in names:
        try:
            return json.loads(client.get_secret_value(SecretId=name)["SecretString"])
        except Exception as e:  # noqa: BLE001 - we genuinely want to try the next one
            errors[name] = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"Could not read any of {names}. Errors: {errors}")


@dataclass
class EgoVerseCreds:
    db_host: str
    db_name: str
    db_user: str
    db_password: str
    db_port: int
    r2_access_key: str
    r2_secret_key: str
    r2_endpoint: str


def load_creds() -> EgoVerseCreds:
    _bootstrap_aws_env()
    sm = boto3.client("secretsmanager", region_name=REGION)

    db = _first_resolvable(sm, DB_SECRETS)
    r2 = _first_resolvable(sm, R2_SECRETS)

    return EgoVerseCreds(
        db_host=db.get("host") or db["HOST"],
        db_name=db.get("dbname", db.get("DBNAME", "appdb")),
        db_user=db.get("username") or db.get("user") or db["USER"],
        db_password=db.get("password") or db["PASSWORD"],
        db_port=int(db.get("port", 5432)),
        r2_access_key=r2["access_key_id"],
        r2_secret_key=r2["secret_access_key"],
        r2_endpoint=r2["endpoint_url"],
    )


def episode_table(creds: EgoVerseCreds | None = None):
    """Pull the full episode table as a DataFrame."""
    import pandas as pd
    from sqlalchemy import URL, create_engine

    creds = creds or load_creds()
    engine = create_engine(
        URL.create(
            "postgresql+psycopg",
            username=creds.db_user,
            password=creds.db_password,
            host=creds.db_host,
            port=creds.db_port,
            database=creds.db_name,
            query={"sslmode": "require"},
        ),
        pool_pre_ping=True,
    )
    # The episode table lives in the 'app' schema, named 'episodes' (plural).
    return pd.read_sql("SELECT * FROM app.episodes", engine)


def r2_client(creds: EgoVerseCreds | None = None):
    from botocore.config import Config

    creds = creds or load_creds()
    _bootstrap_aws_env()
    return boto3.client(
        "s3",
        endpoint_url=creds.r2_endpoint,
        aws_access_key_id=creds.r2_access_key,
        aws_secret_access_key=creds.r2_secret_key,
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            max_pool_connections=64,
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def r2_fs(creds: EgoVerseCreds | None = None):
    """fsspec filesystem over R2, for opening zarr stores lazily."""
    import s3fs

    creds = creds or load_creds()
    _bootstrap_aws_env()
    return s3fs.S3FileSystem(
        key=creds.r2_access_key,
        secret=creds.r2_secret_key,
        client_kwargs={"endpoint_url": creds.r2_endpoint, "region_name": "auto"},
        config_kwargs={"signature_version": "s3v4"},
    )
