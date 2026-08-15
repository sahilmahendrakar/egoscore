"""Isolate the R2 400: try plain boto3 with several client configurations."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Newer botocore sends CRC32 checksums by default, which Cloudflare R2 rejects with a 400.
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

import boto3
import pandas as pd
from botocore.config import Config

from egoscore.access import load_creds

creds = load_creds()
print("endpoint:", creds.r2_endpoint)

REPORTS = Path(__file__).resolve().parent.parent / "reports"
df = pd.read_csv(REPORTS / "slice_fold_clothes.csv")
rl2 = df[(df["lab"] == "rl2") & (df["embodiment"] == "human_bimanual")]
rl2 = rl2[rl2["zarr_processed_path"].fillna("").str.strip() != ""]
prefix = rl2.iloc[0]["zarr_processed_path"].replace("s3://rldb/", "")
print("prefix:", prefix)

for label, cfg in [
    ("default", Config(signature_version="s3v4")),
    ("virtual", Config(signature_version="s3v4", s3={"addressing_style": "virtual"})),
    ("path", Config(signature_version="s3v4", s3={"addressing_style": "path"})),
]:
    try:
        c = boto3.client(
            "s3",
            endpoint_url=creds.r2_endpoint,
            aws_access_key_id=creds.r2_access_key,
            aws_secret_access_key=creds.r2_secret_key,
            region_name="auto",
            config=cfg,
        )
        r = c.list_objects_v2(Bucket="rldb", Prefix=prefix, MaxKeys=15)
        keys = [o["Key"] for o in r.get("Contents", [])]
        print(f"\n[{label}] OK  KeyCount={r.get('KeyCount')}")
        for k in keys[:15]:
            print("   ", k)
        if keys:
            body = c.get_object(Bucket="rldb", Key=keys[0])["Body"].read()
            print(f"   get_object OK, {len(body)} bytes")
        break
    except Exception as e:
        print(f"\n[{label}] FAIL {type(e).__name__}: {str(e)[:200]}")
