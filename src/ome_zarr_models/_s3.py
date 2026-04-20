"""S3 store utilities using obstore for fast remote access."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import obstore as obs
from zarr.storage import ObjectStore

if TYPE_CHECKING:
    pass


@lru_cache(maxsize=64)
def _get_bucket_region(bucket: str) -> str:
    """Discover the AWS region for an S3 bucket using botocore.

    Parameters
    ----------
    bucket :
        S3 bucket name.

    Returns
    -------
    str
        AWS region string (e.g. ``"us-east-1"``).
    """
    import botocore.session

    session = botocore.session.get_session()
    # Use us-east-1 as the initial region for the discovery call
    client = session.create_client("s3", region_name="us-east-1")
    response = client.get_bucket_location(Bucket=bucket)
    location = response.get("LocationConstraint")
    # None means us-east-1 (the default/legacy region)
    return location or "us-east-1"


def is_s3_url(path: str) -> bool:
    """Check if a path is an S3 URL."""
    return isinstance(path, str) and path.startswith("s3://")


def make_s3_store(s3_url: str) -> ObjectStore:
    """Create an obstore-backed zarr ``ObjectStore`` from an S3 URL.

    Automatically discovers the bucket region and forwards AWS credentials
    from environment variables (compatible with ``aws-oidc exec`` and similar
    credential providers).

    Parameters
    ----------
    s3_url :
        Full S3 URL, e.g. ``"s3://my-bucket/path/to/data.ome.zarr"``.

    Returns
    -------
    ObjectStore
        A zarr-compatible store backed by obstore's Rust S3 client.
    """
    parsed = urlparse(s3_url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")

    region = _get_bucket_region(bucket)

    config: dict[str, str] = {}
    if key := os.environ.get("AWS_ACCESS_KEY_ID"):
        config["AWS_ACCESS_KEY_ID"] = key
    if secret := os.environ.get("AWS_SECRET_ACCESS_KEY"):
        config["AWS_SECRET_ACCESS_KEY"] = secret
    if token := os.environ.get("AWS_SESSION_TOKEN"):
        config["AWS_SESSION_TOKEN"] = token

    s3_store = obs.store.S3Store(bucket, prefix=prefix, config=config, region=region)
    return ObjectStore(s3_store, read_only=True)
