"""Chroma snapshot sync to S3 — hydrate on boot, persist after ingest."""

from __future__ import annotations

import logging
from pathlib import Path

from tap_rag.config import Settings

logger = logging.getLogger(__name__)


def _client(settings: Settings):
    import boto3

    return boto3.client("s3", region_name=settings.aws_region)


def _prefix(settings: Settings) -> str:
    prefix = settings.s3_chroma_prefix.lstrip("/")
    return prefix if prefix.endswith("/") else f"{prefix}/"


def pull_chroma_from_s3(settings: Settings) -> bool:
    """Download the Chroma persist dir from S3. Returns True if any objects were pulled."""
    if not settings.chroma_s3_sync:
        return False
    dest = Path(settings.chroma_persist_dir)
    dest.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(settings)
    try:
        client = _client(settings)
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=settings.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                rel = key[len(prefix) :].lstrip("/")
                if not rel:
                    continue
                keys.append(key)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(settings.s3_bucket, key, str(target))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chroma S3 pull failed (%s); continuing with local index", exc)
        return False
    if not keys:
        logger.info("No Chroma snapshot at s3://%s/%s", settings.s3_bucket, prefix)
        return False
    logger.info("Pulled %d Chroma objects from s3://%s/%s", len(keys), settings.s3_bucket, prefix)
    return True


def push_chroma_to_s3(settings: Settings) -> None:
    """Upload the local persist dir to s3://bucket/prefix."""
    if not settings.chroma_s3_sync:
        return
    src = Path(settings.chroma_persist_dir)
    if not src.exists():
        logger.warning("Chroma persist dir %s missing — skip S3 push", src)
        return
    prefix = _prefix(settings)
    try:
        client = _client(settings)
        count = 0
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src).as_posix()
            client.upload_file(str(path), settings.s3_bucket, f"{prefix}{rel}")
            count += 1
        logger.info("Pushed %d Chroma files to s3://%s/%s", count, settings.s3_bucket, prefix)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Chroma S3 push failed: %s", exc)
