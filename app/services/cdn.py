from __future__ import annotations

import mimetypes
from urllib.parse import quote

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings


class CdnError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class Cdn:
    """Cloudflare R2 (S3 API). Public URLs go through this API's /api/files route (api/files.py) unless
    CDN_PUBLIC_URL names the bucket's own public URL."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bucket = settings.r2_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            region_name=settings.r2_region,
            aws_access_key_id=settings.r2_access_key_id.get_secret_value(),
            aws_secret_access_key=settings.r2_secret_access_key.get_secret_value(),
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.r2_endpoint and self.settings.r2_access_key_id.get_secret_value() and self.settings.r2_secret_access_key.get_secret_value())

    @property
    def prefix(self) -> str:
        """Every public file URL starts with this (keys follow it)."""
        return (self.settings.cdn_public_url or f"{self.settings.api_public_url}/api/files").rstrip("/") + "/"

    def public_url(self, key: str) -> str:
        return f"{self.prefix}{quote(key)}"

    def get(self, key: str) -> dict:
        """The stored object (S3 get_object response: Body, ContentType, ContentLength); CdnError 404 when missing."""
        try:
            return self._client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            raise CdnError("Not found.", 404) from exc
        except BotoCoreError as exc:
            raise CdnError(f"Could not read from CDN: {exc}") from exc

    def upload(self, key: str, data: bytes, content_type: str | None = None) -> str:
        guessed = content_type or mimetypes.guess_type(key)[0] or "application/octet-stream"
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=guessed,
            )
        except (BotoCoreError, ClientError) as exc:
            raise CdnError(f"Could not upload to CDN: {exc}") from exc
        return self.public_url(key)

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except (BotoCoreError, ClientError):
            pass


def file_prefixes(settings: Settings) -> tuple[str, ...]:
    """Where the platform's own files live: R2 (through this API or its public URL) and, for older files, Supabase Storage."""
    return (
        (settings.cdn_public_url or f"{settings.api_public_url}/api/files").rstrip("/") + "/",
        f"{settings.supabase_url.rstrip('/')}/storage/v1/object/public/",
    )
