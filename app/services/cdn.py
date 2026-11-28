from __future__ import annotations

import io
import mimetypes
from contextvars import ContextVar
from urllib.parse import quote

import boto3
from boto3.exceptions import S3UploadFailedError
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import Settings


class CdnError(Exception):
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# scheme://host of the request being served (main.public_origin): with API_PUBLIC_URL unset, file URLs name the host
# the caller actually reached, never a localhost default. The Host header only: on Cloud Run it is the service's own
# domain (requests are routed by it); X-Forwarded-Host is set by the caller and is not trusted.
REQUEST_ORIGIN: ContextVar[str | None] = ContextVar("request_origin", default=None)
LOCAL_ORIGIN = "http://localhost:8000"  # scripts and tests: no request, no setting


def api_origin(settings: Settings) -> str:
    return (settings.api_public_url or REQUEST_ORIGIN.get() or LOCAL_ORIGIN).rstrip("/")


UPLOAD = TransferConfig(multipart_threshold=8 * 1024 * 1024, multipart_chunksize=8 * 1024 * 1024, max_concurrency=8, use_threads=True)


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
        return (self.settings.cdn_public_url or f"{api_origin(self.settings)}/api/files").rstrip("/") + "/"

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
            # Large files in parallel parts (generated models are tens of MB): several connections at once instead of
            # one long stream; small files stay a single request.
            self._client.upload_fileobj(io.BytesIO(data), self.bucket, key, ExtraArgs={"ContentType": guessed}, Config=UPLOAD)
        except (BotoCoreError, ClientError, S3UploadFailedError) as exc:
            raise CdnError(f"Could not upload to CDN: {exc}") from exc
        return self.public_url(key)

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except (BotoCoreError, ClientError):
            pass


def file_key(url: str | None, settings: Settings) -> str | None:
    """The storage key of one of the platform's own R2 files, from any URL form it was saved under (this API's
    /api/files route on whatever host, or CDN_PUBLIC_URL). None for anything else."""
    from urllib.parse import unquote, urlsplit

    if not url or ".." in url:
        return None
    public = (settings.cdn_public_url or "").rstrip("/") + "/"
    if settings.cdn_public_url and url.startswith(public):
        return unquote(url[len(public):].split("?", 1)[0])
    path = urlsplit(url).path
    return unquote(path.split("/api/files/", 1)[1]) if "/api/files/" in path else None


def file_prefixes(settings: Settings) -> tuple[str, ...]:
    """Where the platform's own files live: R2 (through this API or its public URL) and, for older files, Supabase Storage."""
    return (
        (settings.cdn_public_url or f"{api_origin(settings)}/api/files").rstrip("/") + "/",
        f"{settings.supabase_url.rstrip('/')}/storage/v1/object/public/",
    )
