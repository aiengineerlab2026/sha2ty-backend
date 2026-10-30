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
    def __init__(self, settings: Settings):
        self.settings = settings
        self.bucket = settings.supabase_storage_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.supabase_s3_endpoint,
            region_name=settings.supabase_s3_region,
            aws_access_key_id=settings.supabase_s3_access_key_id.get_secret_value(),
            aws_secret_access_key=settings.supabase_s3_secret_access_key.get_secret_value(),
            config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
        )

    @property
    def configured(self) -> bool:
        return bool(
            self.settings.supabase_s3_access_key_id.get_secret_value()
            and self.settings.supabase_s3_secret_access_key.get_secret_value()
            and self.settings.supabase_url
        )

    def public_url(self, key: str) -> str:
        base = self.settings.supabase_url.rstrip("/")
        return f"{base}/storage/v1/object/public/{self.bucket}/{quote(key)}"

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
