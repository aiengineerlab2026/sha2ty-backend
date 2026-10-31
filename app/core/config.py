from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    generation_limit: int = 30  # Meshy calls and GPT Astra calls allowed per vendor account (each)
    meshy_api_key: SecretStr = SecretStr("")
    meshy_base_url: str = "https://api.meshy.ai/openapi/v1"
    meshy_request_timeout_seconds: float = 120.0
    meshy_download_timeout_seconds: float = 300.0

    openai_api_key: SecretStr = SecretStr("")
    openai_astra_model: str = "gpt-6-astra"
    # Floor plans: GPT reads room names and measurements before the OpenCV conversion ("" turns it off).
    openai_plan_model: str = "gpt-6-astra"
    openai_plan_timeout_seconds: float = 60.0
    # Match: OpenAI image editing with two reference images (POST /v1/images/edits).
    openai_image_model: str = "gpt-image-2"
    openai_image_size: str = "1536x1024"  # landscape room photo
    openai_image_quality: str = "low"  # low | medium | high (cost and time grow with it)
    openai_image_input_fidelity: str = ""  # "high" keeps reference details on models that support it (gpt-image-1.x); gpt-image-2 rejects it
    openai_image_timeout_seconds: float = 300.0
    openai_image_expected_seconds: float = 30.0  # typical time per image, for the progress percentage

    supabase_url: str = ""
    supabase_anon_key: SecretStr = SecretStr("")
    supabase_service_role_key: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("")

    # Admin console (vendor approval). Change both in production; the session secret defaults to the
    # service role key so tokens can't be forged without it.
    admin_username: str = "admin@admin.com"  # the admin signs in with this email
    admin_password: SecretStr = SecretStr("admin")
    admin_session_secret: SecretStr = SecretStr("")
    admin_session_hours: float = 12.0

    supabase_s3_endpoint: str = ""
    supabase_s3_region: str = "eu-west-2"
    supabase_s3_access_key_id: SecretStr = SecretStr("")
    supabase_s3_secret_access_key: SecretStr = SecretStr("")
    supabase_storage_bucket: str = "cads"

    cors_origins: list[str] = []
    cors_origin_regex: str | None = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"
    storage_dir: Path = BACKEND_DIR / "storage"

    max_image_bytes: int = 20 * 1024 * 1024
    max_model_bytes: int = 300 * 1024 * 1024

    @property
    def meshy_configured(self) -> bool:
        key = self.meshy_api_key.get_secret_value().strip()
        return bool(key) and not key.startswith("<")

    @property
    def astra_configured(self) -> bool:
        key = self.openai_api_key.get_secret_value().strip()
        return bool(key) and key.startswith("sk-")


@lru_cache
def get_settings() -> Settings:
    return Settings()
