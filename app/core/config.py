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
    # The platform's own vendor (admin-added catalogue models): its system user's email (services/platform.py).
    platform_vendor_email: str = "studio@sha2ty.app"
    # After image-to-3D, split the model into its named parts with Meshy Auto Split (10 credits a task).
    meshy_auto_split: bool = True

    openai_api_key: SecretStr = SecretStr("")
    openai_astra_model: str = "gpt-6-astra"
    # ponytail: GPT Sol runs through the same code as Astra, only the model differs; set OPENAI_SOL_MODEL if OpenAI names it differently.
    openai_sol_model: str = "gpt-6-sol"
    # Floor plans: GPT reads room names and measurements before the OpenCV conversion ("" turns it off).
    # Default vision model (floor plans, vendor photos) until the admin picks one (services/vision.py):
    # gemini-* → Google Gemini (GEMINI_API_KEY), otherwise OpenAI Responses (OPENAI_API_KEY).
    plan_vision_model: str = "gemini-3.1-pro-preview"
    gemini_api_key: SecretStr = SecretStr("")
    openai_plan_timeout_seconds: float = 240.0  # GPT-6 Astra can take 1–2 minutes on a full plan
    # Floor plans: the same upload reuses its vision answer and its finished 3D plan (model_cache);
    # off = every upload is read and converted again. Vendor photos are always cached.
    plan_vision_cache: bool = True
    openai_plan_expected_seconds: float = 25.0  # typical answer time, for the progress percentage
    # Match: OpenAI image editing with two reference images (POST /v1/images/edits).
    openai_image_model: str = "gpt-image-2"
    openai_image_size: str = "1536x1024"  # landscape room photo
    openai_image_quality: str = "low"  # low | medium | high (cost and time grow with it)
    openai_image_input_fidelity: str = ""  # "high" keeps reference details on models that support it (gpt-image-1.x); gpt-image-2 rejects it
    openai_image_timeout_seconds: float = 300.0
    # Admin "Generate furniture" product photos: low quality (fast and cheap); all of them together stay under
    # product_photos_max_bytes (each is a JPEG of at most its share).
    openai_product_photo_quality: str = "low"
    product_photos_max_bytes: int = 3 * 1024 * 1024
    openai_image_expected_seconds: float = 30.0  # typical time per image, for the progress percentage

    supabase_url: str = ""
    supabase_anon_key: SecretStr = SecretStr("")
    # Session cookies are Secure when the request came over HTTPS (X-Forwarded-Proto); true forces it.
    cookie_secure: bool = False
    supabase_service_role_key: SecretStr = SecretStr("")
    database_url: SecretStr = SecretStr("")

    # Admin console (vendor approval). Change both in production; the session secret defaults to the
    # service role key so tokens can't be forged without it.
    admin_username: str = "admin@admin.com"  # the admin signs in with this email
    admin_password: SecretStr = SecretStr("admin")
    admin_session_secret: SecretStr = SecretStr("")
    admin_session_hours: float = 12.0

    # File storage: Cloudflare R2 (S3 API). Files are served by this API at {api_public_url}/api/files/<key>
    # unless cdn_public_url is set (the bucket's public r2.dev URL or custom domain).
    r2_endpoint: str = ""
    r2_region: str = "auto"
    r2_access_key_id: SecretStr = SecretStr("")
    r2_secret_access_key: SecretStr = SecretStr("")
    r2_bucket: str = "cads"
    api_public_url: str = "http://localhost:8010"
    cdn_public_url: str = ""

    cors_origins: list[str] = []
    # Local development, also over the LAN (the mobile app's web view). Production: set CORS_ORIGINS to the site.
    cors_origin_regex: str | None = r"^http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?$"
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
