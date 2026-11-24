from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    # After image-to-3D, Meshy Auto Split (by_parts, 10 credits) cuts the visual model into the photo reading's
    # physical components. Its result is used only when the file really has several meshes with the raw model's
    # size, geometry and textures; otherwise the unsplit model is kept and the report says why. Off: no split call.
    meshy_auto_split: bool = True
    # What Meshy textures from (meshy_service.image_to_3d_payload): "prompt" (default) a texture_prompt the vision model
    # writes from each photo (services/texture_prompt.py; one vision call per generation); "image" the meshed photo as
    # texture_image_url; "input" Meshy's default from the input image. The geometry input is unchanged by this.
    meshy_texture_source: Literal["image", "input", "prompt"] = "prompt"
    # Fallback policy for the Meshy cleaning step. True: when Image to Image fails, times out, returns nothing usable or
    # its image fails validation (furniture changed), the ORIGINAL photo is meshed and the model's notes say so.
    # False: the conversion stops and the try is refunded.
    meshy_use_original_on_bad_cleaning: bool = True
    # Vision calls on the generated visual model (grouping its meshes, naming them). Off: one vision call per photo,
    # parts named from their geometry and matched to the spec. Never used for Meshy unless meshy_vision_analysis is on.
    visual_part_ai: bool = False
    # Meshy only: before Image to 3D, clean the photo with Meshy Image to Image (furniture cleaning prompt, background
    # removed). Off: the photo goes straight to Image to 3D. Never uses the configured vision model.
    meshy_clean_image: bool = True
    meshy_image_model: str = "gpt-image-2-5-sunburst"  # Image to Image ai_model (validated: meshy_service.IMAGE_TO_IMAGE_MODELS)
    # "" = the built-in furniture cleaning prompt (generation_prompt.MESHY_CLEAN_PROMPT); any text replaces it.
    meshy_clean_prompt: str = ""
    # Optional extra cleaning instruction per furniture family (bed, wardrobe, sofa, table, chair, door or a catalogue
    # category), e.g. MESHY_CLEAN_HINTS='{"bed": "Keep the upholstered headboard."}'. No vision model writes these.
    meshy_clean_hints: dict[str, str] = {}
    # Transparent background (Meshy remove_background). Off: the prompt asks for a flat neutral-gray backdrop instead.
    # Measured: on light furniture the removal deleted whole doors (RGBA 0,0,0,0), which Image to 3D then textured dark.
    meshy_remove_background: bool = False
    # A cleaned image whose check is inconclusive (it could not be compared with the original): "ask" waits for the
    # user to choose the cleaned image or the original; "use_original" meshes the original; "use_cleaned" meshes the
    # cleaned one and says it was unchecked. A failed check always falls back (meshy_use_original_on_bad_cleaning).
    meshy_inconclusive_cleaning: Literal["ask", "use_original", "use_cleaned"] = "ask"
    meshy_clean_timeout_seconds: float = 600.0  # an Image to Image task still running after this counts as failed
    # Optional advanced analysis for Meshy: read the photo with the configured vision model (listing, estimated size,
    # components, structural model, analysis-based split parts). Off (default): the Meshy path makes no vision call.
    meshy_vision_analysis: bool = False
    # Meshy Image to 3D quality (docs.meshy.ai/en/api/image-to-3d). Checked before submission (meshy_service.check_image_to_3d).
    meshy_geometry_resolution: Literal["standard", "2k", "4k"] = "standard"  # 2k / 4k: the Ultra pass (more credits, time)
    meshy_texture_resolution: Literal["2k", "4k", "8k"] = "2k"
    meshy_texture_model: Literal["", "meshy-7", "meshy-n1", "meshy-6-lite"] = ""  # "" = Meshy's default for the ai_model
    meshy_image_enhancement: bool = False  # true lets Meshy restyle the input before meshing (less faithful)
    meshy_should_remesh: bool = False  # remeshing replaced the generated surface and warped furniture

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
    # This API's public origin, for the file URLs it hands out and saves ({api_public_url}/api/files/<key>). Empty: the
    # origin each request reached (its Host header, services/cdn.REQUEST_ORIGIN). The old default, localhost:8010, was
    # written into every saved URL by a server deployed without this setting. Set it in production to pin the domain.
    api_public_url: str = ""
    cdn_public_url: str = ""

    cors_origins: list[str] = []
    # Local development, also over the LAN (the mobile app's web view). Production: set CORS_ORIGINS to the site.
    cors_origin_regex: str | None = r"^http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?$"
    storage_dir: Path = BACKEND_DIR / "storage"

    max_image_bytes: int = 20 * 1024 * 1024

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
