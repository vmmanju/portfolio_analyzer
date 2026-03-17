"""Application configuration with environment variable loading."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration loaded from environment variables."""

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/aistocks"
    ENVIRONMENT: str = "local"

    # ── Error-model correction ────────────────────────────────────────────────
    # When True, apply the calibrated Bayesian error coefficients to correct
    # predicted returns before portfolio optimisation and ranking.
    # Set to False in .env (USE_ERROR_CORRECTION=false) to disable globally.
    USE_ERROR_CORRECTION: bool = True

    # Default blend weight passed to update_error_model_if_due() when
    # triggering a scheduled recalibration from within the correction path.
    ERROR_CORRECTION_BLEND_WEIGHT: float = 0.6

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
