from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "Soccer Prediction API"
    API_V1_STR: str = "/api/v1"
    ODDS_API_KEY: str
    ODDS_API_URL: str = "https://api.the-odds-api.com/v4"
    DATABASE_URL: str = "postgresql://user:password@db:5432/soccer_predictions"
    MIN_CONFIDENCE_THRESHOLD: float = 0.75
    MODEL_UPDATE_INTERVAL: int = 6
    DARK_MODE: bool = False

    # API Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    # Cache settings
    CACHE_EXPIRY: int = 300  # seconds

    model_config = ConfigDict(env_file=".env")

@lru_cache()
def get_settings():
    return Settings()
