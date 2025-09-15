"""
Centralized configuration management for the negotiation simulator.
Loads settings from environment variables and provides defaults.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

class Settings:
    """Application settings loaded from environment variables."""

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key-here")
    JWT_SECRET: str = os.getenv("JWT_SECRET", "your-jwt-secret-key-here")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_EXPIRATION_HOURS: int = int(os.getenv("JWT_EXPIRATION_HOURS", 24))

    # Database Configuration
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./negotiation_simulator.db")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_TTL: int = int(os.getenv("REDIS_TTL", 3600))  # Cache TTL in seconds

    # Negotiation Settings
    MAX_ROUNDS: int = int(os.getenv("MAX_ROUNDS", 100))
    DEFAULT_PROTOCOL: str = os.getenv("DEFAULT_PROTOCOL", "alternating")
    DEFAULT_ACCEPT_THRESHOLD: float = float(os.getenv("DEFAULT_ACCEPT_THRESHOLD", 0.65))
    DEFAULT_CONCESSION_RATE: float = float(os.getenv("DEFAULT_CONCESSION_RATE", 0.08))

    # Monitoring & Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "negotiation.log")
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", 9090))
    ENABLE_MONITORING: bool = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

    # Performance Settings
    WORKER_THREADS: int = int(os.getenv("WORKER_THREADS", 4))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 100))
    SIMULATION_TIMEOUT: int = int(os.getenv("SIMULATION_TIMEOUT", 300))  # seconds
    MAX_CONCURRENT_SIMULATIONS: int = int(os.getenv("MAX_CONCURRENT_SIMULATIONS", 10))

    # File Storage
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./outputs"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "./uploads"))
    MAX_UPLOAD_SIZE: int = int(os.getenv("MAX_UPLOAD_SIZE", 10485760))  # 10MB

    # Security
    ALLOWED_ORIGINS: list = json.loads(os.getenv("ALLOWED_ORIGINS", '["*"]'))
    ENABLE_CORS: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_REQUESTS: int = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
    RATE_LIMIT_PERIOD: int = int(os.getenv("RATE_LIMIT_PERIOD", 60))  # seconds

    # External Services
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    SMTP_HOST: Optional[str] = os.getenv("SMTP_HOST")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER: Optional[str] = os.getenv("SMTP_USER")
    SMTP_PASSWORD: Optional[str] = os.getenv("SMTP_PASSWORD")
    ALERT_EMAIL: Optional[str] = os.getenv("ALERT_EMAIL")

    # Feature Flags
    ENABLE_ADVANCED_STRATEGIES: bool = os.getenv("ENABLE_ADVANCED_STRATEGIES", "true").lower() == "true"
    ENABLE_LLM_ADVISOR: bool = os.getenv("ENABLE_LLM_ADVISOR", "false").lower() == "true"
    ENABLE_WEBSOCKET: bool = os.getenv("ENABLE_WEBSOCKET", "true").lower() == "true"
    ENABLE_BATCH_PROCESSING: bool = os.getenv("ENABLE_BATCH_PROCESSING", "true").lower() == "true"

    # Development/Production
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

    def __init__(self):
        """Initialize settings and create necessary directories."""
        self._create_directories()
        self._validate_settings()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.DATA_DIR, self.OUTPUT_DIR, self.UPLOAD_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def _validate_settings(self):
        """Validate critical settings."""
        if self.ENVIRONMENT == "production":
            if self.JWT_SECRET == "your-jwt-secret-key-here":
                raise ValueError("JWT_SECRET must be set in production")
            if self.API_KEY == "your-secret-api-key-here":
                raise ValueError("API_KEY must be set in production")
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            key: getattr(self, key)
            for key in dir(self)
            if key.isupper() and not key.startswith("_")
        }

    def get_database_settings(self) -> dict:
        """Get database connection settings."""
        if "postgresql" in self.DATABASE_URL:
            return {
                "pool_size": 20,
                "max_overflow": 40,
                "pool_pre_ping": True,
                "pool_recycle": 3600,
            }
        return {}

    def get_redis_settings(self) -> dict:
        """Get Redis connection settings."""
        return {
            "decode_responses": True,
            "max_connections": 50,
            "socket_keepalive": True,
            "socket_keepalive_options": {
                1: 1,  # TCP_KEEPIDLE
                2: 1,  # TCP_KEEPINTVL
                3: 3,  # TCP_KEEPCNT
            }
        }

    def get_logging_config(self) -> dict:
        """Get logging configuration."""
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                },
                'json': {
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                    'format': '%(asctime)s %(name)s %(levelname)s %(message)s'
                }
            },
            'handlers': {
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': self.LOG_FILE,
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5,
                    'formatter': 'json' if self.ENVIRONMENT == 'production' else 'default',
                },
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'default',
                },
            },
            'root': {
                'level': self.LOG_LEVEL,
                'handlers': ['file', 'console'],
            },
        }

# Singleton instance
settings = Settings()

# Convenience functions
def get_settings() -> Settings:
    """Get settings instance."""
    return settings

def is_production() -> bool:
    """Check if running in production."""
    return settings.ENVIRONMENT == "production"

def is_development() -> bool:
    """Check if running in development."""
    return settings.ENVIRONMENT == "development"

def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return settings.DEBUG
