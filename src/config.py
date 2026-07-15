"""
Configuration management for the invoice extraction system.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load .env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini API
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # Tesseract OCR
    tesseract_cmd: str | None = None

    # Poppler (for pdf2image on Windows)
    poppler_path: str | None = None

    # Processing
    max_file_size_mb: int = 10
    ocr_confidence_threshold: int = 30  # Minimum Tesseract confidence (0-100)
    grounding_match_threshold: float = 0.70  # Minimum fuzzy match score

    # Confidence weights (must sum to 1.0)
    weight_ocr: float = 0.30
    weight_grounding: float = 0.35
    weight_validation: float = 0.35

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    model_config = {
        "env_prefix": "",
        "case_sensitive": False,
        "env_file": ".env",
    }


# Singleton instance
settings = Settings()

# Configure Tesseract path if provided
if settings.tesseract_cmd:
    os.environ["TESSERACT_CMD"] = settings.tesseract_cmd
