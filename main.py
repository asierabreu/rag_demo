"""
ESA Ground Segment RAG — entry point.

Usage:
    python main.py
    uvicorn main:app --reload          # development
    uvicorn main:app --host 0.0.0.0    # production
"""
from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv

from src.api   import create_app
from src.utils import load_config, setup_logging

# Load .env before anything else so API keys are available at import time
load_dotenv()

config = load_config("config.yaml")
setup_logging(
    log_file=f"logs/app.log",
    level=os.getenv("LOG_LEVEL", "INFO"),
)

# Expose the app at module level so uvicorn can import it
app = create_app(config)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=(os.getenv("APP_ENV", "production") == "development"),
        log_level="info",
    )
