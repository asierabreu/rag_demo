"""
Shared utility functions: config loading, logging setup, formatting helpers.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(log_file: str = "logs/app.log", level: str = "INFO") -> None:
    """Configure loguru for console + rotating file output."""
    Path("logs").mkdir(exist_ok=True)
    logger.remove()
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — {message}",
        colorize=True,
    )
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} — {message}",
    )


def format_sources(chunks: list[dict]) -> str:
    """Deduplicate and format source citations from retrieved chunks."""
    seen: set[str] = set()
    lines: list[str] = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        doc  = meta.get("document_name", "Unknown")
        page = meta.get("page", meta.get("row", "—"))
        key  = f"{doc}:{page}"
        if key not in seen:
            seen.add(key)
            mission = meta.get("mission_name", "Unknown")
            lines.append(f"• [{doc}]  mission={mission}  page/row={page}")
    return "\n".join(lines) if lines else "No sources"


def sanitise_filename(name: str) -> str:
    """Strip characters unsafe for filesystem paths."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
