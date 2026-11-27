"""Read per-script configuration from the environment or the project .env."""

import os
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ENV = Path(__file__).resolve().parents[1] / ".env"


def env_value(name: str) -> str | None:
    """Use a process override when present, including an intentionally empty value."""
    if name in os.environ:
        return os.environ[name]
    return dotenv_values(PROJECT_ENV, encoding="utf-8-sig", interpolate=False).get(name)
