import os
from typing import Dict, Any

import toml

from src.utils.console import log_step

PROJECT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


@log_step
def load_config() -> Dict[str, Any]:
    filepath = os.path.join(PROJECT_DIR, "config.toml")
    with open(filepath, "r") as f:
        return toml.load(f)
