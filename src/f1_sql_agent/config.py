"""Project paths (override the data directory with F1_DATA_DIR)."""

import os
from pathlib import Path

# src/f1_sql_agent/config.py -> parents[2] is the repository root (editable install).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("F1_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
