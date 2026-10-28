"""Download the Formula 1 World Championship dataset from Kaggle into data/raw."""

import logging
from pathlib import Path

import kagglehub

from f1_sql_agent.config import RAW_DIR

logger = logging.getLogger(__name__)

DATASET_HANDLE = "rohanrao/formula-1-world-championship-1950-2020"
# Pinned Kaggle version (seasons 1950-2024). Changing it changes the data, so the
# expected answers of the evaluation set must be reviewed too.
DATASET_VERSION = 24

EXPECTED_FILES = frozenset(
    {
        "circuits.csv",
        "constructor_results.csv",
        "constructor_standings.csv",
        "constructors.csv",
        "driver_standings.csv",
        "drivers.csv",
        "lap_times.csv",
        "pit_stops.csv",
        "qualifying.csv",
        "races.csv",
        "results.csv",
        "seasons.csv",
        "sprint_results.csv",
        "status.csv",
    }
)


def validate_files(directory: Path) -> None:
    """Fail fast if any expected CSV is missing from the download."""
    found = {p.name for p in directory.glob("*.csv")}
    missing = EXPECTED_FILES - found
    if missing:
        raise FileNotFoundError(f"Missing files in {directory}: {sorted(missing)}")


def download(raw_dir: Path = RAW_DIR, *, force: bool = False) -> Path:
    """Download the pinned dataset version into raw_dir and validate it."""
    handle = f"{DATASET_HANDLE}/versions/{DATASET_VERSION}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s into %s", handle, raw_dir)
    downloaded = kagglehub.dataset_download(
        handle, output_dir=str(raw_dir), force_download=force
    )
    path = Path(downloaded)
    validate_files(path)
    logger.info("Dataset ready: %d files in %s", len(EXPECTED_FILES), path)
    return path


def main() -> None:
    """Entry point for `uv run f1-download`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    download()
