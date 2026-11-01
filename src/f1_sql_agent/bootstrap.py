"""Build the database on a fresh machine: download the data, then run dbt.

The database and dbt's manifest are build outputs, not part of the
repository. Where they are missing (a new clone, or a cloud host that starts
from the repository) this rebuilds them, the same as running
`uv run f1-download` and `uv run dbt build` by hand. It takes about a minute.
"""

import logging
import os
import subprocess
import sys

from f1_sql_agent.config import DATABASE_PATH, MANIFEST_PATH, PROJECT_ROOT, RAW_DIR
from f1_sql_agent.ingest.download import download

TRANSFORM_DIR = PROJECT_ROOT / "transform"
# dbt has no `python -m dbt`: call its command line function instead, with the
# Python running us, so it works without dbt on the PATH.
DBT = [sys.executable, "-c", "from dbt.cli.main import cli; cli()"]

log = logging.getLogger(__name__)


def is_built() -> bool:
    """True if the database and the manifest the app needs both exist."""
    return DATABASE_PATH.is_file() and MANIFEST_PATH.is_file()


def ensure_database() -> None:
    """Download the raw files and run `dbt build` unless already built."""
    if is_built():
        return
    log.info("Database not found: building it")
    download(RAW_DIR)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [*DBT, "build", "--profiles-dir", "."],
        cwd=TRANSFORM_DIR,  # the profile and the sources use paths relative to it
        env={**os.environ, "F1_DUCKDB_PATH": str(DATABASE_PATH)},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not is_built():
        raise RuntimeError(f"dbt build failed:\n{result.stdout[-3000:]}")
    log.info("Database ready at %s", DATABASE_PATH)
