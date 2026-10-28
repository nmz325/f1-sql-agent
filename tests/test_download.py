"""Unit tests for the dataset download (no network access)."""

from pathlib import Path

import pytest

from f1_sql_agent.ingest import download as dl


def _create_files(directory: Path, names: frozenset[str]) -> None:
    for name in names:
        (directory / name).write_text("id\n1\n", encoding="utf-8")


def test_validate_files_accepts_complete_download(tmp_path: Path) -> None:
    _create_files(tmp_path, dl.EXPECTED_FILES)
    dl.validate_files(tmp_path)  # must not raise


def test_validate_files_reports_missing_file(tmp_path: Path) -> None:
    _create_files(tmp_path, dl.EXPECTED_FILES - {"results.csv"})
    with pytest.raises(FileNotFoundError, match="results.csv"):
        dl.validate_files(tmp_path)


def test_download_requests_pinned_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, str] = {}

    def fake_download(handle: str, *, output_dir: str, force_download: bool) -> str:
        calls["handle"] = handle
        _create_files(Path(output_dir), dl.EXPECTED_FILES)
        return output_dir

    monkeypatch.setattr(dl.kagglehub, "dataset_download", fake_download)
    dl.download(tmp_path)
    assert calls["handle"].endswith(f"/versions/{dl.DATASET_VERSION}")
