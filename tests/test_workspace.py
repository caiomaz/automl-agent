"""Workspace utilities – TDD test suite.

Tests were written BEFORE the implementation.  Run order expected:

  pytest tests/test_workspace.py  → RED (before implementation)
  pytest tests/test_workspace.py  → GREEN (after implementation)
"""

import io
import zipfile
import tarfile
import hashlib
import re
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ── shared helpers ─────────────────────────────────────────────────────────────

def _make_zip_bytes(filename: str = "data.csv", content: bytes = b"a,b\n1,2") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def _make_tar_bytes(filename: str = "data.csv", content: bytes = b"a,b\n1,2") -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name=filename)
        info.size = len(content)
        tf.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def _mock_response(body: bytes, content_disposition: str = ""):
    """Build a mock requests.Response that streams *body*."""
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.headers = {"Content-Disposition": content_disposition}
    mock.iter_content.side_effect = lambda chunk_size=8192: iter(
        [body[i : i + chunk_size] for i in range(0, len(body), chunk_size)] or [b""]
    )
    return mock


def _mock_error_response():
    mock = MagicMock()
    mock.raise_for_status.side_effect = Exception("404 Not Found")
    return mock


# ─────────────────────────────────────────────────────────────────────────────
# TestEnsureWorkspace
# ─────────────────────────────────────────────────────────────────────────────


class TestEnsureWorkspace:
    """ensure_workspace(workspace) must create all standard subdirs."""

    def test_creates_datasets_dir(self, tmp_path):
        from utils.workspace import ensure_workspace
        ws = tmp_path / "agent_workspace"
        ensure_workspace(ws)
        assert (ws / "datasets").is_dir()

    def test_creates_exp_dir(self, tmp_path):
        from utils.workspace import ensure_workspace
        ws = tmp_path / "agent_workspace"
        ensure_workspace(ws)
        assert (ws / "exp").is_dir()

    def test_creates_trained_models_dir(self, tmp_path):
        from utils.workspace import ensure_workspace
        ws = tmp_path / "agent_workspace"
        ensure_workspace(ws)
        assert (ws / "trained_models").is_dir()

    def test_idempotent(self, tmp_path):
        from utils.workspace import ensure_workspace
        ws = tmp_path / "agent_workspace"
        ensure_workspace(ws)
        ensure_workspace(ws)  # second call must not raise
        assert (ws / "datasets").is_dir()

    def test_creates_deep_parent_dirs(self, tmp_path):
        from utils.workspace import ensure_workspace
        ws = tmp_path / "deeply" / "nested" / "agent_workspace"
        ensure_workspace(ws)
        assert (ws / "datasets").is_dir()

    def test_no_arg_uses_project_root(self):
        """Calling without args must not raise even when WORKSPACE_DIR already exists."""
        from utils.workspace import ensure_workspace
        ensure_workspace()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# TestDatasetPathForUrl
# ─────────────────────────────────────────────────────────────────────────────


class TestDatasetPathForUrl:
    """dataset_path_for_url() must return stable, unique, safe paths."""

    def test_same_url_same_path(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        ds = tmp_path / "datasets"
        p1 = dataset_path_for_url("https://example.com/data.zip", "d", datasets_dir=ds)
        p2 = dataset_path_for_url("https://example.com/data.zip", "d", datasets_dir=ds)
        assert p1 == p2

    def test_different_urls_different_paths(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        ds = tmp_path / "datasets"
        p1 = dataset_path_for_url("https://example.com/a.zip", "d", datasets_dir=ds)
        p2 = dataset_path_for_url("https://example.com/b.zip", "d", datasets_dir=ds)
        assert p1 != p2

    def test_path_is_direct_child_of_datasets_dir(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        ds = tmp_path / "datasets"
        p = dataset_path_for_url("https://example.com/data.zip", "name", datasets_dir=ds)
        assert p.parent == ds

    def test_name_appears_in_folder_name(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        ds = tmp_path / "datasets"
        p = dataset_path_for_url("https://example.com/data.zip", "banana_quality", datasets_dir=ds)
        assert "banana_quality" in p.name

    def test_unsafe_chars_sanitized(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        ds = tmp_path / "datasets"
        p = dataset_path_for_url("https://example.com/d.zip", "my data/slash!&weird", datasets_dir=ds)
        assert re.match(r"^[\w\-]+$", p.name), f"Unsafe chars in folder name: {p.name!r}"

    def test_empty_name_gets_default(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        ds = tmp_path / "datasets"
        p = dataset_path_for_url("https://example.com/data.zip", "", datasets_dir=ds)
        assert "dataset" in p.name

    def test_hash_suffix_is_8_hex_chars(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        ds = tmp_path / "datasets"
        p = dataset_path_for_url("https://example.com/data.zip", "mydata", datasets_dir=ds)
        suffix = p.name.rsplit("_", 1)[-1]
        assert len(suffix) == 8
        assert all(c in "0123456789abcdef" for c in suffix)

    def test_hash_matches_sha256(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        url = "https://example.com/data.zip"
        ds = tmp_path / "datasets"
        p = dataset_path_for_url(url, "mydata", datasets_dir=ds)
        expected_hash = hashlib.sha256(url.encode()).hexdigest()[:8]
        assert p.name.endswith(expected_hash)

    def test_uses_custom_datasets_dir(self, tmp_path):
        from utils.workspace import dataset_path_for_url
        custom = tmp_path / "custom_datasets"
        p = dataset_path_for_url("https://example.com/data.zip", "d", datasets_dir=custom)
        assert p.parent == custom


# ─────────────────────────────────────────────────────────────────────────────
# TestRetrieveDownload
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieveDownload:
    """retrieve_download must download files into a standardised workspace path."""

    def test_creates_workspace_dirs_when_missing(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"col1,col2\n1,2"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)):
            retrieve_download(url="https://example.com/data.csv", name="ds", workspace=ws)
        assert (ws / "datasets").is_dir()
        assert (ws / "exp").is_dir()
        assert (ws / "trained_models").is_dir()

    def test_returns_local_directory_path(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"col1,col2\n1,2"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)):
            result = retrieve_download(url="https://example.com/data.csv", name="test", workspace=ws)
        assert result is not None
        assert Path(result).is_dir()

    def test_path_is_inside_workspace_datasets(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"col1,col2\n1,2"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)):
            result = retrieve_download(url="https://example.com/data.csv", name="mydata", workspace=ws)
        assert result.startswith(str(ws / "datasets"))

    def test_downloaded_file_exists_in_dest(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"col1,col2\n1,2"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)):
            result = retrieve_download(url="https://example.com/data.csv", name="test", workspace=ws)
        dest = Path(result)
        assert len(list(dest.iterdir())) >= 1

    def test_uses_content_disposition_filename(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"col1,col2\n1,2"
        resp = _mock_response(body, content_disposition='attachment; filename="my_special_file.csv"')
        with patch("data_agent.retriever.requests.get", return_value=resp):
            result = retrieve_download(url="https://example.com/data.csv", name="test", workspace=ws)
        assert (Path(result) / "my_special_file.csv").exists()

    def test_extracts_zip_archive(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = _make_zip_bytes("inner.csv")
        resp = _mock_response(body, content_disposition='attachment; filename="archive.zip"')
        with patch("data_agent.retriever.requests.get", return_value=resp):
            result = retrieve_download(url="https://example.com/archive.zip", name="test", workspace=ws)
        assert (Path(result) / "inner.csv").exists()

    def test_extracts_tar_gz_archive(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = _make_tar_bytes("inner.csv")
        resp = _mock_response(body, content_disposition='attachment; filename="archive.tar.gz"')
        with patch("data_agent.retriever.requests.get", return_value=resp):
            result = retrieve_download(url="https://example.com/archive.tar.gz", name="test", workspace=ws)
        assert (Path(result) / "inner.csv").exists()

    def test_returns_none_on_http_error(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        with patch("data_agent.retriever.requests.get", return_value=_mock_error_response()):
            result = retrieve_download(url="https://example.com/missing.csv", name="test", workspace=ws)
        assert result is None

    def test_different_urls_get_different_dirs(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"data"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)):
            r1 = retrieve_download(url="https://example.com/a.csv", name="d", workspace=ws)
            r2 = retrieve_download(url="https://example.com/b.csv", name="d", workspace=ws)
        assert r1 != r2

    def test_same_url_is_cached_no_redownload(self, tmp_path):
        """Calling retrieve_download twice with the same URL must only hit the network once."""
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"col1,col2\n1,2"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)) as mock_get:
            retrieve_download(url="https://example.com/data.csv", name="test", workspace=ws)
            retrieve_download(url="https://example.com/data.csv", name="test", workspace=ws)
        assert mock_get.call_count == 1, "Second call should use cache, not re-download"

    def test_same_url_cached_returns_same_path(self, tmp_path):
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"data"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)):
            r1 = retrieve_download(url="https://example.com/data.csv", name="test", workspace=ws)
            r2 = retrieve_download(url="https://example.com/data.csv", name="test", workspace=ws)
        assert r1 == r2

    def test_accepts_extra_kwargs_without_crashing(self, tmp_path):
        """retrieve_download must silently ignore unknown keys like 'task', 'llm_client', etc."""
        from data_agent.retriever import retrieve_download
        ws = tmp_path / "agent_workspace"
        body = b"data"
        with patch("data_agent.retriever.requests.get", return_value=_mock_response(body)):
            result = retrieve_download(
                url="https://example.com/data.csv",
                name="test",
                workspace=ws,
                task="tabular_classification",   # coming from user_requirements pipeline
                llm_client=MagicMock(),
                llm_model="some-model",
            )
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# TestDiscoverDatasets
# ─────────────────────────────────────────────────────────────────────────────


class TestDiscoverDatasets:
    """_discover_datasets() must handle missing workspace gracefully."""

    def test_returns_empty_when_datasets_dir_missing(self, tmp_path):
        from cli import _discover_datasets
        missing = tmp_path / "agent_workspace" / "datasets"
        with patch("utils.workspace.DATASETS_DIR", missing):
            result = _discover_datasets()
        assert result == []

    def test_returns_subdirectories(self, tmp_path):
        from cli import _discover_datasets
        ds = tmp_path / "datasets"
        (ds / "banana").mkdir(parents=True)
        (ds / "iris").mkdir()
        with patch("utils.workspace.DATASETS_DIR", ds):
            result = _discover_datasets()
        names = [Path(p).name for p in result]
        assert "banana" in names
        assert "iris" in names

    def test_returns_files_too(self, tmp_path):
        from cli import _discover_datasets
        ds = tmp_path / "datasets"
        ds.mkdir(parents=True)
        (ds / "data.csv").write_text("a,b\n1,2")
        with patch("utils.workspace.DATASETS_DIR", ds):
            result = _discover_datasets()
        names = [Path(p).name for p in result]
        assert "data.csv" in names

    def test_result_is_sorted(self, tmp_path):
        from cli import _discover_datasets
        ds = tmp_path / "datasets"
        for name in ("zzz", "aaa", "mmm"):
            (ds / name).mkdir(parents=True)
        with patch("utils.workspace.DATASETS_DIR", ds):
            result = _discover_datasets()
        assert result == sorted(result)
