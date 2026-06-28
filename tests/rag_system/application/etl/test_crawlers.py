from pathlib import Path
import pytest
from rag_system.application.etl import crawl_document_paths


def test_crawl_returns_supported_files(tmp_path):
    (tmp_path / "note.md").write_text("hello")
    result = crawl_document_paths(tmp_path)
    assert Path(tmp_path / "note.md") in result


def test_crawl_skips_unsupported_extensions(tmp_path):
    (tmp_path / "note.md").write_text("hello")
    (tmp_path / "image.png").write_bytes(b"")
    result = crawl_document_paths(tmp_path)
    assert all(p.suffix == ".md" for p in result)


def test_crawl_skips_dir_when_stem_matches_file(tmp_path):
    (tmp_path / "notes.md").write_text("hello")
    subdir = tmp_path / "notes"
    subdir.mkdir()
    (subdir / "child.md").write_text("child")
    result = crawl_document_paths(tmp_path)
    names = [p.name for p in result]
    assert "notes.md" in names
    assert "child.md" not in names


def test_crawl_recurses_into_unmatched_dirs(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (subdir / "nested.md").write_text("nested")
    result = crawl_document_paths(tmp_path)
    assert subdir / "nested.md" in result


def test_crawl_raises_when_directory_does_not_exist(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        crawl_document_paths(tmp_path / "nonexistent")


def test_crawl_raises_when_path_is_not_a_directory(tmp_path):
    file = tmp_path / "note.md"
    file.write_text("hello")
    with pytest.raises(ValueError, match="not a directory"):
        crawl_document_paths(file)


def test_crawl_returns_empty_for_empty_directory(tmp_path):
    result = crawl_document_paths(tmp_path)
    assert result == []
