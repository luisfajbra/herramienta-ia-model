# tests/test_respaldar_artefactos_sellados.py
import json

import pytest


def sources(tmp_path):
    folder = tmp_path / "sealed_eval"
    (folder / "nested").mkdir(parents=True)
    (folder / "scores.json").write_text('{"a": 1}', encoding="utf-8")
    (folder / "nested" / "truth.parquet").write_bytes(b"\x00\x01binary")
    single = tmp_path / "partition.json"
    single.write_text('{"p": 2}', encoding="utf-8")
    return [folder, single]


def test_build_and_verify_archive_round_trip(tmp_path):
    from scripts.respaldar_artefactos_sellados import build_archive, verify_archive
    archive = tmp_path / "out" / "backup.zip"
    archive.parent.mkdir()
    manifest = build_archive(sources(tmp_path), archive)
    assert sorted(entry["path"] for entry in manifest["files"]) == [
        "partition.json", "sealed_eval/nested/truth.parquet", "sealed_eval/scores.json"]
    saved = json.loads((tmp_path / "out" / "backup.manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    verify_archive(archive, manifest)


def test_verify_detects_corrupted_archive_and_member_mismatch(tmp_path):
    from scripts.respaldar_artefactos_sellados import build_archive, verify_archive
    archive = tmp_path / "backup.zip"
    manifest = build_archive(sources(tmp_path), archive)
    altered = json.loads(json.dumps(manifest))
    altered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="member hash"):
        verify_archive(archive, altered)
    with archive.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError, match="archive hash"):
        verify_archive(archive, manifest)


def test_build_refuses_existing_output_wal_and_duplicate_names(tmp_path):
    from scripts.respaldar_artefactos_sellados import build_archive
    paths = sources(tmp_path)
    archive = tmp_path / "backup.zip"
    archive.write_bytes(b"")
    with pytest.raises(FileExistsError):
        build_archive(paths, archive)
    (paths[0] / "joint.sqlite3-wal").write_bytes(b"pending")
    with pytest.raises(ValueError, match="WAL"):
        build_archive(paths, tmp_path / "other.zip")
    (paths[0] / "joint.sqlite3-wal").unlink()
    with pytest.raises(ValueError, match="distinct"):
        build_archive([paths[0], paths[0]], tmp_path / "third.zip")
