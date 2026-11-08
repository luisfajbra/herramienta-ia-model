# scripts/respaldar_artefactos_sellados.py
"""Archive sealed artifacts with per-file SHA-256 and verify the archive; never modifies sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def _sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return _sha256_stream(handle)


def _members(source: Path) -> list[tuple[Path, str]]:
    if source.is_file():
        members = [(source, source.name)]
    else:
        members = [(path, f"{source.name}/{path.relative_to(source).as_posix()}")
                   for path in sorted(source.rglob("*")) if path.is_file()]
    for path, _ in members:
        if path.name.endswith("-wal") and path.stat().st_size:
            raise ValueError(f"nonempty SQLite WAL cannot be archived by hash: {path}")
    return members


def _manifest_path(archive: Path) -> Path:
    return archive.with_suffix(".manifest.json")


def build_archive(sources: list[Path], archive: Path) -> dict:
    archive = Path(archive)
    if archive.exists() or _manifest_path(archive).exists():
        raise FileExistsError(archive)
    resolved = [Path(source).resolve(strict=True) for source in sources]
    if len({source.name for source in resolved}) != len(resolved):
        raise ValueError("source names must be distinct inside the archive")
    members = [member for source in resolved for member in _members(source)]
    entries = []
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for path, name in members:
            before = _sha256(path)
            bundle.write(path, name)
            if _sha256(path) != before:
                raise ValueError(f"source changed while archiving: {path}")
            entries.append({"path": name, "sha256": before, "bytes": path.stat().st_size})
    manifest = {"schema": "sealed_artifact_backup_v1", "archive": archive.name,
                "archive_sha256": _sha256(archive), "files": entries}
    with _manifest_path(archive).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest


def verify_archive(archive: Path, manifest: dict) -> None:
    archive = Path(archive)
    if _sha256(archive) != manifest["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(entry["path"] for entry in manifest["files"]):
            raise ValueError("archive member list mismatch")
        for entry in manifest["files"]:
            with bundle.open(entry["path"]) as member:
                if _sha256_stream(member) != entry["sha256"]:
                    raise ValueError(f"member hash mismatch: {entry['path']}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_archive(args.source, args.archive)
    verify_archive(args.archive, manifest)
    print(json.dumps({"archive": str(args.archive), "archive_sha256": manifest["archive_sha256"],
                      "files": len(manifest["files"])}, indent=2))


if __name__ == "__main__":
    main()
