from pathlib import Path

import pytest


def test_materialize_cli_requires_explicit_sources_hashes_and_output(capsys):
    from scripts import materialize_joint_dataset as cli
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err


def test_materialize_cli_only_calls_builder_with_explicit_values(monkeypatch, capsys):
    from scripts import materialize_joint_dataset as cli
    calls = []
    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return {"status": "materialized_not_training_ready", "joint_database": {"sha256": "d" * 64}}
    monkeypatch.setattr(cli, "materialize_joint_dataset", build)
    cli.main(["--sur-db", "sur.sqlite3", "--sur-manifest", "sur.json",
              "--norte-db", "norte.sqlite3", "--norte-manifest", "norte.json",
              "--sur-inp-sha256", "a" * 64, "--norte-inp-sha256", "b" * 64,
              "--norte-labels-sha256", "c" * 64, "--output", "new-output"])
    assert calls[0][1]["expected_network_sha256s"] == ("a" * 64, "b" * 64)
    assert calls[0][1]["expected_b_labels_sha256"] == "c" * 64
    assert str(calls[0][0][-1]) == str(Path("new-output"))
    assert "not_training_ready" in capsys.readouterr().out


def test_prepare_cli_requires_explicit_protocol_and_threshold(capsys):
    from scripts import prepare_joint_dataset as cli
    with pytest.raises(SystemExit) as exc:
        cli.main(["--package", "package", "--output", "output"])
    assert exc.value.code == 2
    assert "required" in capsys.readouterr().err


def test_prepare_cli_only_prepares_and_records_explicit_protocol(monkeypatch, capsys):
    from types import SimpleNamespace
    from scripts import prepare_joint_dataset as cli
    calls = []
    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(prep_id="fixture", quality={"n_rows": 4})
    monkeypatch.setattr(cli, "prepare_joint_dataset", prepare)
    monkeypatch.setattr(cli, "development_run_ids", lambda package, partition: [11, 12])
    monkeypatch.setattr(cli, "verify_partition_hash", lambda partition, digest: None)
    cli.main(["--package", "package", "--output", "output", "--protocol", "LeaveOneShapeOut",
              "--max-continuity-pct", "5", "--partition", "partition.json", "--partition-sha256", "a" * 64])
    assert calls[0][1]["protocols"] == ("LeaveOneShapeOut",)
    assert calls[0][1]["max_continuity_error_pct"] == 5.0
    assert calls[0][1]["run_ids"] == [11, 12]
    assert "fixture" in capsys.readouterr().out


def test_partition_hash_detects_changes(tmp_path):
    import hashlib
    from scripts.prepare_joint_dataset import verify_partition_hash
    file = tmp_path / "partition.json"
    file.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    verify_partition_hash(file, digest)
    file.write_bytes(b"modified")
    with pytest.raises(ValueError, match="hash"):
        verify_partition_hash(file, digest)


def test_partition_membership_rejects_reserved_run_in_development(tmp_path):
    import json
    import sqlite3
    from scripts.prepare_joint_dataset import development_run_ids
    package = tmp_path / "package"
    package.mkdir()
    (package / "status.json").write_text(json.dumps({"manifest_sha256": "a" * 64}))
    db = sqlite3.connect(package / "joint.sqlite3")
    db.executescript("""CREATE TABLE runs(run_id INTEGER, scenario_id INTEGER, network_id INTEGER);
        CREATE TABLE scenarios(scenario_id INTEGER, network_id INTEGER, shape_id TEXT);
        CREATE TABLE run_hydraulic_quality(run_id INTEGER, eligible INTEGER);
        INSERT INTO runs VALUES(1,1,1),(2,2,1);
        INSERT INTO scenarios VALUES(1,1,'training'),(2,1,'reserved');
        INSERT INTO run_hydraulic_quality VALUES(1,1),(2,1);""")
    db.commit()
    db.close()
    partition = tmp_path / "partition.json"
    data = {"status": "membership_only_not_training_ready", "source_manifest_sha256": "a" * 64,
            "reserved_shapes": ["reserved"], "run_ids": {"development": [1], "holdout": [2], "excluded": []}}
    partition.write_text(json.dumps(data))
    assert development_run_ids(package, partition) == [1]
    data["run_ids"]["development"] = [1, 2]
    data["run_ids"]["holdout"] = []
    partition.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="partition"):
        development_run_ids(package, partition)
