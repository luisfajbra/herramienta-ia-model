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
    cli.main(["--package", "package", "--output", "output", "--protocol", "LeaveOneShapeOut",
              "--max-continuity-pct", "5"])
    assert calls[0][1]["protocols"] == ("LeaveOneShapeOut",)
    assert calls[0][1]["max_continuity_error_pct"] == 5.0
    assert "fixture" in capsys.readouterr().out
