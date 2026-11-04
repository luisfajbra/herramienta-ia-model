import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest

from swmm_resilience.evaluation import event_selection
import main


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def plan(monkeypatch):
    root = Path("outputs") / ("test_event_selection_" + uuid4().hex)
    root.mkdir(parents=True)
    network = root / "network.inp"
    network.write_text("known network", encoding="utf-8")
    shape = root / "shape.csv"
    shape.write_text("time_h,q_norm\n0,0\n0.5,1\n1,0\n", encoding="utf-8")
    monkeypatch.setattr(event_selection, "APPROVED_NETWORK_SHA256", sha(network))
    data = {"schema_version": "chico_norte_event_selection_v1", "network_inp": {"path": "network.inp", "sha256": sha(network)},
            "hydrographs": [{"id": "pulso", "path": "shape.csv", "sha256": sha(shape)}],
            "factors": [0.7, 2.7], "input_rule": "nodal_peaks_from_inp"}
    path = root / "plan.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    yield path, data, network, shape
    assert root.resolve().is_relative_to((Path.cwd() / "outputs").resolve())
    shutil.rmtree(root)


def save(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_valid_plan_resolves_relative_files_without_writes(plan):
    path, _, network, shape = plan
    before = {p.name: sha(p) for p in path.parent.iterdir()}
    result = event_selection.inspect_plan(path)
    assert result["scenario_count"] == 2
    assert result["network_inp"]["sha256"] == sha(network)
    assert result["hydrographs"][0]["sha256"] == sha(shape)
    assert {p.name: sha(p) for p in path.parent.iterdir()} == before


def test_rejects_changed_network_and_changed_hydrograph(plan):
    path, _, network, shape = plan
    network.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="network"):
        event_selection.inspect_plan(path)
    network.write_text("known network", encoding="utf-8")
    shape.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hydrograph"):
        event_selection.inspect_plan(path)


@pytest.mark.parametrize("factors", [[0.1], [5.1], [float('nan')], [True], [1.0001], [1.0, 1.0], []])
def test_invalid_factors_are_rejected(plan, factors):
    path, data, _, _ = plan
    data["factors"] = factors
    save(path, data)
    with pytest.raises(ValueError, match="factor"):
        event_selection.inspect_plan(path)


def test_invalid_shape_and_unexpected_field_are_rejected(plan):
    path, data, _, shape = plan
    shape.write_text("time_h,q_norm\n0,0\n0.5,1\n1,0.1\n", encoding="utf-8")
    data["hydrographs"][0]["sha256"] = sha(shape)
    save(path, data)
    with pytest.raises(ValueError, match="zero"):
        event_selection.inspect_plan(path)
    data["unexpected"] = 1
    save(path, data)
    with pytest.raises(ValueError, match="fields"):
        event_selection.inspect_plan(path)


def test_rejects_hydrograph_row_with_extra_column(plan):
    path, data, _, shape = plan
    shape.write_text("time_h,q_norm\n0,0,99\n0.5,1\n1,0\n", encoding="utf-8")
    data["hydrographs"][0]["sha256"] = sha(shape)
    save(path, data)
    with pytest.raises(ValueError, match="columns"):
        event_selection.inspect_plan(path)


def test_rejects_duplicate_json_keys(plan):
    path, _, _, _ = plan
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw[:-1] + ', "input_rule": "nodal_peaks_from_inp"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        event_selection.inspect_plan(path)


def test_rejects_duplicate_and_unsafe_hydrograph_ids(plan):
    path, data, _, _ = plan
    data["hydrographs"].append(dict(data["hydrographs"][0]))
    save(path, data)
    with pytest.raises(ValueError, match="unique"):
        event_selection.inspect_plan(path)
    data["hydrographs"] = [dict(data["hydrographs"][0], id="../escape")]
    save(path, data)
    with pytest.raises(ValueError, match="id"):
        event_selection.inspect_plan(path)


def test_cli_inspection_does_not_start_pipeline(plan, monkeypatch, capsys):
    path, _, _, _ = plan
    monkeypatch.setattr(main, "load_config", lambda *_: pytest.fail("must not start pipeline"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--inspect-event-plan", str(path)])
    main.main()
    assert json.loads(capsys.readouterr().out)["scenario_count"] == 2


def test_cli_inspection_rejects_other_modes(plan, monkeypatch):
    path, _, _, _ = plan
    monkeypatch.setattr(sys, "argv", ["main.py", "--inspect-event-plan", str(path), "--only-ml"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2


def test_prepare_materializes_selected_events_without_simulating(plan, monkeypatch):
    path, data, _, _ = plan
    from scripts import run_chico_norte_novel_hydrograph_eval as backend
    data["network_inp"] = {"path": str(backend.NETWORK_INP), "sha256": sha(backend.NETWORK_INP)}
    monkeypatch.setattr(event_selection, "APPROVED_NETWORK_SHA256", sha(backend.NETWORK_INP))
    save(path, data)
    output = path.parent / "prepared"
    result = event_selection.prepare_plan(path, output)
    assert result["status"] == "prepared_not_simulated"
    assert len(result["scenarios"]) == 2
    for scenario in result["scenarios"]:
        event_path = output / scenario["event_csv"]
        assert event_path.is_file()
        assert sha(event_path) == scenario["event_csv_sha256"]
    assert not list(output.rglob("*.rpt"))
    assert not list(output.rglob("*.out"))
    assert not list(output.rglob("*.sqlite3"))
    with pytest.raises(FileExistsError):
        event_selection.prepare_plan(path, output)


def test_prepare_rejects_shape_changed_during_generation(plan, monkeypatch):
    path, data, _, shape = plan
    from scripts import run_chico_norte_novel_hydrograph_eval as backend
    data["network_inp"] = {"path": str(backend.NETWORK_INP), "sha256": sha(backend.NETWORK_INP)}
    monkeypatch.setattr(event_selection, "APPROVED_NETWORK_SHA256", sha(backend.NETWORK_INP))
    save(path, data)
    original = backend._build_event_inputs
    def tamper_after_build(*args):
        result = original(*args)
        shape.write_text("changed", encoding="utf-8")
        return result
    monkeypatch.setattr(backend, "_build_event_inputs", tamper_after_build)
    output = path.parent / "partial"
    with pytest.raises(ValueError, match="hydrograph"):
        event_selection.prepare_plan(path, output)
    assert not (output / "prepared_events.json").exists()


def test_prepare_cli_requires_output(plan, monkeypatch, capsys):
    path, _, _, _ = plan
    monkeypatch.setattr(sys, "argv", ["main.py", "--prepare-event-plan", str(path)])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
    assert "--event-plan-output" in capsys.readouterr().err


def test_prepare_cli_dispatches_without_main_pipeline(plan, monkeypatch, capsys):
    path, _, _, _ = plan
    calls = []
    monkeypatch.setattr(event_selection, "prepare_plan", lambda p, out: calls.append((p, out)) or {"status": "prepared_not_simulated", "scenarios": []})
    monkeypatch.setattr(main, "load_config", lambda *_: pytest.fail("must not start main pipeline"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--prepare-event-plan", str(path), "--event-plan-output", "outputs/new-events"])
    main.main()
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "prepared_not_simulated"


def test_freeze_creates_verifiable_protocol_before_any_simulation(plan, monkeypatch):
    from scripts import run_chico_norte_novel_hydrograph_eval as backend
    from swmm_resilience.database.b_transfer_protocol import verify_b_transfer_protocol_anchor
    path, data, _, _ = plan
    data["network_inp"] = {"path": str(backend.NETWORK_INP), "sha256": sha(backend.NETWORK_INP)}
    data["factors"] = [0.7]
    monkeypatch.setattr(event_selection, "APPROVED_NETWORK_SHA256", sha(backend.NETWORK_INP))
    save(path, data)
    output = path.parent / "frozen"
    result = event_selection.freeze_plan(path, output)
    protocol = output / "protocol.json"
    receipt = json.loads((output / "anchor_receipt.json").read_text(encoding="utf-8"))
    verify_b_transfer_protocol_anchor(output.parent / "frozen_protocol_anchor.sqlite3", protocol, receipt)
    assert result["status"] == "frozen_before_simulation"
    assert len(json.loads(protocol.read_text(encoding="utf-8"))["scenarios"]) == 1
    assert not list(output.rglob("*.rpt")) and not list(output.rglob("*.out"))


def test_anchor_verification_closes_sqlite_connection(plan, monkeypatch):
    from scripts import run_chico_norte_novel_hydrograph_eval as backend
    from swmm_resilience.database import b_transfer_protocol as anchor_module
    path, data, _, _ = plan
    data["network_inp"] = {"path": str(backend.NETWORK_INP), "sha256": sha(backend.NETWORK_INP)}
    data["factors"] = [0.7]
    monkeypatch.setattr(event_selection, "APPROVED_NETWORK_SHA256", sha(backend.NETWORK_INP))
    save(path, data)
    output = path.parent / "connection-check"
    event_selection.freeze_plan(path, output)
    receipt = json.loads((output / "anchor_receipt.json").read_text(encoding="utf-8"))
    real_connect = anchor_module.sqlite3.connect
    opened = []
    class ConnectionProxy:
        def __init__(self, connection):
            self.connection = connection
            self.closed = False
        def __getattr__(self, name):
            return getattr(self.connection, name)
        def __enter__(self):
            self.connection.__enter__()
            return self
        def __exit__(self, *args):
            return self.connection.__exit__(*args)
        def close(self):
            self.connection.close()
            self.closed = True
    def connect(*args, **kwargs):
        proxy = ConnectionProxy(real_connect(*args, **kwargs))
        opened.append(proxy)
        return proxy
    monkeypatch.setattr(anchor_module.sqlite3, "connect", connect)
    anchor_module.verify_b_transfer_protocol_anchor(output.parent / "connection-check_protocol_anchor.sqlite3", output / "protocol.json", receipt)
    assert opened and all(connection.closed for connection in opened)


def test_freeze_cli_requires_new_output_and_only_freezes(plan, monkeypatch, capsys):
    path, _, _, _ = plan
    calls = []
    monkeypatch.setattr(event_selection, "freeze_plan", lambda p, out: calls.append((p, out)) or {"status": "frozen_before_simulation"})
    monkeypatch.setattr(main, "load_config", lambda *_: pytest.fail("must not start main pipeline"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--freeze-event-plan", str(path), "--event-plan-output", "outputs/new-frozen"])
    main.main()
    assert len(calls) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "frozen_before_simulation"


def test_running_frozen_plan_rejects_modified_event_before_swmm(plan, monkeypatch):
    from scripts import run_chico_norte_novel_hydrograph_eval as backend
    from swmm_resilience.simulation import frozen_pair_runner
    path, data, _, _ = plan
    data["network_inp"] = {"path": str(backend.NETWORK_INP), "sha256": sha(backend.NETWORK_INP)}
    data["factors"] = [0.7]
    monkeypatch.setattr(event_selection, "APPROVED_NETWORK_SHA256", sha(backend.NETWORK_INP))
    save(path, data)
    output = path.parent / "frozen-run"
    event_selection.freeze_plan(path, output)
    manifest = json.loads((output / "prepared_events.json").read_text(encoding="utf-8"))
    event = output / manifest["scenarios"][0]["event_csv"]
    event.write_text("altered", encoding="utf-8")
    monkeypatch.setattr(frozen_pair_runner, "run_frozen_experiment", lambda *a, **k: pytest.fail("SWMM must not run"))
    with pytest.raises(ValueError, match="event"):
        event_selection.run_frozen_plan(output)


def test_running_frozen_plan_refuses_an_existing_progress_file(plan, monkeypatch):
    output = plan[0].parent / "already_run"
    output.mkdir()
    (output / "batch_progress.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        event_selection.run_frozen_plan(output)


def test_run_frozen_cli_dispatches_alone(plan, monkeypatch, capsys):
    path, _, _, _ = plan
    calls = []
    monkeypatch.setattr(event_selection, "run_frozen_plan", lambda p: calls.append(p) or {"status": "complete", "completed": 1})
    monkeypatch.setattr(main, "load_config", lambda *_: pytest.fail("must not enter main pipeline"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--run-frozen-event-plan", str(path.parent)])
    main.main()
    assert calls == [path.parent]
    assert json.loads(capsys.readouterr().out)["completed"] == 1


def test_scoring_requires_completed_swmm_and_never_overwrites(plan):
    output = plan[0].parent / "scoring"
    output.mkdir()
    with pytest.raises(FileNotFoundError):
        event_selection.score_frozen_plan(output)
    assert not (output / "selected_staging.sqlite3").exists()
    (output / "frozen_b_manifest.json").write_text("{}", encoding="utf-8")
    (output / "selected_staging.sqlite3").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        event_selection.score_frozen_plan(output)
    assert (output / "selected_staging.sqlite3").read_text(encoding="utf-8") == "keep"


def test_score_cli_only_dispatches_scoring(plan, monkeypatch, capsys):
    path, _, _, _ = plan
    calls = []
    monkeypatch.setattr(event_selection, "score_frozen_plan", lambda p: calls.append(p) or {"status": "complete"})
    monkeypatch.setattr(main, "load_config", lambda *_: pytest.fail("must not start main pipeline"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--score-frozen-event-plan", str(path.parent)])
    main.main()
    assert calls == [path.parent]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"


def test_figures_require_completed_scores_and_no_existing_directory(plan):
    output = plan[0].parent / "figures-run"
    output.mkdir()
    with pytest.raises(FileNotFoundError):
        event_selection.render_selected_figures(output)
    (output / "selected_scoring_status.json").write_text('{"status":"complete"}', encoding="utf-8")
    (output / "selected_figures").mkdir()
    with pytest.raises(FileExistsError):
        event_selection.render_selected_figures(output)


def test_figures_cli_only_dispatches_figures(plan, monkeypatch, capsys):
    path, _, _, _ = plan
    calls = []
    monkeypatch.setattr(event_selection, "render_selected_figures", lambda p: calls.append(p) or {"status": "complete"})
    monkeypatch.setattr(main, "load_config", lambda *_: pytest.fail("must not start main pipeline"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--figures-frozen-event-plan", str(path.parent)])
    main.main()
    assert calls == [path.parent]
    assert json.loads(capsys.readouterr().out)["status"] == "complete"
