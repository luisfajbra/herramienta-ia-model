"""The opt-in frozen evaluation must not fall through to training."""
import sys
from types import ModuleType

import pytest

import main


def test_parser_offers_explicit_frozen_evaluation_and_dry_run_by_default():
    args = main.build_parser().parse_args(["--evaluate-frozen-chico-norte"])
    assert args.evaluate_frozen_chico_norte is True
    assert args.execute_evaluation is False
    assert args.evaluation_output is None


def test_execution_requires_an_output_directory(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-frozen-chico-norte", "--execute-evaluation"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
    assert "--evaluation-output" in capsys.readouterr().err


def test_execution_flag_cannot_be_used_without_this_evaluation(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--execute-evaluation"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
    assert "require --evaluate-frozen-chico-norte" in capsys.readouterr().err


def test_evaluation_cannot_run_with_training_or_other_pipeline_modes(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-frozen-chico-norte", "--only-ml"])
    with pytest.raises(SystemExit) as exc:
        main.main()
    assert exc.value.code == 2
    assert "must be run by itself" in capsys.readouterr().err


def test_dispatch_is_exclusive_and_passes_explicit_output(monkeypatch):
    calls = []
    workflow = ModuleType("swmm_resilience.evaluation.frozen_workflow")
    workflow.run = lambda *, execute, output: calls.append((execute, output))
    monkeypatch.setitem(sys.modules, "swmm_resilience.evaluation.frozen_workflow", workflow)
    monkeypatch.setattr(main, "load_config", lambda *_: pytest.fail("must not enter main pipeline"))
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-frozen-chico-norte",
                                      "--execute-evaluation", "--evaluation-output", "outputs/test-run"])
    main.main()
    assert len(calls) == 1
    assert calls[0][0] is True
    assert str(calls[0][1]).replace("\\", "/") == "outputs/test-run"
