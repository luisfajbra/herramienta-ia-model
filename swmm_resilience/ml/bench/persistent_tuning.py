"""Single-runner Optuna checkpointing; the caller must verify data provenance."""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import platform
import sqlite3

import optuna
from optuna.trial import TrialState

from ...config import ML_RANDOM_STATE
from .tune import SEARCH_SPACES, _objective, build_tuning_folds


@contextmanager
def exclusive_runner(directory: Path):
    """Hold an OS-released SQLite lock in a separate database for the whole study."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = sqlite3.connect(directory / 'runner_lock.sqlite3', timeout=0)
    try:
        try:
            lock.execute('BEGIN EXCLUSIVE')
        except sqlite3.OperationalError as exc:
            raise RuntimeError('another study runner holds the lock') from exc
        yield
    finally:
        lock.close()


def run_persistent_study(prepared, family: str, directory: Path, input_contract: dict, *, budget: int = 50,
                         max_new_trials: int | None = None, seed: int = ML_RANDOM_STATE,
                         fixed_params: dict | None = None, integrity_check=None) -> dict:
    """Run at most ``budget`` total attempts, retaining each Optuna trial.

    An interrupted RUNNING attempt becomes FAIL and counts toward the budget.
    Sampler seeds depend on trial number, so a given recorded history is repeatable;
    an interruption may change that history compared with an uninterrupted run.
    """
    if family not in SEARCH_SPACES or budget < 1 or (max_new_trials is not None and max_new_trials < 0):
        raise ValueError('invalid family or budget')
    root = Path(directory)
    with exclusive_runner(root):
        descriptor = {'schema': 'persistent_shape_tuning_v1', 'prep_id': prepared.prep_id,
                      'input_contract': input_contract, 'family': family, 'budget_attempts': budget,
                      'objective': 'macro_network_rmse', 'n_splits': 5, 'seed': seed,
                      'sampler': 'tpe_seed_plus_trial_number_v1', 'fixed_params': fixed_params or {},
                      'search_space': SEARCH_SPACES[family], 'python': platform.python_version(),
                      'versions': {name: version(name) for name in ('optuna', 'numpy', 'pandas', 'scikit-learn', 'torch', 'xgboost')}}
        package_root = Path(__file__).resolve().parents[2]
        descriptor['implementation_sha256'] = {
            path.relative_to(package_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package_root.rglob('*.py'))
        }
        # JSON normalization also makes tuples in the search space unambiguous.
        encoded = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + '\n'
        contract_path = root / 'study_contract.json'
        database_path = root / 'study.sqlite3'
        if contract_path.exists() != database_path.exists():
            raise ValueError('study contract and database must both exist or both be absent')
        if contract_path.exists():
            if contract_path.read_text(encoding='utf-8') != encoded:
                raise ValueError('study contract differs from the existing study')
        else:
            with contract_path.open('x', encoding='utf-8', newline='\n') as f:
                f.write(encoded)
        study = optuna.create_study(study_name='joint_' + family, direction='minimize',
                                    storage='sqlite:///' + database_path.resolve().as_posix(),
                                    load_if_exists=True)
        for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
            study.tell(trial.number, state=TrialState.FAIL)
        folds = build_tuning_folds(prepared.keys, n_splits=5, seed=seed, protocol='GroupKFold')
        existing = len(study.trials)
        count = min(max(0, budget - existing), max_new_trials if max_new_trials is not None else budget)
        for _ in range(count):
            if integrity_check is not None:
                integrity_check()
            next_number = len(study.trials)
            study.sampler = optuna.samplers.TPESampler(seed=(seed + next_number) % (2**32))
            trial = study.ask()
            try:
                value = _objective(trial, prepared, family, folds, fold_n_jobs=1,
                                   fixed_params=fixed_params, objective='macro_network_rmse')
                if integrity_check is not None:
                    integrity_check()
                if not math.isfinite(float(value)):
                    raise ValueError('trial objective is not finite')
                study.tell(trial, value)
            except Exception:
                study.tell(trial, state=TrialState.FAIL)
                raise
            print(json.dumps({'event': 'trial_completed', 'trial': trial.number,
                              'objective': value}, allow_nan=False), flush=True)
        trials = study.trials
        complete = sum(t.state == TrialState.COMPLETE for t in trials)
        failed = sum(t.state == TrialState.FAIL for t in trials)
        status = ('in_progress' if len(trials) < budget else
                  'budget_finished_with_failures' if failed else 'budget_finished')
        return {'status': status, 'attempts': len(trials), 'complete': complete,
                'failed': failed, 'study': study.study_name}
