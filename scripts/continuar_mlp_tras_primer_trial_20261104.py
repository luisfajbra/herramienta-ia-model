"""Continue only after the measured first MLP trial finished successfully."""
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
study = root / 'outputs/experiments/tune_joint_20261104/mlp'
log_path = root / 'outputs/experiments/tune_joint_20261104/mlp_supervisor.log'
with log_path.open('a', encoding='utf-8', buffering=1) as log:
    def say(message):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), message, file=log, flush=True)
    say('Waiting for first trial to finish and release its runner lock.')
    while True:
        try:
            lock = sqlite3.connect(study / 'runner_lock.sqlite3', timeout=0)
            try:
                lock.execute('BEGIN EXCLUSIVE')
                with sqlite3.connect((study / 'study.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
                    trials = db.execute('SELECT number,state FROM trials ORDER BY number').fetchall()
                if trials == [(0, 'COMPLETE')]:
                    say('First trial complete; starting remaining budget.')
                    break
                say(f'Not continuing; unexpected or interrupted study state: {trials!r}')
                sys.exit(1)
            finally:
                lock.close()
        except sqlite3.OperationalError:
            time.sleep(20)
    args = [sys.executable, str(root / 'scripts/tune_joint_checkpointed.py'),
            '--package', str(root / 'outputs/experiments/joint_chico_sur_norte_20261104_v1'),
            '--partition', str(root / 'outputs/experiments/particion_chico_sur_norte_20261104.json'),
            '--partition-sha256', 'b81fd31efa6a0b28b51e66e7bdfb6255d80de37ea25f0cb57714010bb2e892f4',
            '--prepared', str(root / 'outputs/experiments/prepared_chico_sur_norte_development_20261104_v1/65b26490afac36bd'),
            '--output', str(study), '--family', 'mlp']
    result = subprocess.run(args, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=False)
    say(f'Continuation exited with code {result.returncode}.')
    sys.exit(result.returncode)
