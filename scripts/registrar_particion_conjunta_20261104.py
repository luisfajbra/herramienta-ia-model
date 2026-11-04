"""Register run membership for the approved shape holdout; does not train."""
import hashlib
import json
import sqlite3
from pathlib import Path

root = Path('outputs/experiments/joint_chico_sur_norte_20261104_v1')
reservation = Path('outputs/experiments/reserva_formas_chico_sur_norte_20261104.json')
target = Path('outputs/experiments/particion_chico_sur_norte_20261104.json')
status = json.loads((root / 'status.json').read_text(encoding='utf-8'))
manifest = json.loads((root / 'joint_source_manifest.json').read_text(encoding='utf-8'))
draw = json.loads(reservation.read_text(encoding='utf-8'))
assert status['status'] == 'materialized_not_training_ready'
assert hashlib.sha256((root / 'joint_source_manifest.json').read_bytes()).hexdigest() == status['manifest_sha256'] == draw['source_manifest_sha256']
database = root / 'joint.sqlite3'
with database.open('rb') as f:
    assert hashlib.file_digest(f, 'sha256').hexdigest() == manifest['joint_database']['sha256']
reserved = set(draw['selected_sorted'])
assert len(reserved) == 7
with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
    rows = db.execute('''SELECT r.run_id, r.network_id, s.shape_id, q.eligible
        FROM runs r JOIN scenarios s ON s.scenario_id=r.scenario_id AND s.network_id=r.network_id
        JOIN run_hydraulic_quality q ON q.run_id=r.run_id ORDER BY r.run_id''').fetchall()
development, holdout, excluded = [], [], []
counts = {'development': {}, 'holdout': {}, 'excluded': {}}
for run, network, shape, eligible in rows:
    group = 'excluded' if not eligible else ('holdout' if shape in reserved else 'development')
    {'development': development, 'holdout': holdout, 'excluded': excluded}[group].append(run)
    key = str(network)
    counts[group][key] = counts[group].get(key, 0) + 1
assert len(holdout) == 321 and len(development) == 1332 and len(excluded) == 322
assert len(set(development) | set(holdout) | set(excluded)) == len(rows) == 1975
result = {'status': 'membership_only_not_training_ready', 'source_manifest_sha256': status['manifest_sha256'],
          'reservation_sha256': hashlib.sha256(reservation.read_bytes()).hexdigest(),
          'reserved_shapes': sorted(reserved), 'run_ids': {'development': development, 'holdout': holdout,
          'excluded': excluded}, 'counts_by_network_id': counts,
          'note': 'All factors and both networks are assigned by shape; excluded QA runs are never used.'}
with target.open('x', encoding='utf-8', newline='\n') as f:
    json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
    f.write('\n')
print(json.dumps({'status': result['status'], 'counts_by_network_id': counts, 'output': str(target)}, indent=2))
