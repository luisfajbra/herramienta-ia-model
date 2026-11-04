import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path

root = Path('outputs/experiments/joint_chico_sur_norte_20261104_v1')
status = json.loads((root / 'status.json').read_text(encoding='utf-8'))
assert status['status'] == 'materialized_not_training_ready'
manifest = root / 'joint_source_manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == status['manifest_sha256']
with sqlite3.connect((root / 'joint.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
    rows = db.execute('''SELECT s.shape_id, s.network_id, count(*)
        FROM scenarios s JOIN runs r ON r.scenario_id = s.scenario_id AND r.network_id = s.network_id
        JOIN run_hydraulic_quality q ON q.run_id = r.run_id
        WHERE q.eligible = 1 GROUP BY s.shape_id, s.network_id ORDER BY s.shape_id, s.network_id''').fetchall()
    by_shape = {}
    for shape, network, count in rows:
        by_shape.setdefault(shape, {})[network] = count
    population = sorted(shape for shape, counts in by_shape.items() if set(counts) == {1, 2})
    assert len(population) == 35
    chosen = random.Random(42).sample(population, 7)
    result = {'status': 'reservation_selection_only', 'method': 'random.Random(42).sample(sorted_common_shapes, 7)',
              'python': sys.version, 'seed': 42, 'source_manifest_sha256': status['manifest_sha256'],
              'population': population, 'selected_order': chosen, 'selected_sorted': sorted(chosen),
              'eligible_counts': {shape: by_shape[shape] for shape in chosen},
              'scope': 'all factors and both networks; exclude before model tuning and training',
              'limitation': 'does not establish independence of similar shapes or generalization to unseen networks'}
    out = Path('outputs/experiments/reserva_formas_chico_sur_norte_20261104.json')
    with out.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
