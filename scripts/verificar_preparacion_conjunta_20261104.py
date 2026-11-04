from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swmm_resilience.ml.bench.schemas import PreparedDataset

base = Path('outputs/experiments/prepared_chico_sur_norte_development_20261104_v1/65b26490afac36bd')
split = json.loads(Path('outputs/experiments/particion_chico_sur_norte_20261104.json').read_text(encoding='utf-8'))
prepared = PreparedDataset.load(base)
keys = prepared.keys.reset_index(drop=True)
assert set(keys.run_id.unique()) == set(split['run_ids']['development'])
assert not set(keys.shape_id.unique()) & set(split['reserved_shapes'])
summary = []
for fold_id, fold in prepared.folds.groupby('fold_id'):
    train = keys.iloc[fold.loc[fold['split'] == 'train', 'sample_idx'].to_numpy()]
    test = keys.iloc[fold.loc[fold['split'] == 'test', 'sample_idx'].to_numpy()]
    assert set(train.shape_id).isdisjoint(test.shape_id)
    assert set(train.network_id) == set(test.network_id) == {1, 2}
    summary.append({'fold': int(fold_id), 'train_rows': len(train), 'test_rows': len(test),
                    'test_networks': sorted(int(n) for n in test.network_id.unique()),
                    'test_shapes': sorted(str(s) for s in test.shape_id.unique())})
assert len(summary) == 5
result = {'prep_id': prepared.prep_id, 'rows': len(keys), 'runs': int(keys.run_id.nunique()),
          'shapes': int(keys.shape_id.nunique()), 'reserved_shape_overlap': [], 'folds': summary}
out = Path('outputs/experiments/verificacion_preparacion_conjunta_20261104.json')
with out.open('x', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
