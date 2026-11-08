"""Compara variables de eventos nuevos con rangos de desarrollo por nodo en Chico Norte."""
from pathlib import Path
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.prepare_joint_dataset import development_run_ids, verify_partition_hash
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21

PACKAGE = ROOT / 'outputs/experiments/joint_chico_sur_norte_20261104_v1'
PARTITION = ROOT / 'outputs/experiments/particion_chico_sur_norte_20261104.json'
SHA = 'b81fd31efa6a0b28b51e66e7bdfb6255d80de37ea25f0cb57714010bb2e892f4'
DATA = ROOT / 'outputs/experiments/conjuntos_escenarios_nuevos_chico_norte_20261108/evaluacion_modelos_conjuntos'


def main():
    output = DATA / 'cobertura_por_variable.json'
    if output.exists():
        raise FileExistsError(output)
    verify_partition_hash(PARTITION, SHA)
    ids = development_run_ids(PACKAGE, PARTITION)
    database = PACKAGE / 'joint.sqlite3'
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
        fields = ','.join(FEATURE_COLUMNS_V21)
        params = ','.join('?' for _ in ids)
        development = pd.read_sql_query(f'SELECT node_id,{fields} FROM training_samples_v4_21 '
            f'WHERE network_id=2 AND run_id IN ({params})', conn, params=ids)
    keys = pd.read_parquet(DATA / 'keys.parquet')
    inputs = pd.read_parquet(DATA / 'inputs.parquet')
    if len(keys) != len(inputs):
        raise ValueError('Las claves y variables no coinciden')
    development['node_id'] = development.node_id.astype(str)
    node_ids = keys.node_id.astype(str)
    if not set(node_ids).issubset(set(development.node_id)):
        raise ValueError('Nodos nuevos sin desarrollo de referencia')
    records = []
    for feature in FEATURE_COLUMNS_V21:
        bounds = development.groupby('node_id')[feature].agg(['min', 'max'])
        bounds = bounds.reindex(node_ids.to_numpy()).reset_index(drop=True)
        values = inputs[feature].astype(float).reset_index(drop=True)
        valid = values.notna() & bounds['min'].notna() & bounds['max'].notna()
        below = valid & (values < bounds['min'] - 1e-9)
        above = valid & (values > bounds['max'] + 1e-9)
        records.append({'variable': feature, 'valid_pairs': int(valid.sum()),
                        'below_node_min': int(below.sum()), 'above_node_max': int(above.sum()),
                        'no_comparable_bound': int((~valid).sum())})
    result = {'scope': 'rangos univariados por nodo frente al desarrollo de Chico Norte; no prueba cobertura conjunta',
              'partition_sha256': SHA, 'rows': len(keys), 'variables': records}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
