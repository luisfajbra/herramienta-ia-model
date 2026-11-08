"""Evalúa candidatos conjuntos en los diez escenarios exploratorios conservados."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scripts import build_chico_norte_presentation_figures as figures
from scripts.evaluate_joint_holdout_20261108 import metric, route_predictions
from scripts.run_chico_norte_novel_hydrograph_eval import _draw_volume_curves
from swmm_resilience.ml.bench.train import load_candidate_pair
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V21, TABULAR_V4_21

RUN = figures.RUN
CANDIDATES = ROOT / 'outputs/experiments/final_joint_candidates_20261108_v1'
OUTPUT = ROOT / 'outputs/experiments/conjuntos_diez_hidrogramas_chico_norte_20261108_v1'
FIGURES = ROOT / 'docs/figures_conjuntos_diez_hidrogramas_chico_norte_20261108_v1'
KEYS = ['run_id', 'node_id']


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_alignment(inputs, truth):
    for frame in (inputs, truth):
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError('Claves de nodo y corrida incompletas o duplicadas')
    result = inputs[KEYS].merge(truth[KEYS], on=KEYS, how='outer', indicator=True, validate='one_to_one')
    if not (result['_merge'] == 'both').all():
        raise ValueError('Cobertura de variables y etiquetas diferente')


def main():
    if OUTPUT.exists() or FIGURES.exists():
        raise FileExistsError('La salida ya existe; se protege el experimento anterior')
    paths = figures._prediction_paths()
    input_frames = []
    provenance = {}
    for family, pred in paths.items():
        seal_path = pred.parent / 'prediction_seal.json'
        seal = json.loads(seal_path.read_text(encoding='utf8'))
        inputs = pred.parent / 'inputs.csv'
        if sha(inputs) != seal['inputs_sha256']:
            raise ValueError('Las variables no coinciden con el sello anterior')
        input_frames.append(pd.read_csv(inputs, dtype={'node_id': str}, float_precision='round_trip'))
        provenance[family] = {'inputs_sha256': sha(inputs), 'prior_prediction_seal_sha256': sha(seal_path)}
    if not input_frames[0].equals(input_frames[1]):
        raise ValueError('Las variables selladas difieren entre las dos familias')
    inputs = input_frames[0]
    if len(inputs) != 5220 or inputs.run_id.nunique() != 10:
        raise ValueError('El lote no contiene los diez escenarios esperados')
    X = TABULAR_V4_21.validate_frame(inputs.loc[:, list(FEATURE_COLUMNS_V21)])
    historical, metadata, manifest = figures.load_verified_frame()
    truth = historical[['run_id', 'node_id', 'inunda', 'vol_inundacion_m3']].copy()
    validate_alignment(inputs, truth)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    inputs.to_parquet(OUTPUT / 'inputs.parquet', index=False)
    truth.to_parquet(OUTPUT / 'truth.parquet', index=False)
    result = {'schema': 'joint_ten_scenarios_exploratory_v1',
              'scope': 'post hoc; Chico Norte participó en el entrenamiento; no es transferencia entre redes',
              'n_scenarios': 10, 'n_node_scenarios': len(inputs), 'prior_sources': provenance,
              'source_manifest': manifest, 'models': {}}
    joined = historical.copy()
    for family, sub, prefix in [('xgboost', 'xgboost', 'xgb'), ('mlp', 'mlp/ensemble', 'mlp')]:
        directory = CANDIDATES / sub
        clf, reg, meta = load_candidate_pair(directory, directory)
        if meta['family'] != family or meta['feature_contract_id'] != 'tabular_v4_21' or len(meta['source_network_ids']) != 2:
            raise ValueError('El candidato no corresponde al entrenamiento conjunto')
        probability = np.asarray(clf.predict_proba(X))[:, 1]
        predicted_class, volume = route_predictions(probability, reg.predict(X))
        predictions = inputs[KEYS].copy()
        predictions['y_prob_clf'], predictions['y_pred_clf'], predictions['volume_routed_m3'] = probability, predicted_class, volume
        predictions.to_parquet(OUTPUT / f'{family}_predictions.parquet', index=False)
        scored = truth.merge(predictions, on=KEYS, validate='one_to_one')
        result['models'][family] = {'overall': metric(scored),
            'classifier_sha256': meta['classifier_sha256'], 'regressor_sha256': meta['regressor_sha256'],
            'metadata_sha256': sha(directory / 'metadata.json'),
            'predictions_sha256': sha(OUTPUT / f'{family}_predictions.parquet')}
        joined = joined.drop(columns=[f'{prefix}_probability', f'{prefix}_class', f'{prefix}_volume_m3'])
        joined = joined.merge(predictions.rename(columns={'y_prob_clf': f'{prefix}_probability',
            'y_pred_clf': f'{prefix}_class', 'volume_routed_m3': f'{prefix}_volume_m3'}), on=KEYS, validate='one_to_one')
    result['inputs_sha256'], result['truth_sha256'] = sha(OUTPUT / 'inputs.parquet'), sha(OUTPUT / 'truth.parquet')
    joined.to_parquet(OUTPUT / 'comparacion.parquet', index=False)
    FIGURES.mkdir(parents=True, exist_ok=False)
    figures.OUTPUT = FIGURES
    for shape in figures.SHAPE_LABELS:
        figures.draw_hydrograph(figures.SHAPE_DIR / f'{shape}.csv', shape)
    for family in ('xgboost', 'mlp'):
        figures.draw_parity(joined, family)
    available = [{'shape_id': item['shape_id'], 'factor_mult': item['factor_mult'],
                  'scenario_key': item['scenario_key']} for item in metadata.values()]
    result['maps'] = []
    for row in figures.expand_map_scenarios(available):
        filename = f"mapa_{row['shape_id']}_f{float(row['factor_mult']):.1f}_{row['model_key']}.png"
        result['maps'].append(figures.draw_network_map(joined, row, figures.INP, FIGURES / filename))
    _draw_volume_curves(FIGURES, joined)
    result['figures_sha256'] = {p.name: sha(p) for p in sorted(FIGURES.glob('*.png'))}
    (OUTPUT / 'resultados.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({k: v['overall'] for k, v in result['models'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
