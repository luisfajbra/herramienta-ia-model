"""Independently inspect CSV, effective INP, RPT and sampled OUT for four runs."""
from pathlib import Path
import hashlib
import json
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import networkx as nx
import numpy as np
import pandas as pd
from swmm_resilience.database.run_source_artifacts import verify_run_source_artifacts
from swmm_resilience.simulation.swmm_api_io import read_out_timeseries, get_node_series
from swmm_resilience.visualization._inp_parser import parse_conduits

DB = ROOT / 'outputs/experiments/joint_chico_sur_norte_20261104_v1/joint.sqlite3'
DATA = ROOT / 'outputs/experiments/evaluacion_reserva_conjunta_20261108_v1'
OUTPUT = ROOT / 'outputs/experiments/cotejo_fuentes_bloque_20261108_corregido.json'
RUNS = (690, 700, 1665, 1675)


def hours(value):
    hh, mm = str(value).split(':')
    return int(hh) + int(mm) / 60.


def inp_parts(path, node):
    section = ''
    series, inflows = [], []
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line.startswith('['):
            section = line.upper()
        elif line and not line.startswith(';'):
            parts = line.split()
            if section == '[TIMESERIES]' and len(parts) >= 3 and parts[0] == node:
                series.append((hours(parts[1]), float(parts[2])))
            elif section == '[INFLOWS]' and len(parts) >= 7 and parts[0] == node and parts[1].upper() == 'FLOW':
                inflows.append(parts[:7])
    return np.asarray(series, dtype=float), inflows


def report_volume(path, node):
    text = path.read_text(encoding='utf-8', errors='replace')
    block = text.split('Node Flooding Summary', 1)[1].split('Outfall Loading Summary', 1)[0]
    if '10^6 ltr' not in block:
        raise ValueError('unexpected report units')
    rows = [line.split() for line in block.splitlines() if line.strip().split()[:1] == [node]]
    if len(rows) > 1:
        raise ValueError('duplicate node in report')
    return float(rows[0][-2]) * 1000. if rows else 0.


def simulation_start(path):
    section = ''
    options = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line.startswith('['):
            section = line.upper()
        elif section == '[OPTIONS]' and line and not line.startswith(';'):
            fields = line.split()
            if len(fields) >= 2 and fields[0] in ('START_DATE', 'START_TIME'):
                options[fields[0]] = fields[1]
    if set(options) != {'START_DATE', 'START_TIME'}:
        raise ValueError('effective INP lacks explicit simulation start')
    return pd.to_datetime(options['START_DATE'] + ' ' + options['START_TIME'], format='%m/%d/%Y %H:%M:%S')


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    seal = json.loads((DATA / 'prediction_seal.json').read_text(encoding='utf-8'))
    scores = json.loads((DATA / 'scores.json').read_text(encoding='utf-8'))
    status = json.loads((DATA / 'status.json').read_text(encoding='utf-8'))
    def sha(path):
        with path.open('rb') as f:
            return hashlib.file_digest(f, 'sha256').hexdigest()
    if (status.get('status') != 'holdout_scored' or sha(DATA / 'scores.json') != status['scores_sha256']
            or sha(DATA / 'prediction_seal.json') != scores['seal_sha256']
            or sha(DATA / 'truth.parquet') != scores['truth_sha256']
            or sha(DATA / 'holdout_inputs.parquet') != seal['inputs_sha256']):
        raise ValueError('sealed evaluation files changed')
    with DB.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != seal['source_database_sha256']:
            raise ValueError('database differs from sealed evaluation')
    features = pd.read_parquet(DATA / 'holdout_inputs.parquet')
    truth = pd.read_parquet(DATA / 'truth.parquet')
    con = sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True)
    rows = []
    for run in RUNS:
        verified = verify_run_source_artifacts(con, run)['artifacts']
        paths = {role: Path(value['resolved_path']) for role, value in verified.items()}
        event = pd.read_csv(paths['event_source'], dtype={'node_id': str})
        event['hours'] = event['time'].map(hours)
        own = event[event.node_id == '100C'].sort_values('hours')
        if own.empty or own.hours.duplicated().any():
            raise ValueError('missing or duplicate input time')
        inp_series, inflows = inp_parts(paths['effective_inp'], '100C')
        if len(inflows) != 1 or inflows[0][2:7] != ['100C', 'FLOW', '1', '1', '0']:
            raise ValueError('unexpected inflow series or multiplier')
        expected = own[['hours', 'value_lps']].to_numpy(dtype=float)
        if len(inp_series) < len(expected) or not np.allclose(inp_series[:len(expected)], expected, rtol=0, atol=1e-6):
            raise ValueError('event CSV and effective INP differ')
        if len(inp_series) > len(expected) and not np.allclose(inp_series[len(expected):, 1], 0., atol=1e-12):
            raise ValueError('nonzero values in added drain-down tail')
        local = float(np.trapezoid(own.value_lps.to_numpy(), own.hours.to_numpy()) * 3.6)
        graph = nx.DiGraph((a, b) for _, a, b in parse_conduits(paths['source_network_inp']))
        ancestors = (nx.ancestors(graph, '100C') if '100C' in graph else set()) | {'100C'}
        contributing = event[event.node_id.isin(ancestors)]
        total = sum(float(np.trapezoid(g.sort_values('hours').value_lps, g.sort_values('hours').hours) * 3.6)
                    for _, g in contributing.groupby('node_id'))
        feat = features[(features.run_id == run) & (features.node_id == '100C')].iloc[0]
        if not np.isclose(local, feat.volumen_local_integrado_m3, rtol=1e-9, atol=1e-6):
            raise ValueError('local integrated volume mismatch')
        if not np.isclose(total, feat.volumen_tributario_integrado_m3, rtol=1e-9, atol=1e-6):
            raise ValueError('tributary integrated volume mismatch')
        rpt_m3 = report_volume(paths['swmm_rpt'], '100C')
        target = truth[(truth.run_id == run) & (truth.node_id == '100C')].iloc[0]
        if not np.isclose(rpt_m3, target.vol_inundacion_m3, atol=1e-9):
            raise ValueError('report volume and sealed target differ')
        out = read_out_timeseries(paths['swmm_out'])
        start = simulation_start(paths['effective_inp'])
        out_df = get_node_series(out, '100C', simulation_start=start)
        timestamps = pd.to_datetime(out_df['index'])
        elapsed = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
        flooding = out_df.flooding_lps.to_numpy(dtype=float)
        sampled = float(np.trapezoid(flooding, elapsed) / 1000.)
        rows.append({'run_id': run, 'network_id': int(feat.network_id), 'factor': float(feat.factor_mult),
            'node_id': '100C', 'event_points': len(expected), 'drain_tail_points': len(inp_series) - len(expected),
            'local_volume_csv_m3': local, 'tributary_volume_csv_m3': total,
            'contributing_input_nodes': int(contributing.node_id.nunique()), 'report_flood_volume_m3': rpt_m3,
            'out_sampled_flood_integral_m3': sampled, 'out_first_timestamp': str(timestamps.iloc[0]),
            'out_last_timestamp': str(timestamps.iloc[-1]), 'out_sample_interval_seconds': float(np.median(np.diff(elapsed))),
            'simulation_start': str(start), 'helper_first_time_sec_value': float(out_df.time_sec.iloc[0]),
            'note': 'OUT integral is sampled, excludes first report interval and is not the RPT routing-step integral.'})
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    con.close()
    OUTPUT.write_text(json.dumps({'results': rows, 'source_database_sha256': seal['source_database_sha256'],
        'scope': 'four runs, one node per network; descriptive not network-wide validation'}, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
