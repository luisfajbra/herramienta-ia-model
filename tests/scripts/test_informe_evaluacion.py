import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swmm_resilience.ml.bench.reports import write_oof
from tests.ml.bench.prepared_factory import PREP_ID, make_prepared, perfect_oof


def _with_threshold(prepared, threshold=1.0):
    # Los manifiestos reales traen flood_threshold_m3; la fábrica de pruebas no.
    prepared.manifest["flood_threshold_m3"] = threshold
    return prepared


def _prepared():
    rows = []
    for shape, scale in (("A", 1.0), ("B", 2.0)):
        for factor in (1.0, 2.0, 3.0):
            rows.append(dict(shape_id=shape, node_id="N0", factor_mult=factor, vol=10.0 * factor * scale,
                             duracion_horas=1.0, tiempo_al_pico_h=0.5))
    return _with_threshold(make_prepared(rows, ["LOSO", "LeaveOneShapeOut"]))


def test_build_tables_has_the_four_tables_and_a_confusion_row_per_model_and_protocol():
    from scripts.informe_evaluacion import build_tables

    prepared = _prepared()
    tables = build_tables(prepared, {"modelo": perfect_oof(prepared)}, flood_threshold_m3=1.0)

    assert set(tables) == {"comparacion_protocolos", "matriz_confusion", "banda_limitrofe", "por_forma"}
    confusion = tables["matriz_confusion"]
    assert set(confusion["modelo"]) == {"modelo"}
    assert set(confusion["protocolo"]) == {"LOSO", "LeaveOneShapeOut"}
    row = confusion[confusion["protocolo"] == "LOSO"].iloc[0]
    assert row["tp"] + row["fn"] == int(prepared.y_clf.sum())


def test_boundary_band_splits_false_negatives_by_true_volume():
    from scripts.informe_evaluacion import build_tables

    # Forma A: cinco nodos inundados con volumen real 1, 2, 10, 100 y 150 m3.
    # Forma B: los mismos nodos sin inundar (0.5 m3).
    volumes = {"N1": 1.0, "N2": 2.0, "N3": 10.0, "N4": 100.0, "N5": 150.0}
    rows = []
    for shape in ("A", "B"):
        for factor in (1.0, 2.0):
            for node, vol in volumes.items():
                rows.append(dict(shape_id=shape, node_id=node, factor_mult=factor,
                                 vol=vol if shape == "A" else 0.5,
                                 duracion_horas=1.0, tiempo_al_pico_h=0.5))
    prepared = _with_threshold(make_prepared(rows, ["LOSO", "LeaveOneShapeOut"]))
    oof = perfect_oof(prepared)
    flooded = prepared.y_clf.to_numpy()[oof["sample_idx"].to_numpy()] == 1
    oof.loc[flooded, "y_pred_clf"] = 0          # todos los inundados pasan a falsos negativos
    oof.loc[flooded, "y_pred_reg"] = 0.0        # el volumen predicho no debe intervenir

    band = build_tables(prepared, {"modelo": oof}, flood_threshold_m3=1.0)["banda_limitrofe"]

    assert list(band["protocolo"]) == ["LOSO", "LeaveOneShapeOut"]
    for _, row in band.iterrows():
        # Por protocolo: 2 factores x 5 nodos inundados.
        assert int(row["fn_total"]) == 10
        assert int(row["fn_en_banda_1_2m3"]) == 2        # solo vol 1.0 (2.0 queda fuera: banda [1, 2))
        assert int(row["fn_fuera_de_banda"]) == 8        # 2, 10, 100 y 150
        assert int(row["fn_con_vol_ge_100m3"]) == 4      # 100 y 150 (>= 100)


def test_tables_do_not_depend_on_the_order_models_are_given():
    from scripts.informe_evaluacion import build_tables

    prepared = _prepared()
    oof_b = perfect_oof(prepared)
    oof_b.loc[oof_b["sample_idx"] == 0, "y_pred_clf"] = 0

    first = build_tables(prepared, {"z_modelo": perfect_oof(prepared), "a_modelo": oof_b}, 1.0)
    second = build_tables(prepared, {"a_modelo": oof_b, "z_modelo": perfect_oof(prepared)}, 1.0)

    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert first["matriz_confusion"]["modelo"].iloc[0] == "a_modelo"


def test_build_tables_rejects_a_threshold_different_from_the_manifest():
    from scripts.informe_evaluacion import build_tables

    prepared = _prepared()

    with pytest.raises(ValueError, match="flood_threshold_m3"):
        build_tables(prepared, {"modelo": perfect_oof(prepared)}, flood_threshold_m3=2.0)


def test_build_tables_rejects_a_manifest_without_threshold():
    from scripts.informe_evaluacion import build_tables

    prepared = _prepared()
    del prepared.manifest["flood_threshold_m3"]

    with pytest.raises(ValueError, match="flood_threshold_m3"):
        build_tables(prepared, {"modelo": perfect_oof(prepared)}, flood_threshold_m3=1.0)


def test_write_tables_uses_lf_line_endings(tmp_path):
    from scripts.informe_evaluacion import build_tables, write_tables

    prepared = _prepared()
    paths = write_tables(build_tables(prepared, {"modelo": perfect_oof(prepared)}, 1.0), tmp_path)

    for path in paths.values():
        content = path.read_bytes()
        assert b"\r" not in content
        assert content.endswith(b"\n")


def test_grid_counts_report_runs_full_grid_and_missing_cells():
    from scripts.informe_evaluacion import grid_counts

    rows = [
        dict(shape_id=shape, node_id="N0", factor_mult=factor, vol=5.0, duracion_horas=1.0, tiempo_al_pico_h=0.5)
        for shape, factor in (("A", 1.0), ("A", 2.0), ("B", 1.0))   # falta (B, 2.0)
    ]
    keys = make_prepared(rows, ["LeaveOneShapeOut"]).keys

    assert grid_counts(keys) == {"n_runs": 3, "n_runs_full_grid": 4, "n_missing_cells": 1}


def test_main_reads_the_prepared_dir_and_oof_root_given_and_writes_four_csvs(tmp_path, monkeypatch):
    import scripts.informe_evaluacion as informe

    prepared = _prepared()
    write_oof(perfect_oof(prepared), tmp_path / "oof", PREP_ID, "modelo")
    calls = []

    def fake_resolve(output_dir, prep_id=None, protocols=None):
        calls.append((Path(output_dir), prep_id))
        return prepared

    monkeypatch.setattr(informe, "resolve_prepared", fake_resolve)

    code = informe.main(
        [
            "--prep-id", PREP_ID,
            "--prepared-dir", str(tmp_path / "prepared_v4_21"),
            "--oof-root", str(tmp_path / "oof"),
            "--out", str(tmp_path / "datos"),
        ]
    )

    assert code == 0
    assert calls == [(tmp_path / "prepared_v4_21", PREP_ID)]
    written = sorted(path.name for path in (tmp_path / "datos").glob("*.csv"))
    assert written == [
        "banda_limitrofe.csv", "comparacion_protocolos.csv", "matriz_confusion.csv", "por_forma.csv",
    ]
    comparison = pd.read_csv(tmp_path / "datos" / "comparacion_protocolos.csv")
    assert set(comparison["modelo"]) == {"modelo"}


def test_main_writes_tables_provenance_with_hashes_command_and_git_state(tmp_path, monkeypatch):
    import hashlib
    import json

    import scripts.informe_evaluacion as informe

    prepared = _prepared()
    oof_path = write_oof(perfect_oof(prepared), tmp_path / "oof", PREP_ID, "modelo")
    monkeypatch.setattr(informe, "resolve_prepared", lambda *a, **k: prepared)
    monkeypatch.setattr(informe, "git_state", lambda: {"git_commit": "abc123", "git_dirty": False})

    informe.main(
        ["--prep-id", PREP_ID, "--prepared-dir", str(tmp_path / "prep"), "--oof-root", str(tmp_path / "oof"),
         "--out", str(tmp_path / "datos")]
    )

    record_text = (tmp_path / "datos" / "tables_provenance.json").read_text(encoding="utf-8")
    record = json.loads(record_text)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert record["prep_id"] == PREP_ID
    assert record["git_commit"] == "abc123"
    assert record["git_dirty"] is False
    assert record["flood_threshold_m3"] == 1.0
    assert record["command"].startswith("python scripts/informe_evaluacion.py --prep-id " + PREP_ID)
    assert record["inputs_sha256"] == {f"{PREP_ID}/modelo.parquet": sha(oof_path)}
    assert record["outputs_sha256"] == {
        name: sha(tmp_path / "datos" / name)
        for name in ("banda_limitrofe.csv", "comparacion_protocolos.csv", "matriz_confusion.csv", "por_forma.csv")
    }
    assert {"n_rows", "n_runs", "n_runs_full_grid", "n_missing_cells"} <= set(record)
    # Sin rutas absolutas de la máquina (tmp_path está fuera del repositorio).
    assert str(tmp_path) not in record_text
    assert tmp_path.as_posix() not in record_text


def test_portable_path_is_repo_relative_inside_the_repo_and_anonymous_outside(tmp_path):
    from scripts.informe_evaluacion import REPO_ROOT, portable_path

    assert portable_path(REPO_ROOT / "outputs" / "bench" / "oof") == "outputs/bench/oof"
    assert portable_path(Path("outputs/bench/oof")) == "outputs/bench/oof"
    outside = portable_path(tmp_path / "datos")
    assert tmp_path.as_posix() not in outside
    assert outside.endswith("datos")


def test_main_without_oof_files_fails_instead_of_writing_empty_tables(tmp_path, monkeypatch):
    import scripts.informe_evaluacion as informe

    monkeypatch.setattr(informe, "resolve_prepared", lambda *a, **k: _prepared())

    with pytest.raises(SystemExit):
        informe.main(
            ["--prep-id", PREP_ID, "--prepared-dir", str(tmp_path), "--oof-root", str(tmp_path / "oof"),
             "--out", str(tmp_path / "datos")]
        )
    assert not (tmp_path / "datos").exists()
