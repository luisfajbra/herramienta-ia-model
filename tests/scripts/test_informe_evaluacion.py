import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from swmm_resilience.ml.bench.reports import write_oof
from tests.ml.bench.prepared_factory import PREP_ID, make_prepared, perfect_oof


def _prepared():
    rows = []
    for shape, scale in (("A", 1.0), ("B", 2.0)):
        for factor in (1.0, 2.0, 3.0):
            rows.append(dict(shape_id=shape, node_id="N0", factor_mult=factor, vol=10.0 * factor * scale,
                             duracion_horas=1.0, tiempo_al_pico_h=0.5))
    return make_prepared(rows, ["LOSO", "LeaveOneShapeOut"])


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

    prepared = _prepared()
    oof = perfect_oof(prepared)
    oof.loc[oof["sample_idx"] == 0, "y_pred_clf"] = 0   # nodo con vol 10 -> FN fuera de la banda 1-2

    tables = build_tables(prepared, {"modelo": oof}, flood_threshold_m3=1.0)

    band = tables["banda_limitrofe"]
    assert {"modelo", "protocolo", "fn_total", "fn_en_banda_1_2m3", "fn_fuera_de_banda"} <= set(band.columns)
    assert int(band.iloc[0]["fn_total"]) == int(band["fn_en_banda_1_2m3"].iloc[0] + band["fn_fuera_de_banda"].iloc[0])


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


def test_main_without_oof_files_fails_instead_of_writing_empty_tables(tmp_path, monkeypatch):
    import pytest

    import scripts.informe_evaluacion as informe

    monkeypatch.setattr(informe, "resolve_prepared", lambda *a, **k: _prepared())

    with pytest.raises(SystemExit):
        informe.main(
            ["--prep-id", PREP_ID, "--prepared-dir", str(tmp_path), "--oof-root", str(tmp_path / "oof"),
             "--out", str(tmp_path / "datos")]
        )
    assert not (tmp_path / "datos").exists()
