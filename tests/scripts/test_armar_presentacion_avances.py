"""Pruebas del armado de la presentacion del 13 de octubre (avances)."""

import csv
import hashlib
import sys
from pathlib import Path

import pytest
from pptx import Presentation
from pptx.util import Emu

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import armar_presentacion_avances_13_10 as deck  # noqa: E402

DATOS_REALES = (
    Path(__file__).resolve().parents[2] / "docs" / "auditoria" / "correcciones" / "datos"
)

CAMPOS = [
    "modelo", "protocolo", "que_mide", "f1_agregado", "recall_agregado",
    "precision_agregada", "f1_media_por_fold_con_ceros", "folds_f1_indefinido",
    "rmse_vol_agregado_m3", "rmse_vol_media_por_fold_m3", "error_vol_total_pct",
    "nse_regresor_oracle",
]


def _fila(modelo, protocolo, f1, rmse, nse):
    return {
        "modelo": modelo, "protocolo": protocolo, "que_mide": "x", "f1_agregado": f1,
        "recall_agregado": 0.9, "precision_agregada": 0.9,
        "f1_media_por_fold_con_ceros": 0.9, "folds_f1_indefinido": 0,
        "rmse_vol_agregado_m3": rmse, "rmse_vol_media_por_fold_m3": rmse,
        "error_vol_total_pct": -1.0, "nse_regresor_oracle": nse,
    }


def _csv_falso(carpeta: Path) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = [
        _fila("xgboost", "LeaveOneShapeOut", 0.9801, 90.5, 0.91),
        _fila("mlp", "LeaveOneShapeOut", 0.9755, 95.25, 0.9),
        _fila("svm", "LeaveOneShapeOut", 0.9, 150.5, 0.5),
        _fila("baseline_forma_mas_cercana", "LeaveOneShapeOut", 0.9764, 172.86, 0.71),
        _fila("xgboost", "LOSO", 0.9667, 57.89, 0.96),  # otro protocolo: se ignora
    ]
    ruta = carpeta / "comparacion_protocolos.csv"
    with open(ruta, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerows(filas)
    return ruta


def _prs_base():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Emu(deck.SLIDE_W), Emu(deck.SLIDE_H)
    prs.slides.add_slide(prs.slide_layouts[6])
    return prs


def _textos(slide):
    out = []
    for shape in slide.shapes:
        if shape.has_text_frame:
            out.append(shape.text_frame.text)
        if shape.has_table:
            out.extend(c.text for row in shape.table.rows for c in row.cells)
    return "\n".join(out)


def _titulo(slide):
    return next(s for s in slide.shapes if s.name == "TextBox 3").text_frame.text


def _pie(slide):
    return next(s for s in slide.shapes if s.name == "TextBox 5").text_frame.text


def _numero(slide):
    return next(s for s in slide.shapes if s.name == "TextBox 6").text_frame.text


def test_fmt_usa_formato_espanol():
    assert deck.fmt(106526, 0) == "106.526"
    assert deck.fmt(9.1263, 2) == "9,13"
    assert deck.fmt(0.98170, 4) == "0,9817"
    assert deck.fmt(172.8642, 2) == "172,86"


def test_agregar_slide_resultados_agrega_dos_slides_con_tabla_y_graficos(tmp_path):
    datos = tmp_path / "datos_parte2"
    _csv_falso(datos)
    prs = _prs_base()
    nuevas = deck.agregar_slide_resultados(prs, datos)
    assert len(nuevas) == 2 and len(prs.slides) == 3

    resultados, conclusiones = nuevas
    assert _titulo(resultados) == "Resultados: XGBoost, MLP y SVM aproximado"
    assert _titulo(conclusiones) == "Conclusiones y próximos pasos"
    for slide in nuevas:  # los titulos no terminan en punto
        assert not _titulo(slide).rstrip().endswith(".")

    tabla = next(s for s in resultados.shapes if s.has_table).table
    celdas = [[c.text for c in row.cells] for row in tabla.rows]
    assert celdas[0] == ["Modelo", "RMSE", "F1", "NSE"]
    assert [fila[0] for fila in celdas[1:]] == [
        "XGBoost", "MLP", "SVM aproximado", "Forma más cercana (ref.)"]
    assert celdas[1][1:] == ["90,50", "0,9801", "0,9100"]
    assert celdas[3][1] == "150,50"

    graficos = [s.chart for s in resultados.shapes if s.has_chart]
    assert len(graficos) == 2
    rmse, f1 = graficos
    assert list(rmse.plots[0].categories) == [
        "XGBoost", "MLP", "SVM aproximado", "Forma más cercana (ref.)"]
    assert list(rmse.plots[0].series[0].values) == [90.5, 95.25, 150.5, 172.86]
    assert list(f1.plots[0].series[0].values) == [0.9801, 0.9755, 0.9, 0.9764]

    texto = _textos(conclusiones)
    assert "90,50 m³" in texto and "XGBoost" in texto  # menor RMSE entre candidatos
    assert "0,9801" in texto  # mayor F1


def test_las_referencias_no_se_presentan_como_ganadoras(tmp_path):
    datos = tmp_path / "datos_parte2"
    _csv_falso(datos)
    ruta = datos / "comparacion_protocolos.csv"
    with open(ruta, "a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=CAMPOS).writerow(
            _fila("baseline_interpolacion_factor", "LeaveOneShapeOut", 0.999, 1.0, 0.99))
    _, conclusiones = deck.agregar_slide_resultados(_prs_base(), datos)
    assert "Interpolación" not in _textos(conclusiones)
    assert "1,00 m³" not in _textos(conclusiones)


def test_los_slides_de_resultados_tienen_notas_y_numeracion_continua(tmp_path):
    datos = tmp_path / "datos_parte2"
    _csv_falso(datos)
    prs = _prs_base()
    resultados, conclusiones = deck.agregar_slide_resultados(prs, datos)
    assert [_numero(resultados), _numero(conclusiones)] == ["2", "3"]
    for slide in (resultados, conclusiones):
        assert "comparacion_protocolos.csv" in slide.notes_slide.notes_text_frame.text
    assert "90,50" in resultados.notes_slide.notes_text_frame.text


def test_agregar_slide_resultados_falla_sin_filas_de_leave_one_shape_out(tmp_path):
    datos = tmp_path / "datos_parte2"
    datos.mkdir()
    with open(datos / "comparacion_protocolos.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerow(_fila("xgboost", "LOSO", 0.9, 50.0, 0.9))
    with pytest.raises(ValueError, match="LeaveOneShapeOut"):
        deck.agregar_slide_resultados(_prs_base(), datos)


def test_construir_sin_datos_parte2_agrega_seis_slides_y_no_toca_la_base(tmp_path):
    base = tmp_path / "base.pptx"
    _prs_base().save(str(base))
    huella = hashlib.sha256(base.read_bytes()).hexdigest()
    salida = tmp_path / "salida.pptx"

    prs = deck.construir(base, salida, DATOS_REALES, tmp_path / "no_existe")
    assert len(prs.slides) == 1 + 6
    assert hashlib.sha256(base.read_bytes()).hexdigest() == huella
    assert len(Presentation(str(salida)).slides) == 7

    # Idempotente: reconstruir da el mismo numero de slides, no acumula.
    prs2 = deck.construir(base, salida, DATOS_REALES, tmp_path / "no_existe")
    assert len(prs2.slides) == 7

    nuevas = list(prs2.slides)[1:]
    assert [_titulo(s) for s in nuevas] == [
        "Auditoría: qué se encontró y qué se corrigió",
        "Qué mide cada protocolo (y por qué importa)",
        "Matriz de confusión de XGBoost con formas no vistas",
        "La etiqueta de inundación y su resolución",
        "Cambio de arquitectura del SVM",
        "Cómo se hará la comparación justa",
    ]
    for pos, slide in enumerate(nuevas, start=2):
        assert not _titulo(slide).rstrip().endswith(".")
        assert _numero(slide) == str(pos)
        assert _pie(slide).startswith(deck.FOOTER_TEXT)
        assert len(slide.notes_slide.notes_text_frame.text) > 300
    assert "resultados en la siguiente diapositiva" not in _pie(nuevas[-1])


def test_construir_con_datos_parte2_agrega_los_slides_de_resultados(tmp_path):
    base = tmp_path / "base.pptx"
    _prs_base().save(str(base))
    datos2 = tmp_path / "datos_parte2"
    _csv_falso(datos2)
    prs = deck.construir(base, tmp_path / "salida.pptx", DATOS_REALES, datos2)
    assert len(prs.slides) == 1 + 6 + 2
    assert _pie(list(prs.slides)[6]).endswith("resultados en la siguiente diapositiva")


def test_construir_se_niega_a_sobrescribir_la_base(tmp_path):
    base = tmp_path / "base.pptx"
    _prs_base().save(str(base))
    with pytest.raises(SystemExit):
        deck.construir(base, base, DATOS_REALES, tmp_path / "x")


def test_las_cifras_de_las_slides_de_auditoria_coinciden_con_los_csv():
    prs = _prs_base()
    cp = {(r["modelo"], r["protocolo"]): r
          for r in deck.cargar_csv(DATOS_REALES / "comparacion_protocolos.csv")}
    mc = {(r["modelo"], r["protocolo"]): r
          for r in deck.cargar_csv(DATOS_REALES / "matriz_confusion.csv")}
    protocolos = deck.slide_protocolos(prs, DATOS_REALES)
    matriz = deck.slide_matriz(prs, DATOS_REALES)

    claves = (
        ("baseline_interpolacion_factor", "LOSO"),
        ("baseline_forma_mas_cercana", "LeaveOneShapeOut"),
    )
    graficos = [s.chart for s in protocolos.shapes if s.has_chart]
    assert len(graficos) == 1
    series = graficos[0].plots[0].series
    assert list(series[0].values) == [
        round(float(cp[k]["rmse_vol_agregado_m3"]), 2) for k in claves]
    assert list(series[1].values) == [
        round(float(cp[("xgboost", p)]["rmse_vol_agregado_m3"]), 2)
        for p in ("LOSO", "LeaveOneShapeOut")]

    texto_m = _textos(matriz)
    m = mc[("xgboost", "LeaveOneShapeOut")]
    for campo in ("tn", "fp", "fn", "tp"):
        assert deck.fmt(m[campo], 0) in texto_m
    assert "333 de 380" in texto_m
    assert deck.fmt(cp[("xgboost", "LeaveOneShapeOut")]["f1_agregado"], 4) in texto_m
