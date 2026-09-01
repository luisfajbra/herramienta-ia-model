"""
Pipeline principal: simulación SWMM → dataset → XGBoost → evaluación → mapas.

Uso:
  python main.py                                      # Pipeline completo
  python main.py --skip-extraction                    # Ya tienes dataset_final.csv
  python main.py --only-ml                            # Solo entrenar y evaluar
  python main.py --only-maps                          # Solo generar mapas desde CSV
  python main.py --predict --factor 3.5              # Inferencia sin SWMM
"""
import argparse
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.dataset.assembler import assemble_dataset
from swmm_resilience.dataset.validator import validate_dataset
from swmm_resilience.extraction.dynamic_features import compute_dynamic_features
from swmm_resilience.extraction.labels import extract_labels
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import compute_topology_features
from swmm_resilience.ml.evaluator import evaluate_models
from swmm_resilience.ml.feature_importance import generate_feature_importance_plots
from swmm_resilience.ml.predict import predict_network
from swmm_resilience.simulation.runner import run_simulation_simple
from swmm_resilience.ml.trainer import train_models
from swmm_resilience.simulation.runner import run_simulation_simple
from swmm_resilience.visualization.flood_map import (
    generate_flood_map,
    generate_flood_maps_by_shape,
)
from swmm_resilience.visualization.runtime_caption import format_runtime_text
from swmm_resilience.visualization.hydrograph import plot_hydrograph
from swmm_resilience.visualization.network_map import generate_network_map
from swmm_resilience.analysis.resilience import compute_resilience_curve
from swmm_resilience.visualization.resilience_curve import plot_resilience_curve
from swmm_resilience.analysis.flood_volume import compute_flood_volume_curve
from swmm_resilience.analysis.factor_comparison import generate_factor_comparisons
from swmm_resilience.visualization.flood_volume_curve import (
    plot_flood_volume_combined,
    plot_flood_volume_curve,
)

MODELS_DIR = Path("outputs/models")
METRICS_DIR = Path("outputs/metrics")
SQL_DB_PATH = Path("outputs/training_v17.sqlite3")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de predicción hidráulica — Chico Sur")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Saltar extracción (leer CSV existente)")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Saltar SWMM y reusar .rpt existentes no soportado aun; usa --skip-extraction si ya tienes dataset")
    parser.add_argument("--only-ml", action="store_true",
                        help="Solo entrenar y evaluar desde CSV existente")
    parser.add_argument("--only-maps", action="store_true",
                        help="Solo generar mapas desde CSV existente")
    parser.add_argument("--persist-sql", action="store_true",
                        help="Persistir dataset_final.csv y un entrenamiento GroupKFold5 "
                             "en outputs/training_v17.sqlite3 (esquema SQLite v17)")
    parser.add_argument("--predict", action="store_true",
                        help="Inferencia sin SWMM para el factor dado")
    parser.add_argument("--simulate", action="store_true",
                        help="Correr SWMM + ML para un factor arbitrario y generar ambos mapas")
    parser.add_argument("--factor", type=float, help="Factor para --predict o --simulate")
    parser.add_argument("--hydrograph", action="store_true",
                        help="Graficar hidrograma del nodo con mayor caudal pico")
    parser.add_argument("--network-map", action="store_true",
                        help="Generar mapa de topología de la red con clasificación de tuberías")
    parser.add_argument("--resilience-curve", action="store_true",
                        help="Calcular y graficar curva de resiliencia SWMM vs ML")
    parser.add_argument("--flood-volume-curve", action="store_true",
                        help="Graficar volumen total de inundación por factor (SWMM vs ML)")
    parser.add_argument(
        "--factor-comparison",
        "--factor-comparisons",
        dest="factor_comparison",
        action="store_true",
        help="Graficar volumen SWMM vs XGBoost por nodo y paridad para cada factor",
    )
    parser.add_argument("--evaluate-hydrographs", metavar="DIR",
                        help="Directorio de archivos CSV de hidrogramas para validación batch")
    parser.add_argument("--evaluate-shapes", action="store_true",
                        help="Evaluar SWMM vs ML para cada forma de hidrograma en hydrograph_shapes_dir")
    parser.add_argument("--evaluate-generalization", action="store_true",
                        help="Evaluar generalización: SWMM vs ML en factores no vistos (puntos medios entre factores de entrenamiento)")
    parser.add_argument("--analyze-features", action="store_true",
                        help="Correlación, ablación y SHAP para los features del modelo")
    parser.add_argument("--base-inp", metavar="PATH",
                        help="Ruta al archivo .inp base de SWMM (requerido con --evaluate-hydrographs)")
    parser.add_argument("--clf-path", metavar="PATH",
                        help="Ruta al clasificador entrenado (.pkl) (requerido con --evaluate-hydrographs)")
    parser.add_argument("--reg-path", metavar="PATH",
                        help="Ruta al regresor entrenado (.pkl) (requerido con --evaluate-hydrographs)")
    parser.add_argument("--flood-threshold", type=float, default=None,
                        help="Umbral mínimo de volumen (m³) para considerar un nodo inundado (default: desde config.yaml o 1.0)")
    parser.add_argument("--allow-inp-mismatch", action="store_true",
                        help="Continuar (con warning) si el .inp base no coincide con el hash de entrenamiento")
    parser.add_argument("--out-dir", metavar="PATH", default="./validation_output",
                        help="Directorio de salida para validación batch (default: ./validation_output)")
    args = parser.parse_args()

    if args.skip_simulation and not (args.skip_extraction or args.only_ml):
        parser.error("--skip-simulation requiere --skip-extraction o --only-ml en esta version; el pipeline aun no indexa .rpt persistentes")

    config = load_config("config.yaml")

    # ── Modo: inferencia ──────────────────────────────────────────────────────
    if args.predict:
        if args.factor is None:
            parser.error("--predict requiere --factor VALUE")
        print(f"\nPrediciendo para factor={args.factor}...")
        t0 = time.perf_counter()
        result = predict_network(args.factor, config, MODELS_DIR)
        t_ml = time.perf_counter() - t0
        map_out = config.visualization.output_path / f"flood_map_pred_{args.factor:.2f}.png"
        generate_flood_map(
            config.network.inp_path,
            result.rename(columns={"vol_pred_m3": "vol_inundacion_m3"}),
            args.factor, map_out, config.network.name,
            config.visualization.colormap, config.visualization.show_labels_top_n,
            runtime_text=format_runtime_text(t_ml),
        )
        flooded = result[result["inunda_pred"] == 1]
        print(f"\n{len(flooded)} nodos predichos como inundados:")
        print(flooded.to_string(index=False))

        print(f"\nCorriendo SWMM para verificación (factor={args.factor})...")
        swmm_run_dir = Path(tempfile.mkdtemp(prefix="swmm_verify_"))
        rpt_path = run_simulation_simple(config.network.inp_path, args.factor, swmm_run_dir)
        node_ids = result["node_id"].tolist()
        swmm_df = extract_labels(rpt_path, node_ids, config.dataset.flood_threshold_m3)
        swmm_map_out = config.visualization.output_path / f"flood_map_swmm_{args.factor:.2f}.png"
        generate_flood_map(
            config.network.inp_path,
            swmm_df,
            args.factor, swmm_map_out, config.network.name,
            config.visualization.colormap, config.visualization.show_labels_top_n,
        )
        swmm_flooded = swmm_df[swmm_df["inunda"] == 1]
        print(f"{len(swmm_flooded)} nodos inundados según SWMM")
        print(f"  Mapa ML  : {map_out}")
        print(f"  Mapa SWMM: {swmm_map_out}")
        return

    # ── Modo: simulación + inferencia para un factor arbitrario ──────────────
    if args.simulate:
        if args.factor is None:
            parser.error("--simulate requiere --factor VALUE")
        factor = args.factor
        print(f"\nSimulando factor={factor} con SWMM y ML...")

        # ── SWMM ─────────────────────────────────────────────────────────────
        all_node_ids = extract_static_features(config.network.inp_path)["node_id"].tolist()
        run_dir = Path(tempfile.mkdtemp(prefix="swmm_simulate_"))
        try:
            t0 = time.perf_counter()
            rpt_path = run_simulation_simple(config.network.inp_path, factor, run_dir)
            t_swmm = time.perf_counter() - t0
        finally:
            # keep rpt long enough to extract labels, then clean up
            pass

        try:
            labels_df = extract_labels(rpt_path, all_node_ids, config.dataset.flood_threshold_m3)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

        swmm_out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
        generate_flood_map(
            config.network.inp_path, labels_df, factor, swmm_out,
            config.network.name, config.visualization.colormap,
            config.visualization.show_labels_top_n,
            runtime_text=format_runtime_text(t_swmm),
        )
        print(f"  SWMM  ({t_swmm:.2f} s) → {swmm_out}")

        # ── ML ────────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        result = predict_network(factor, config, MODELS_DIR)
        t_ml = time.perf_counter() - t0

        pred_out = config.visualization.output_path / f"flood_map_pred_{factor:.2f}.png"
        generate_flood_map(
            config.network.inp_path,
            result.rename(columns={"vol_pred_m3": "vol_inundacion_m3"}),
            factor, pred_out, config.network.name,
            config.visualization.colormap, config.visualization.show_labels_top_n,
            runtime_text=format_runtime_text(t_ml),
        )
        print(f"  ML    ({t_ml:.4f} s) → {pred_out}")
        return

    # ── Modo: hidrograma ──────────────────────────────────────────────────────
    if args.hydrograph:
        out = config.visualization.output_path / "hydrograph_Qx1.png"
        plot_hydrograph(config.network.inp_path, out)
        return

    # ── Modo: mapa de red ────────────────────────────────────────────────────
    if args.network_map:
        out = config.visualization.output_path / "network_map.png"
        generate_network_map(config.network.inp_path, out, config.network.name)
        return

    # ── Modo: curva de resiliencia ────────────────────────────────────────────
    if args.resilience_curve:
        print("\nCalculando curva de resiliencia...")
        df = pd.read_csv(config.dataset.output_path)
        if "shape_id" in df.columns:
            df = df[df["shape_id"] == "base"]
        factors = sorted(df["factor_mult"].unique())
        result = compute_resilience_curve(df, factors, config, MODELS_DIR)
        print("\nResiliencia por factor:")
        print(result.to_string(index=False))
        plot_resilience_curve(result, METRICS_DIR)
        return

    # ── Modo: curva de volumen de inundación ─────────────────────────────────
    if args.flood_volume_curve:
        print("\nCalculando curva de volumen de inundación...")
        df = pd.read_csv(config.dataset.output_path)
        if "shape_id" in df.columns:
            df = df[df["shape_id"] == "base"]
        factors = sorted(df["factor_mult"].unique())
        result = compute_flood_volume_curve(df, factors, config, MODELS_DIR)
        print("\nVolumen total por factor:")
        print(result.to_string(index=False))
        plot_flood_volume_curve(result, METRICS_DIR)
        plot_flood_volume_combined(result, METRICS_DIR)
        return

    # -- Modo: comparacion SWMM vs XGBoost por factor -------------------------
    if args.factor_comparison:
        output_dir = METRICS_DIR / "factor_comparison"
        print("\nGenerando comparaciones SWMM vs XGBoost por factor...")
        paths = generate_factor_comparisons(
            dataset_path=config.dataset.output_path,
            config=config,
            models_dir=MODELS_DIR,
            output_dir=output_dir,
        )
        print(f"\nGraficas generadas ({len(paths)}):")
        for path in paths:
            print(f"  {path}")
        return

    # ── Modo: análisis de features ───────────────────────────────────────────
    if args.analyze_features:
        import joblib
        from swmm_resilience.ml.feature_analysis import (
            plot_correlation,
            run_ablation,
            plot_shap,
        )

        dataset_path = Path(config.dataset.output_path)
        if not dataset_path.exists():
            parser.error(
                f"--analyze-features requiere el dataset en {dataset_path}; "
                "ejecuta el pipeline completo primero"
            )

        clf_path = MODELS_DIR / "classifier.joblib"
        reg_path = MODELS_DIR / "regressor.joblib"
        if not clf_path.exists() or not reg_path.exists():
            parser.error(
                "--analyze-features requiere modelos entrenados en outputs/models/; "
                "ejecuta python main.py --only-ml primero"
            )

        df_analysis = pd.read_csv(dataset_path)
        # joblib/pickle is safe here: files are written by this pipeline's own
        # train_models() call and never loaded from an external or untrusted source.
        clf_pipeline = joblib.load(clf_path)
        reg_pipeline = joblib.load(reg_path)
        out_dir = Path("outputs/feature_analysis")

        print("\nAnálisis de features...")
        print("  1/3  Correlación...")
        plot_correlation(df_analysis, out_dir)
        print("  2/3  Ablación (LOSO × 2 runs) — puede tardar ~2 min...")
        ablation_result = run_ablation(df_analysis, config, out_dir)
        print(
            f"        Full  → F1={ablation_result['full']['classifier']['f1']:.3f}  "
            f"NSE={ablation_result['full']['regressor_oracle']['nse']:.3f}"
        )
        print(
            f"        Reduced → F1={ablation_result['reduced']['classifier']['f1']:.3f}  "
            f"NSE={ablation_result['reduced']['regressor_oracle']['nse']:.3f}"
        )
        print("  3/3  SHAP...")
        plot_shap(clf_pipeline, reg_pipeline, df_analysis, out_dir)
        print(f"\nResultados guardados en {out_dir}/")
        return

    # ── Modo: evaluación de formas de hidrograma ─────────────────────────────
    if args.evaluate_shapes:
        if not config.dataset.output_path.exists():
            parser.error(
                f"--evaluate-shapes requiere el dataset entrenado en "
                f"{config.dataset.output_path}; ejecuta el pipeline completo primero"
            )

        from swmm_resilience.simulation.hydrograph_shapes import load_all_shapes
        from swmm_resilience.validation.hydrograph_csv import write_shape_validation_csv
        from swmm_resilience.validation.hydrograph_batch import run_batch_validation

        shapes = load_all_shapes(config.simulation.hydrograph_shapes_dir) if config.simulation.hydrograph_shapes_dir else {}

        # Add the original .inp base timeseries as a shape called "base"
        from swmm_resilience.simulation.swmm_api_io import load_inp as _load_inp
        _base_inp = _load_inp(config.network.inp_path)
        _base_ts_raw = (
            next(
                (list(ts.data) for ts in _base_inp["TIMESERIES"].values() if ts.data),
                [],
            )
            if "TIMESERIES" in _base_inp
            else []
        )
        if _base_ts_raw:
            _peak = max(v for _, v in _base_ts_raw)
            if _peak > 0:
                shapes = {"base": [(t, v / _peak) for t, v in _base_ts_raw], **shapes}

        _df = pd.read_csv(config.dataset.output_path)
        base_inflows = _df.groupby("node_id")["base_inflow_lps"].first().to_dict()
        # Only nodes with non-zero inflow are written to validation CSVs
        expected_nodes = {nid for nid, v in base_inflows.items() if v > 0}
        factors_to_eval = config.factors()          # all 25 factors (0.2 → 5.0)
        tmp_val_root = Path(tempfile.mkdtemp(prefix="shape_val_"))

        print(
            f"\nEvaluando {len(shapes)} formas × {len(factors_to_eval)} factores "
            f"({factors_to_eval}) ..."
        )
        for shape_id, shape in shapes.items():
            shape_csv_dir = tmp_val_root / shape_id
            for factor in factors_to_eval:
                write_shape_validation_csv(shape_id, shape, base_inflows, factor, shape_csv_dir)

            out_dir = config.visualization.output_path / shape_id
            print(f"\n  [{shape_id}]  → {out_dir}")
            summary = run_batch_validation(
                csv_dir=shape_csv_dir,
                base_inp_path=config.network.inp_path,
                clf_path=MODELS_DIR / "classifier.joblib",
                reg_path=MODELS_DIR / "regressor.joblib",
                flood_threshold_m3=config.dataset.flood_threshold_m3,
                out_dir=out_dir,
                expected_nodes=expected_nodes,
                drain_down_hours=config.validation.drain_down_hours,
                factor_range=(config.simulation.factor_min, config.simulation.factor_max),
            )
            cls = summary["classification"]
            vol = summary["volume"]
            print(
                f"    Escenarios: {summary['n_scenarios']}  "
                f"F1={cls.get('f1', 0):.3f}  "
                f"NSE={vol.get('nse', float('nan')):.3f}  "
                f"Vol_error={vol.get('error_pct_total', float('nan')):.1f}%"
            )
            if summary.get("timings"):
                t_sw = sum(t["t_swmm_s"] for t in summary["timings"])
                t_ml = sum(t["t_features_s"] + t["t_inference_s"] for t in summary["timings"])
                speedups = [t["speedup"] for t in summary["timings"] if t.get("speedup")]
                print(
                    f"    SWMM: {t_sw:.2f} s  ML: {t_ml:.4f} s"
                    + (f"  x{sum(speedups)/len(speedups):.0f}" if speedups else "")
                )
        return

    # ── Modo: evaluación de generalización (factores no vistos) ──────────────
    if args.evaluate_generalization:
        if not config.dataset.output_path.exists():
            parser.error(
                f"--evaluate-generalization requiere el dataset en "
                f"{config.dataset.output_path}; ejecuta el pipeline completo primero"
            )

        from swmm_resilience.simulation.hydrograph_shapes import load_all_shapes
        from swmm_resilience.validation.hydrograph_csv import write_shape_validation_csv
        from swmm_resilience.validation.hydrograph_batch import run_batch_validation

        shapes = load_all_shapes(config.simulation.hydrograph_shapes_dir) if config.simulation.hydrograph_shapes_dir else {}

        from swmm_resilience.simulation.swmm_api_io import load_inp as _load_inp
        _base_inp = _load_inp(config.network.inp_path)
        _base_ts_raw = (
            next(
                (list(ts.data) for ts in _base_inp["TIMESERIES"].values() if ts.data),
                [],
            )
            if "TIMESERIES" in _base_inp
            else []
        )
        if _base_ts_raw:
            _peak = max(v for _, v in _base_ts_raw)
            if _peak > 0:
                shapes = {"base": [(t, v / _peak) for t, v in _base_ts_raw], **shapes}

        _df = pd.read_csv(config.dataset.output_path)
        base_inflows = _df.groupby("node_id")["base_inflow_lps"].first().to_dict()
        expected_nodes = {nid for nid, v in base_inflows.items() if v > 0}

        training_factors = config.factors()
        unseen_factors = [
            round((training_factors[i] + training_factors[i + 1]) / 2, 3)
            for i in range(len(training_factors) - 1)
        ]

        tmp_gen_root = Path(tempfile.mkdtemp(prefix="gen_eval_"))
        out_root = Path("outputs/generalization")

        print(
            f"\nGeneralization evaluation: {len(shapes)} shapes × "
            f"{len(unseen_factors)} unseen factors (midpoints between training steps)"
        )
        print(f"Unseen factors: {unseen_factors}")

        for shape_id, shape in shapes.items():
            shape_csv_dir = tmp_gen_root / shape_id
            for factor in unseen_factors:
                write_shape_validation_csv(shape_id, shape, base_inflows, factor, shape_csv_dir)

            out_dir = out_root / shape_id
            print(f"\n  [{shape_id}]  → {out_dir}")
            summary = run_batch_validation(
                csv_dir=shape_csv_dir,
                base_inp_path=config.network.inp_path,
                clf_path=MODELS_DIR / "classifier.joblib",
                reg_path=MODELS_DIR / "regressor.joblib",
                flood_threshold_m3=config.dataset.flood_threshold_m3,
                out_dir=out_dir,
                expected_nodes=expected_nodes,
                drain_down_hours=config.validation.drain_down_hours,
                factor_range=(config.simulation.factor_min, config.simulation.factor_max),
            )
            cls = summary["classification"]
            vol = summary["volume"]
            print(
                f"    Scenarios: {summary['n_scenarios']}  "
                f"F1={cls.get('f1', 0):.3f}  "
                f"NSE={vol.get('nse', float('nan')):.3f}  "
                f"Vol_error={vol.get('error_pct_total', float('nan')):.1f}%"
            )
            if summary.get("timings"):
                t_sw = sum(t["t_swmm_s"] for t in summary["timings"])
                t_ml = sum(t["t_features_s"] + t["t_inference_s"] for t in summary["timings"])
                speedups = [t["speedup"] for t in summary["timings"] if t.get("speedup")]
                print(
                    f"    SWMM: {t_sw:.2f} s  ML: {t_ml:.4f} s"
                    + (f"  x{sum(speedups)/len(speedups):.0f}" if speedups else "")
                )
        return

    # ── Modo: validación batch de hidrogramas ────────────────────────────────
    if args.evaluate_hydrographs:
        if args.base_inp is None:
            parser.error("--evaluate-hydrographs requiere --base-inp PATH")
        if args.clf_path is None:
            parser.error("--evaluate-hydrographs requiere --clf-path PATH")
        if args.reg_path is None:
            parser.error("--evaluate-hydrographs requiere --reg-path PATH")

        # Resolve flood threshold: CLI > config.yaml > fallback 1.0
        if args.flood_threshold is not None:
            flood_threshold = args.flood_threshold
        else:
            try:
                flood_threshold = config.dataset.flood_threshold_m3
            except Exception:
                flood_threshold = 1.0

        try:
            drain_down_hours = config.validation.drain_down_hours
        except Exception:
            drain_down_hours = 6.0
        try:
            factor_range = (config.simulation.factor_min, config.simulation.factor_max)
        except Exception:
            factor_range = None

        from swmm_resilience.validation.hydrograph_batch import run_batch_validation

        print(f"\nValidando hidrogramas en '{args.evaluate_hydrographs}'...")
        summary = run_batch_validation(
            csv_dir=Path(args.evaluate_hydrographs),
            base_inp_path=Path(args.base_inp),
            clf_path=Path(args.clf_path),
            reg_path=Path(args.reg_path),
            flood_threshold_m3=flood_threshold,
            out_dir=Path(args.out_dir),
            drain_down_hours=drain_down_hours,
            allow_inp_mismatch=args.allow_inp_mismatch,
            factor_range=factor_range,
        )

        print("\n" + "=" * 50)
        print("RESUMEN DE VALIDACIÓN DE HIDROGRAMAS")
        print("=" * 50)
        print(f"  Escenarios procesados : {summary['n_scenarios']}")
        print("\n  Métricas de clasificación:")
        for k, v in summary["classification"].items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        if summary.get("pr_auc") is not None:
            print(f"    pr_auc: {summary['pr_auc']:.4f}")
        print("\n  Métricas de volumen:")
        for k, v in summary["volume"].items():
            if v is None:
                print(f"    {k}: N/A")
            elif isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        for k, v in summary.get("volume_flooded_only", {}).items():
            print(f"    {k}: {'N/A' if v is None else v}")

        if summary.get("per_scenario"):
            print("\n  Volumen total inundado por escenario (SWMM vs ML):")
            for rec in summary["per_scenario"]:
                err_pct = rec.get("error_pct")
                err_txt = f"{err_pct:+.1f}%" if err_pct is not None else "N/A"
                extra = rec.get("n_extrapolated", 0)
                extra_txt = f"  [extrapolados: {extra}]" if extra else ""
                print(
                    f"    {rec['scenario_id']}: SWMM={rec['vol_total_swmm_m3']:.1f} m³  "
                    f"ML={rec['vol_total_pred_m3']:.1f} m³  ({err_txt}){extra_txt}"
                )

        if summary.get("timings"):
            speedups = [t["speedup"] for t in summary["timings"] if t.get("speedup")]
            t_swmm_total = sum(t["t_swmm_s"] for t in summary["timings"])
            t_ml_total = sum(
                t["t_features_s"] + t["t_inference_s"] for t in summary["timings"]
            )
            print("\n  Tiempos de cómputo por escenario:")
            for t in summary["timings"]:
                t_ml = t["t_features_s"] + t["t_inference_s"]
                sp = t.get("speedup")
                sp_txt = f"  x{sp:.0f}" if sp else ""
                print(
                    f"    {t['scenario_id']:<30}  SWMM: {t['t_swmm_s']:.3f} s  "
                    f"ML: {t_ml:.4f} s{sp_txt}"
                )
            print(f"    {'TOTAL':<30}  SWMM: {t_swmm_total:.3f} s  ML: {t_ml_total:.4f} s")
            if speedups:
                print(f"    Speed-up medio  : x{sum(speedups) / len(speedups):.0f}")

        print(f"\n  CSV de resumen     : {summary['summary_csv_path']}")
        print(f"  Totales/escenario  : {summary['scenario_totals_csv_path']}")
        print(f"  Tiempos            : {summary['timings_csv_path']}")
        print(f"  Métricas/escenario : {summary['metrics_per_scenario_csv_path']}")
        return

    # ── Modo: persistir en SQLite v17 ───────────────────────────────────────
    if args.persist_sql:
        from swmm_resilience.database.connection import connect_managed_database
        from swmm_resilience.database.migrations import apply_migrations
        from swmm_resilience.database.csv_backfill import (
            backfill_networks_and_runs,
            persist_training_run,
        )

        print(f"\nLeyendo dataset desde {config.dataset.output_path}...")
        df = pd.read_csv(config.dataset.output_path)
        if "shape_id" not in df.columns:
            parser.error(
                "dataset_final.csv no tiene columna shape_id — vuelve a correr "
                "el pipeline completo antes de --persist-sql"
            )
        print(f"  {df.shape[0]} filas x {df.shape[1]} cols")

        SQL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = connect_managed_database(SQL_DB_PATH)
        apply_migrations(conn)

        print(f"\nPersistiendo redes/nodos/escenarios/corridas en {SQL_DB_PATH}...")
        info = backfill_networks_and_runs(conn, df, config.network.inp_path, config.network.name)
        print(f"  network_id={info['network_id']}  nodos={len(info['node_pk_by_id'])}  "
              f"corridas={len(info['run_id_by_key'])}")

        print("\nEntrenando (GroupKFold5 por factor_mult) y persistiendo evidencia...")
        training_run_id = persist_training_run(
            conn, df, info["run_id_by_key"], info["node_pk_by_id"], config
        )
        metrics = conn.execute(
            "SELECT metric_name, value FROM model_metrics WHERE owner_kind = 'model'"
        ).fetchall()
        print(f"  training_run_id={training_run_id} (status=COMPLETE)")
        for name, value in metrics:
            print(f"  {name}: {value:.4f}" if value is not None else f"  {name}: NULL")
        print(
            "\n  Nota: esto guarda training_runs / model_evaluations / oof_predictions / "
            "trained_models con evidencia real, pero NO construye el sistema opcional de "
            "candidatos/rankings/promociones — los modelos quedan como artefactos "
            "historicos validos, no como una 'seleccion activa'."
        )
        conn.close()
        return

    # ── Modo: solo mapas ─────────────────────────────────────────────────────
    if args.only_maps:
        df = pd.read_csv(config.dataset.output_path)
        if "shape_id" in df.columns:
            df = df[df["shape_id"] == "base"]
        factors_in_dataset = sorted(df["factor_mult"].unique())
        print(f"Generando {len(factors_in_dataset)} mapas SWMM...")
        for factor in factors_in_dataset:
            df_f = df[abs(df["factor_mult"] - factor) < 1e-6]
            out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
            # Volúmenes salen del dataset; se corre SWMM solo para medir el
            # tiempo de cómputo que se estampa en el mapa.
            run_dir = Path(tempfile.mkdtemp(prefix="swmm_timing_"))
            try:
                t0 = time.perf_counter()
                run_simulation_simple(config.network.inp_path, factor, run_dir)
                t_swmm = time.perf_counter() - t0
            finally:
                shutil.rmtree(run_dir, ignore_errors=True)
            generate_flood_map(
                config.network.inp_path, df_f, factor, out,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
                runtime_text=format_runtime_text(t_swmm),
            )
        print(f"  Mapas guardados en {config.visualization.output_path}")

        df_all = pd.read_csv(config.dataset.output_path)
        if "shape_id" in df_all.columns:
            print("\nGenerando mapas de inundación por forma de hidrograma...")
            written = generate_flood_maps_by_shape(
                config.network.inp_path, df_all, config.visualization.output_path,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )
            n_maps = sum(len(paths) for paths in written.values())
            print(f"  {n_maps} mapas en {len(written)} carpetas (una por forma) bajo "
                  f"{config.visualization.output_path}")
        return

    # ── Pipeline completo (o parcial) ─────────────────────────────────────────
    use_existing_dataset = args.skip_extraction or args.only_ml

    factors = config.factors()
    n_factors = len(factors)
    static_topo_df = None
    all_node_ids = []
    n_nodes = 0

    if not use_existing_dataset:
        print("\nExtrayendo features estáticas...")
        static_df = extract_static_features(config.network.inp_path)
        print(f"  {len(static_df)} nodos junction")

        print("Calculando features topológicas...")
        static_topo_df = compute_topology_features(static_df, config.network.inp_path)

        all_node_ids = static_topo_df["node_id"].tolist()
        n_nodes = len(all_node_ids)
        print(f"  Red: {n_nodes} nodos, {n_factors} factores ({factors[0]:.2f}–{factors[-1]:.2f})")

    if not use_existing_dataset:
        from swmm_resilience.simulation.batch import run_batch, run_batch_shapes
        from swmm_resilience.simulation.hydrograph_shapes import (
            get_shape_stats, load_all_shapes,
        )
        from swmm_resilience.simulation.swmm_api_io import load_inp as _load_inp
        run_dir = Path(tempfile.mkdtemp(prefix="swmm_runs_"))

        # ── Base shape: derive duracion_horas / tiempo_al_pico_h from .inp ──
        _base_inp = _load_inp(config.network.inp_path)
        _base_ts_data = (
            next(
                (list(ts.data) for ts in _base_inp["TIMESERIES"].values() if ts.data),
                [],
            )
            if "TIMESERIES" in _base_inp
            else []
        )
        base_dur, base_t_pico = get_shape_stats(_base_ts_data)

        print(
            f"\nCorriendo {n_factors} simulaciones SWMM (forma base, "
            f"dur={base_dur:.2f}h, t_pico={base_t_pico:.2f}h)..."
        )
        sim_results = run_batch(config, run_dir)

        print("\nExtrayendo labels y features dinámicas...")
        simulation_results = []
        for factor, rpt_path in sim_results:
            dynamic_df = compute_dynamic_features(
                static_topo_df, factor,
                duracion_horas=base_dur,
                tiempo_al_pico_h=base_t_pico,
            )
            dynamic_df["shape_id"] = "base"
            labels_df = extract_labels(rpt_path, all_node_ids, config.dataset.flood_threshold_m3)
            simulation_results.append((factor, dynamic_df, labels_df))

        # ── Additional shapes ─────────────────────────────────────────────
        if config.simulation.hydrograph_shapes_dir is not None:
            shapes = load_all_shapes(config.simulation.hydrograph_shapes_dir)
            base_inflows = dict(
                zip(
                    static_topo_df["node_id"].astype(str),
                    static_topo_df["base_inflow_lps"],
                )
            )
            n_extra = len(shapes) * n_factors
            print(
                f"\nCorriendo {len(shapes)} formas × {n_factors} factores "
                f"= {n_extra} simulaciones SWMM adicionales..."
            )
            shape_results = run_batch_shapes(config, shapes, base_inflows, run_dir)
            for shape_id, factor, rpt_path in shape_results:
                dur_h, t_pico_h = get_shape_stats(shapes[shape_id])
                dynamic_df = compute_dynamic_features(
                    static_topo_df, factor,
                    duracion_horas=dur_h,
                    tiempo_al_pico_h=t_pico_h,
                )
                dynamic_df["shape_id"] = shape_id
                labels_df = extract_labels(
                    rpt_path, all_node_ids, config.dataset.flood_threshold_m3
                )
                simulation_results.append((factor, dynamic_df, labels_df))

        n_simulations = len(simulation_results)
        print("\nEnsamblando dataset...")
        df = assemble_dataset(static_topo_df, simulation_results, config.dataset.output_path)
        print("Validando dataset...")
        validate_dataset(df, n_nodes, n_simulations)
        print(f"  Dataset validado: {df.shape}")
    else:
        print(f"\nLeyendo dataset desde {config.dataset.output_path}...")
        df = pd.read_csv(config.dataset.output_path)
        print(f"  {df.shape[0]} filas × {df.shape[1]} cols")

    print("\nEntrenando modelos finales...")
    clf, reg = train_models(df, config, MODELS_DIR)

    print("\nEvaluando modelos...")
    results = evaluate_models(df, config, METRICS_DIR)

    print("\nGenerando gráficos de importancia de variables...")
    generate_feature_importance_plots(clf, reg, METRICS_DIR)

    if not args.only_ml:
        print("\nGenerando mapas de inundación (forma base)...")
        df_base = df[df["shape_id"] == "base"] if "shape_id" in df.columns else df
        for factor in config.visualization.factors_to_plot:
            df_f = df_base[abs(df_base["factor_mult"] - factor) < 1e-6]
            if df_f.empty:
                continue
            out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
            generate_flood_map(
                config.network.inp_path, df_f, factor, out,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )

        if "shape_id" in df.columns:
            print("\nGenerando mapas de inundación por forma de hidrograma...")
            written = generate_flood_maps_by_shape(
                config.network.inp_path, df, config.visualization.output_path,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )
            n_maps = sum(len(paths) for paths in written.values())
            print(f"  {n_maps} mapas en {len(written)} carpetas (una por forma) bajo "
                  f"{config.visualization.output_path}")

    print("\n" + "=" * 50)
    print("RESUMEN DE MÉTRICAS (LOSO)")
    print("=" * 50)
    loso = results.get("LOSO", {})
    labels_map = {
        "classifier": "Clasificador",
        "regressor_oracle": "Regresor (oracle — etiquetas reales para filtrar)",
        "end_to_end": "Sistema end-to-end",
    }
    for level, label in labels_map.items():
        m = loso.get(level, {})
        if m:
            print(f"\n{label}:")
            for k, v in m.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
