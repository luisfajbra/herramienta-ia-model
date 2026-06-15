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
import tempfile
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
from swmm_resilience.ml.trainer import train_models
from swmm_resilience.visualization.flood_map import generate_flood_map
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
    parser.add_argument("--predict", action="store_true",
                        help="Inferencia sin SWMM para el factor dado")
    parser.add_argument("--factor", type=float, help="Factor para --predict")
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
        result = predict_network(args.factor, config, MODELS_DIR)
        map_out = config.visualization.output_path / f"flood_map_pred_{args.factor:.2f}.png"
        generate_flood_map(
            config.network.inp_path,
            result.rename(columns={"vol_pred_m3": "vol_inundacion_m3"}),
            args.factor, map_out, config.network.name,
            config.visualization.colormap, config.visualization.show_labels_top_n,
        )
        flooded = result[result["inunda_pred"] == 1]
        print(f"\n{len(flooded)} nodos predichos como inundados:")
        print(flooded.to_string(index=False))
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
            print("\n  Tiempos de cómputo:")
            print(f"    SWMM total      : {t_swmm_total:.2f} s")
            print(f"    ML total        : {t_ml_total:.4f} s (features + inferencia)")
            if speedups:
                print(f"    Speed-up medio  : x{sum(speedups) / len(speedups):.0f}")

        print(f"\n  CSV de resumen     : {summary['summary_csv_path']}")
        print(f"  Totales/escenario  : {summary['scenario_totals_csv_path']}")
        print(f"  Tiempos            : {summary['timings_csv_path']}")
        print(f"  Métricas/escenario : {summary['metrics_per_scenario_csv_path']}")
        return

    # ── Modo: solo mapas ─────────────────────────────────────────────────────
    if args.only_maps:
        df = pd.read_csv(config.dataset.output_path)
        factors_in_dataset = sorted(df["factor_mult"].unique())
        print(f"Generando {len(factors_in_dataset)} mapas SWMM...")
        for factor in factors_in_dataset:
            df_f = df[abs(df["factor_mult"] - factor) < 1e-6]
            out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
            generate_flood_map(
                config.network.inp_path, df_f, factor, out,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )
        print(f"  Mapas guardados en {config.visualization.output_path}")
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
        from swmm_resilience.simulation.batch import run_batch
        run_dir = Path(tempfile.mkdtemp(prefix="swmm_runs_"))
        print(f"\nCorriendo {n_factors} simulaciones SWMM en {run_dir}...")
        sim_results = run_batch(config, run_dir)

        print("\nExtrayendo labels y features dinámicas...")
        simulation_results = []
        for factor, rpt_path in sim_results:
            dynamic_df = compute_dynamic_features(static_topo_df, factor)
            labels_df = extract_labels(rpt_path, all_node_ids, config.dataset.flood_threshold_m3)
            simulation_results.append((factor, dynamic_df, labels_df))

        print("Ensamblando dataset...")
        df = assemble_dataset(static_topo_df, simulation_results, config.dataset.output_path)
        print("Validando dataset...")
        validate_dataset(df, n_nodes, n_factors)
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
        print("\nGenerando mapas de inundación...")
        for factor in config.visualization.factors_to_plot:
            df_f = df[abs(df["factor_mult"] - factor) < 1e-6]
            if df_f.empty:
                continue
            out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
            generate_flood_map(
                config.network.inp_path, df_f, factor, out,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )

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
