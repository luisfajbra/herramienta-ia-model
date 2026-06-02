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

    # ── Modo: solo mapas ─────────────────────────────────────────────────────
    if args.only_maps:
        df = pd.read_csv(config.dataset.output_path)
        for factor in config.visualization.factors_to_plot:
            df_f = df[abs(df["factor_mult"] - factor) < 1e-6]
            if df_f.empty:
                print(f"Factor {factor:.2f} no encontrado en dataset, saltando.")
                continue
            out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
            generate_flood_map(
                config.network.inp_path, df_f, factor, out,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )
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
