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
import math
import sys
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.dataset.assembler import assemble_dataset
from swmm_resilience.dataset.shape_selection import augmented_shape_rows, base_shape_rows
from swmm_resilience.dataset.validator import validate_dataset
from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.extraction.dynamic_features import compute_dynamic_features
from swmm_resilience.extraction.labels import extract_labels
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import build_network_graph, compute_topology_features
from swmm_resilience.ml.evaluator import evaluate_models
from swmm_resilience.ml.feature_importance import generate_feature_importance_plots
from swmm_resilience.ml.predict import predict_network
from swmm_resilience.visualization.runner import write_figure_provenance
from swmm_resilience.simulation.runner import run_simulation_simple
from swmm_resilience.ml.trainer import train_models
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


BENCH_ROOT = Path("outputs/bench")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hydraulic prediction pipeline — Chico Sur")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip extraction (read the existing CSV)")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Skip SWMM and reuse existing .rpt files (not supported yet; use --skip-extraction if you already have a dataset)")
    parser.add_argument("--only-ml", action="store_true",
                        help="Only train and evaluate from the existing CSV")
    parser.add_argument("--only-maps", action="store_true",
                        help="Only generate maps from the existing CSV")
    parser.add_argument("--persist-sql", action="store_true",
                        help="Persist dataset_final.csv and a GroupKFold5 training run "
                             "in outputs/training_v17.sqlite3 (SQLite schema v17)")
    parser.add_argument(
        "--v4-21-dataset",
        metavar="PATH",
        help="Explicit v4_21 CSV to backfill alongside --persist-sql; no implicit dataset path is used",
    )
    parser.add_argument("--predict", action="store_true",
                        help="Inference without SWMM for the given factor")
    parser.add_argument("--simulate", action="store_true",
                        help="Run SWMM + ML for an arbitrary factor and generate both maps")
    parser.add_argument("--factor", type=float, help="Factor for --predict or --simulate")
    parser.add_argument("--hydrograph", action="store_true",
                        help="Plot the hydrograph of the node with the highest peak flow")
    parser.add_argument("--network-map", action="store_true",
                        help="Generate the network topology map with pipe classification")
    parser.add_argument("--resilience-curve", action="store_true",
                        help="Compute and plot the SWMM vs ML resilience curve")
    parser.add_argument("--flood-volume-curve", action="store_true",
                        help="Plot total flood volume per factor (SWMM vs ML)")
    parser.add_argument(
        "--factor-comparison",
        "--factor-comparisons",
        dest="factor_comparison",
        action="store_true",
        help="Plot SWMM vs XGBoost volume per node and parity for each factor",
    )
    parser.add_argument("--evaluate-hydrographs", metavar="DIR",
                        help="Directory of hydrograph CSV files for batch validation")
    parser.add_argument("--evaluate-shapes", action="store_true",
                        help="Evaluate SWMM vs ML for each hydrograph shape in hydrograph_shapes_dir")
    parser.add_argument("--evaluate-generalization", action="store_true",
                        help="Evaluate generalization: SWMM vs ML on unseen factors (midpoints between training factors)")
    parser.add_argument("--evaluate-validation-shapes", metavar="DIR",
                        help="Validate normalized shapes from a directory outside of "
                             "training (expands them to absolute-flow CSVs per factor)")
    parser.add_argument("--factors", metavar="LIST", default=None,
                        help="Comma-separated factors for --evaluate-validation-shapes "
                             "(default: unseen midpoints between the training factors)")
    parser.add_argument("--analyze-features", action="store_true",
                        help="Correlation, ablation and SHAP for the model features")
    parser.add_argument("--base-inp", metavar="PATH",
                        help="Path to the base SWMM .inp file (required with --evaluate-hydrographs)")
    parser.add_argument("--clf-path", metavar="PATH",
                        help="Path to the trained classifier (.pkl) (required with --evaluate-hydrographs)")
    parser.add_argument("--reg-path", metavar="PATH",
                        help="Path to the trained regressor (.pkl) (required with --evaluate-hydrographs)")
    parser.add_argument("--flood-threshold", type=float, default=None,
                        help="Minimum volume threshold (m³) to consider a node flooded (default: from config.yaml or 1.0)")
    parser.add_argument("--allow-inp-mismatch", action="store_true",
                        help="Continue (with a warning) if the base .inp does not match the training hash")
    parser.add_argument("--out-dir", metavar="PATH", default="./validation_output",
                        help="Output directory for batch validation (default: ./validation_output)")
    parser.add_argument("--bench-prepare", action="store_true",
                        help="Bench stage 1: SQL -> outputs/bench/prepared/")
    parser.add_argument("--bench-train", action="store_true",
                        help="Bench stage 2: train each enabled family")
    parser.add_argument("--bench-evaluate", action="store_true",
                        help="Bench stage 3: folds -> OOF -> metrics -> ranking")
    parser.add_argument("--bench-promote", action="store_true",
                        help="Copy the winning candidate to outputs/models/")
    parser.add_argument("--bench-tune", action="store_true",
                        help="Optuna hyperparameter search over ONE family (exploratory)")
    parser.add_argument("--bench", action="store_true",
                        help="Run the four bench stages in sequence")
    parser.add_argument("--models", metavar="LIST", default=None,
                        help="Comma-separated families (default: those enabled in config.yaml)")
    parser.add_argument("--prep-id", metavar="ID", default=None,
                        help="Specific PreparedDataset (default: the most recent one)")
    return parser


def _run_bench(args, config) -> None:
    from swmm_resilience.ml.bench.baselines import (
        BASELINES,
        BaselineNotApplicable,
        evaluate_baseline,
    )
    from swmm_resilience.ml.bench.evaluate import evaluate_candidate
    from swmm_resilience.ml.bench.preprocess import prepare_dataset, resolve_prepared
    from swmm_resilience.ml.bench.promote import promote_candidate
    from swmm_resilience.ml.bench.ranking import RankingCriterion, rank_candidates
    from swmm_resilience.ml.bench.reports import (
        write_metrics_files,
        write_oof,
        write_provenance,
        write_reports,
    )
    from swmm_resilience.ml.bench.train import train_candidate

    if config.bench is None:
        raise SystemExit(
            "config.yaml has no `bench:` block. Add it before using the --bench-* commands."
        )

    prepared_dir = BENCH_ROOT / "prepared"
    candidates_dir = BENCH_ROOT / "candidates"
    reports_dir = BENCH_ROOT / "reports"
    oof_dir = BENCH_ROOT / "oof"

    run_all = args.bench
    families = (
        [name.strip() for name in args.models.split(",")]
        if args.models
        else config.bench.enabled_families()
    )

    unknown_families = [name for name in families if name not in config.bench.families]
    if unknown_families:
        known = ", ".join(sorted(config.bench.families)) or "(none)"
        raise SystemExit(
            f"Unknown family(ies) in --models: {', '.join(unknown_families)}. "
            f"Families declared in config.yaml: {known}"
        )

    if args.bench_tune and len(families) != 1:
        raise SystemExit(
            "--bench-tune requires exactly one family in --models "
            f"(received {len(families)}: {', '.join(families) or '(none)'})."
        )

    if run_all or args.bench_prepare:
        prepared = prepare_dataset(
            config.dataset.db_path,
            protocols=tuple(config.bench.protocols),
            flood_threshold_m3=config.dataset.flood_threshold_m3,
            output_dir=prepared_dir,
            max_continuity_error_pct=config.validation.max_continuity_error_pct,
        )
        print(f"PreparedDataset {prepared.prep_id}: {prepared.quality['n_rows']} rows, "
              f"{prepared.quality['class_balance']['n_flooded']} flooded")
    elif args.bench_train or args.bench_evaluate or args.bench_tune:
        try:
            prepared = resolve_prepared(
                prepared_dir, args.prep_id, tuple(config.bench.protocols)
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        prepared = None

    if run_all or args.bench_train:
        for family in families:
            spec = config.bench.families[family]
            artifacts = train_candidate(
                prepared, family, spec.classifier, spec.regressor, candidates_dir
            )
            print(f"  {family}: {artifacts.classifier_path}")

    ranking = None
    if run_all or args.bench_evaluate:
        manifest_threshold = prepared.manifest.get("flood_threshold_m3")
        if manifest_threshold is None:
            raise SystemExit(
                "The PreparedDataset manifest has no flood_threshold_m3; "
                "the baselines cannot label flooding. "
                "Run `python main.py --bench-prepare` to regenerate it."
            )
        baseline_threshold = float(manifest_threshold)
        if baseline_threshold != float(config.dataset.flood_threshold_m3):
            print(
                f"WARNING: the flood threshold in the config ({config.dataset.flood_threshold_m3}) "
                f"differs from the PreparedDataset one ({baseline_threshold}); the baselines "
                "use the PreparedDataset threshold to label the same way as `inunda`."
            )

        metrics_by_family = {}
        for family in families:
            spec = config.bench.families[family]
            oof, metrics = evaluate_candidate(prepared, family, spec.classifier, spec.regressor)
            metrics_by_family[family] = metrics
            write_oof(oof, oof_dir, prepared.prep_id, family)
            print(f"  {family} evaluated")

        criterion = RankingCriterion(
            primary_metric=config.bench.ranking.primary_metric,
            primary_direction=config.bench.ranking.primary_direction,
            tie_breakers=config.bench.ranking.tie_breakers,
        )
        ranking_protocol = config.bench.protocols[0]
        ranking = rank_candidates(metrics_by_family, criterion, ranking_protocol)
        write_reports(metrics_by_family, ranking, reports_dir, prep_id=prepared.prep_id)

        # Las referencias van DESPUES de escribir ranking y metricas de las
        # familias: si una falla, las horas de computo de las familias no se
        # pierden.
        baseline_metrics = {}
        baselines_failed = []
        for baseline in BASELINES:
            try:
                baseline_oof_frame, metrics = evaluate_baseline(
                    prepared, baseline, baseline_threshold
                )
            except BaselineNotApplicable as exc:
                print(f"  {baseline}: skipped ({exc})")
                continue
            except ValueError as exc:
                print(f"  {baseline} failed: {exc}")
                baselines_failed.append(baseline)
                continue
            baseline_metrics[baseline] = metrics
            write_oof(baseline_oof_frame, oof_dir, prepared.prep_id, baseline)
            print(f"  {baseline} evaluated")
        write_metrics_files(baseline_metrics, reports_dir, prep_id=prepared.prep_id)
        write_provenance(
            reports_dir,
            prepared,
            {
                family: {
                    "classifier": config.bench.families[family].classifier,
                    "regressor": config.bench.families[family].regressor,
                }
                for family in families
            },
            sorted(baseline_metrics),
            sorted(baselines_failed),
        )
        print(f"Ranking computed on protocol: {ranking_protocol}")
        print(ranking.to_string(index=False))

    if args.bench_tune:
        from swmm_resilience.ml.bench.tune import run_study, save_tune_result

        tuning_cfg = config.bench.tuning
        result = run_study(
            prepared, families[0],
            n_trials=tuning_cfg.n_trials, n_splits=tuning_cfg.n_splits,
        )
        study_path = save_tune_result(result, BENCH_ROOT / "tuning")
        print(f"Optuna study saved to {study_path}")
        print(f"status={result.status}: {result.status_reason}")
        print(
            f"Best value ({result.objective_metric}): {result.best_value:.4f} "
            f"after {result.n_trials} trials"
        )

    if args.bench_promote:
        chosen = config.bench.promote
        if chosen in {"none", "auto"}:
            raise SystemExit(
                "--bench-promote requires an explicit family other than 'none'; "
                "config.bench.promote cannot select automatically from the ranking."
            )
        record = promote_candidate(
            candidates_dir / chosen, Path("outputs/models"), config.network.inp_path
        )
        print(f"Promoted: {record['family']} (prep_id={record['prep_id']})")


def _make_stdout_unicode_safe() -> None:
    """Keep the CLI's Unicode output from crashing a legacy Windows console.

    Several branches print box-drawing and arrow characters that cp1252 cannot
    encode, which raised UnicodeEncodeError mid-run instead of just rendering
    a substitute glyph. pytest's capture objects have no reconfigure(), hence
    the guard.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def _fmt_metric(value, spec: str = ".3f", fallback: str = "N/A") -> str:
    """Format a metric the evaluator may report as None or NaN.

    dict.get(key, 0) does not help here: the keys exist, their values are
    None when a scenario has too few flooded nodes to define the metric.
    """
    if value is None:
        return fallback
    try:
        if isinstance(value, float) and math.isnan(value):
            return fallback
        return format(value, spec)
    except (TypeError, ValueError):
        return fallback


def main():
    _make_stdout_unicode_safe()
    parser = build_parser()
    args = parser.parse_args()

    if args.skip_simulation and not (args.skip_extraction or args.only_ml):
        parser.error("--skip-simulation requires --skip-extraction or --only-ml in this version; the pipeline does not yet index persistent .rpt files")
    if args.v4_21_dataset and not args.persist_sql:
        parser.error("--v4-21-dataset requires --persist-sql")

    config = load_config("config.yaml")

    if (args.bench or args.bench_prepare or args.bench_train or args.bench_evaluate
            or args.bench_promote or args.bench_tune):
        _run_bench(args, config)
        return

    # ── Modo: inferencia ──────────────────────────────────────────────────────
    if args.predict:
        if args.factor is None:
            parser.error("--predict requires --factor VALUE")
        print(f"\nPredicting for factor={args.factor}...")
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
        print(f"\n{len(flooded)} nodes predicted as flooded:")
        print(flooded.to_string(index=False))

        print(f"\nRunning SWMM for verification (factor={args.factor})...")
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
        print(f"{len(swmm_flooded)} nodes flooded according to SWMM")
        print(f"  ML map  : {map_out}")
        print(f"  SWMM map: {swmm_map_out}")
        return

    # ── Modo: simulación + inferencia para un factor arbitrario ──────────────
    if args.simulate:
        if args.factor is None:
            parser.error("--simulate requires --factor VALUE")
        factor = args.factor
        print(f"\nSimulating factor={factor} with SWMM and ML...")

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
        print("\nComputing resilience curve...")
        try:
            df_curve = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--resilience-curve requires data in {config.dataset.db_path}; "
                f"run --persist-sql first to populate it ({load_error})"
            )
        df = base_shape_rows(df_curve)
        factors = sorted(df["factor_mult"].unique())
        result = compute_resilience_curve(df, factors, config, MODELS_DIR)
        print("\nResilience per factor:")
        print(result.to_string(index=False))
        plot_resilience_curve(result, METRICS_DIR)
        write_figure_provenance(METRICS_DIR, MODELS_DIR)
        return

    # ── Modo: curva de volumen de inundación ─────────────────────────────────
    if args.flood_volume_curve:
        print("\nComputing flood volume curve...")
        try:
            df_curve = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--flood-volume-curve requires data in {config.dataset.db_path}; "
                f"run --persist-sql first to populate it ({load_error})"
            )
        df = base_shape_rows(df_curve)
        factors = sorted(df["factor_mult"].unique())
        result = compute_flood_volume_curve(df, factors, config, MODELS_DIR)
        print("\nTotal volume per factor:")
        print(result.to_string(index=False))
        plot_flood_volume_curve(result, METRICS_DIR)
        plot_flood_volume_combined(result, METRICS_DIR)
        write_figure_provenance(METRICS_DIR, MODELS_DIR)
        return

    # -- Modo: comparacion SWMM vs XGBoost por factor -------------------------
    if args.factor_comparison:
        output_dir = METRICS_DIR / "factor_comparison"
        print("\nGenerating SWMM vs XGBoost comparisons per factor...")
        try:
            factor_comparison_frame = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--factor-comparison requires data in {config.dataset.db_path}; "
                f"run --persist-sql first to populate it ({load_error})"
            )
        paths = generate_factor_comparisons(
            frame=factor_comparison_frame,
            config=config,
            models_dir=MODELS_DIR,
            output_dir=output_dir,
        )
        write_figure_provenance(output_dir, MODELS_DIR)
        print(f"\nPlots generated ({len(paths)}):")
        for path in paths:
            print(f"  {path}")
        return

    # ── Modo: análisis de features ───────────────────────────────────────────
    if args.analyze_features:
        import joblib
        from swmm_resilience.ml.feature_analysis import (
            FEATURE_COLS as ANALYSIS_FEATURE_COLS,
            plot_correlation,
            run_ablation,
            plot_shap,
        )

        try:
            df_analysis = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--analyze-features requires data in {config.dataset.db_path}; "
                f"run the full pipeline first ({load_error})"
            )

        clf_path = MODELS_DIR / "classifier.joblib"
        reg_path = MODELS_DIR / "regressor.joblib"
        if not clf_path.exists() or not reg_path.exists():
            parser.error(
                "--analyze-features requires trained models in outputs/models/; "
                "run python main.py --only-ml first"
            )
        # joblib/pickle is safe here: files are written by this pipeline's own
        # train_models() call and never loaded from an external or untrusted source.
        clf_pipeline = joblib.load(clf_path)
        reg_pipeline = joblib.load(reg_path)
        out_dir = Path("outputs/feature_analysis")

        print("\nFeature analysis...")
        print("  1/3  Correlation...")
        plot_correlation(df_analysis, out_dir)
        print("  2/3  Ablation (LOSO × 2 runs) — may take ~2 min...")
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
        n_expected = len(ANALYSIS_FEATURE_COLS)
        if clf_pipeline.n_features_in_ != n_expected or reg_pipeline.n_features_in_ != n_expected:
            print(
                f"        WARNING: the model in {MODELS_DIR} was trained with "
                f"{clf_pipeline.n_features_in_}/{reg_pipeline.n_features_in_} features "
                f"(clf/reg) but --analyze-features operates on the legacy "
                f"tabular_v3_17 contract ({n_expected} features). The active model does not "
                "match the tabular_v3_17 contract that --analyze-features expects. Train a "
                "compatible model with python main.py --only-ml (first move or delete "
                "classifier.joblib/regressor.joblib in outputs/models/ if needed; "
                "git checkout -- outputs/models/ restores them if you make a mistake). "
                "The SHAP step is skipped."
            )
        else:
            plot_shap(clf_pipeline, reg_pipeline, df_analysis, out_dir)
        print(f"\nResults saved to {out_dir}/")
        return

    # ── Modo: evaluación de formas de hidrograma ─────────────────────────────
    if args.evaluate_shapes:
        try:
            _df = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--evaluate-shapes requires data in {config.dataset.db_path}; "
                f"run the full pipeline first ({load_error})"
            )

        from swmm_resilience.visualization.labels import print_in_sample_banner

        print_in_sample_banner()

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

        base_inflows = _df.groupby("node_id")["base_inflow_lps"].first().to_dict()
        # Only nodes with non-zero inflow are written to validation CSVs
        expected_nodes = {nid for nid, v in base_inflows.items() if v > 0}
        factors_to_eval = config.factors()          # all 25 factors (0.2 → 5.0)
        tmp_val_root = Path(tempfile.mkdtemp(prefix="shape_val_"))

        print(
            f"\nEvaluating {len(shapes)} shapes × {len(factors_to_eval)} factors "
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
                csv_inflow_closure_mode="close",  # forma generada (write_shape_validation_csv): no es CSV de usuario, siempre auto-cierra
                inflow_closure_tolerance_lps=config.validation.inflow_closure_tolerance_lps,
                max_continuity_error_pct=config.validation.max_continuity_error_pct,
            )
            cls = summary["classification"]
            vol = summary["volume"]
            print(
                f"    Scenarios: {summary['n_scenarios']}  "
                f"F1={_fmt_metric(cls.get('f1'))}  "
                f"NSE={_fmt_metric(vol.get('nse'))}  "
                f"Vol_error={_fmt_metric(vol.get('error_pct_total'), '.1f')}%"
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
        try:
            _df = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--evaluate-generalization requires data in {config.dataset.db_path}; "
                f"run the full pipeline first ({load_error})"
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
                csv_inflow_closure_mode="close",  # forma generada (write_shape_validation_csv): no es CSV de usuario, siempre auto-cierra
                inflow_closure_tolerance_lps=config.validation.inflow_closure_tolerance_lps,
                max_continuity_error_pct=config.validation.max_continuity_error_pct,
            )
            cls = summary["classification"]
            vol = summary["volume"]
            print(
                f"    Scenarios: {summary['n_scenarios']}  "
                f"F1={_fmt_metric(cls.get('f1'))}  "
                f"NSE={_fmt_metric(vol.get('nse'))}  "
                f"Vol_error={_fmt_metric(vol.get('error_pct_total'), '.1f')}%"
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

    # ── Modo: validación con formas normalizadas fuera del entrenamiento ─────
    if args.evaluate_validation_shapes:
        shapes_dir = Path(args.evaluate_validation_shapes)
        if not shapes_dir.is_dir():
            parser.error(
                f"--evaluate-validation-shapes: directory {shapes_dir} does not exist"
            )

        from swmm_resilience.simulation.hydrograph_shapes import (
            get_shape_stats, load_all_shapes,
        )
        from swmm_resilience.validation.hydrograph_csv import write_shape_validation_csv
        from swmm_resilience.validation.hydrograph_batch import run_batch_validation

        shapes = load_all_shapes(shapes_dir)
        if not shapes:
            parser.error(
                f"--evaluate-validation-shapes: no shape CSVs in {shapes_dir}"
            )

        # Guard rail: a shape that also lives in the training directory is not
        # out-of-sample, and silently reporting it as such would be a lie.
        training_shapes = (
            load_all_shapes(config.simulation.hydrograph_shapes_dir)
            if config.simulation.hydrograph_shapes_dir is not None
            else {}
        )
        overlap = sorted(set(shapes) & set(training_shapes))
        if overlap:
            parser.error(
                "--evaluate-validation-shapes: these shapes are also in "
                f"{config.simulation.hydrograph_shapes_dir}, they are not out of "
                f"sample: {overlap}"
            )

        # base_inflows comes from the .inp, not the training DB: validating an
        # unseen shape must not require having run --persist-sql first.
        static_df = extract_static_features(config.network.inp_path)
        base_inflows = dict(
            zip(static_df["node_id"].astype(str), static_df["base_inflow_lps"])
        )
        expected_nodes = {nid for nid, value in base_inflows.items() if value > 0}
        if not expected_nodes:
            parser.error(
                f"--evaluate-validation-shapes: no node with inflow > 0 in "
                f"{config.network.inp_path}"
            )

        training_factors = config.factors()
        if args.factors:
            try:
                factors_to_eval = [float(x) for x in args.factors.split(",") if x.strip()]
            except ValueError:
                parser.error(f"--factors: invalid list of numbers: {args.factors!r}")
            if not factors_to_eval:
                parser.error("--factors: the list is empty")
        else:
            factors_to_eval = [
                round((training_factors[i] + training_factors[i + 1]) / 2, 3)
                for i in range(len(training_factors) - 1)
            ]

        # Duration range actually covered by training, so each validation shape
        # can be labelled as interpolation or extrapolation in the report.
        training_durations = [
            get_shape_stats(train_shape)[0] for train_shape in training_shapes.values()
        ]
        dur_min = min(training_durations) if training_durations else 0.0
        dur_max = max(training_durations) if training_durations else 0.0

        tmp_val_root = Path(tempfile.mkdtemp(prefix="val_shapes_"))
        out_root = Path(args.out_dir)

        print()
        print(
            f"Out-of-sample shape validation: {len(shapes)} shapes × "
            f"{len(factors_to_eval)} factors = "
            f"{len(shapes) * len(factors_to_eval)} SWMM scenarios"
        )
        print(f"Factors: {factors_to_eval}")
        print(f"Trained duration range: {dur_min:.2f} h – {dur_max:.2f} h")

        for shape_id, shape in shapes.items():
            dur_h, t_pico_h = get_shape_stats(shape)
            regime = "INTERPOLATION" if dur_min <= dur_h <= dur_max else "EXTRAPOLATION"
            shape_csv_dir = tmp_val_root / shape_id
            for factor in factors_to_eval:
                write_shape_validation_csv(
                    shape_id, shape, base_inflows, factor, shape_csv_dir
                )

            out_dir = out_root / shape_id
            print()
            print(
                f"  [{shape_id}]  dur={dur_h:.2f} h  t_peak={t_pico_h:.2f} h  "
                f"{regime}  → {out_dir}"
            )
            summary = run_batch_validation(
                csv_dir=shape_csv_dir,
                base_inp_path=config.network.inp_path,
                clf_path=MODELS_DIR / "classifier.joblib",
                reg_path=MODELS_DIR / "regressor.joblib",
                flood_threshold_m3=config.dataset.flood_threshold_m3,
                out_dir=out_dir,
                expected_nodes=expected_nodes,
                drain_down_hours=config.validation.drain_down_hours,
                allow_inp_mismatch=args.allow_inp_mismatch,
                factor_range=(config.simulation.factor_min, config.simulation.factor_max),
                csv_inflow_closure_mode="close",  # forma generada (write_shape_validation_csv): no es CSV de usuario, siempre auto-cierra
                inflow_closure_tolerance_lps=config.validation.inflow_closure_tolerance_lps,
                max_continuity_error_pct=config.validation.max_continuity_error_pct,
            )
            cls = summary["classification"]
            vol = summary["volume"]
            print(
                f"    Scenarios: {summary['n_scenarios']}  "
                f"F1={_fmt_metric(cls.get('f1'))}  "
                f"NSE={_fmt_metric(vol.get('nse'))}  "
                f"Vol_error={_fmt_metric(vol.get('error_pct_total'), '.1f')}%"
            )
        shutil.rmtree(tmp_val_root, ignore_errors=True)
        return

    # ── Modo: validación batch de hidrogramas ────────────────────────────────
    if args.evaluate_hydrographs:
        if args.base_inp is None:
            parser.error("--evaluate-hydrographs requires --base-inp PATH")
        if args.clf_path is None:
            parser.error("--evaluate-hydrographs requires --clf-path PATH")
        if args.reg_path is None:
            parser.error("--evaluate-hydrographs requires --reg-path PATH")

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
        try:
            csv_inflow_closure_mode = config.validation.csv_inflow_closure_mode
        except Exception:
            csv_inflow_closure_mode = "strict"
        try:
            inflow_closure_tolerance_lps = config.validation.inflow_closure_tolerance_lps
        except Exception:
            inflow_closure_tolerance_lps = 0.01
        try:
            max_continuity_error_pct = config.validation.max_continuity_error_pct
        except Exception:
            max_continuity_error_pct = 5.0

        from swmm_resilience.validation.hydrograph_batch import run_batch_validation

        print(f"\nValidating hydrographs in '{args.evaluate_hydrographs}'...")
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
            csv_inflow_closure_mode=csv_inflow_closure_mode,
            inflow_closure_tolerance_lps=inflow_closure_tolerance_lps,
            max_continuity_error_pct=max_continuity_error_pct,
        )

        print("\n" + "=" * 50)
        print("HYDROGRAPH VALIDATION SUMMARY")
        print("=" * 50)
        print(f"  Scenarios processed   : {summary['n_scenarios']}")
        print("\n  Classification metrics:")
        for k, v in summary["classification"].items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        if summary.get("pr_auc") is not None:
            print(f"    pr_auc: {summary['pr_auc']:.4f}")
        print("\n  Volume metrics:")
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
            print("\n  Total flooded volume per scenario (SWMM vs ML):")
            for rec in summary["per_scenario"]:
                err_pct = rec.get("error_pct")
                err_txt = f"{err_pct:+.1f}%" if err_pct is not None else "N/A"
                extra = rec.get("n_extrapolated", 0)
                extra_txt = f"  [extrapolated: {extra}]" if extra else ""
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
            print("\n  Compute times per scenario:")
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
                print(f"    Mean speed-up   : x{sum(speedups) / len(speedups):.0f}")

        print(f"\n  Summary CSV        : {summary['summary_csv_path']}")
        print(f"  Totals/scenario    : {summary['scenario_totals_csv_path']}")
        print(f"  Timings            : {summary['timings_csv_path']}")
        print(f"  Metrics/scenario   : {summary['metrics_per_scenario_csv_path']}")
        return

    # ── Modo: persistir en SQLite v17 ───────────────────────────────────────
    if args.persist_sql:
        from swmm_resilience.database.connection import connect_managed_database
        from swmm_resilience.database.migrations import apply_migrations
        from swmm_resilience.database.csv_backfill import (
            backfill_networks_and_runs,
            persist_training_run,
        )

        print(f"\nReading dataset from {config.dataset.output_path}...")
        df = pd.read_csv(config.dataset.output_path)
        if "shape_id" not in df.columns:
            parser.error(
                "dataset_final.csv has no shape_id column — re-run "
                "the full pipeline before --persist-sql"
            )
        if (df["inunda"] == 1).sum() == 0:
            parser.error(
                "dataset_final.csv has no flooded rows — the regressor cannot be trained; "
                "nothing to persist"
            )
        print(f"  {df.shape[0]} rows x {df.shape[1]} cols")

        config.dataset.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = connect_managed_database(config.dataset.db_path)
        apply_migrations(conn)

        print(
            "\nPersisting networks/nodes/scenarios/runs in "
            f"{config.dataset.db_path}..."
        )
        info = backfill_networks_and_runs(
            conn,
            df,
            config.network.inp_path,
            config.network.name,
            source_dataset_path=config.dataset.output_path,
            hydrograph_shapes_dir=config.simulation.hydrograph_shapes_dir,
        )
        print(f"  network_id={info['network_id']}  nodes={len(info['node_pk_by_id'])}  "
              f"runs={len(info['run_id_by_key'])}")

        if args.v4_21_dataset:
            from swmm_resilience.database.csv_backfill import backfill_tributary_features

            v4_21_path = Path(args.v4_21_dataset).expanduser().resolve()
            if not v4_21_path.is_file():
                parser.error(
                    f"The explicit v4_21 dataset does not exist: {v4_21_path}"
                )
            from scripts.migrar_dataset_v4_21 import validate_dataset_manifest

            manifest_path = v4_21_path.with_suffix(v4_21_path.suffix + ".manifest.json")
            try:
                v4_manifest = validate_dataset_manifest(
                    manifest_path,
                    v4_21_path,
                    config.network.inp_path,
                    config.dataset.output_path,
                    config.simulation.hydrograph_shapes_dir,
                )
            except (TypeError, ValueError) as exc:
                parser.error(str(exc))
            df_v4_21 = pd.read_csv(v4_21_path)
            if len(df_v4_21) != v4_manifest["output_dataset"].get("rows"):
                parser.error(
                    f"v4_21 row count does not match its manifest: {v4_21_path}"
                )
            n_v4_21 = backfill_tributary_features(
                conn, df_v4_21, info["run_id_by_key"], info["node_pk_by_id"], info["network_id"],
                source_dataset_path=v4_21_path,
                source_manifest_path=manifest_path,
            )
            print(f"  tabular_v4_21: {n_v4_21} rows of tributary features persisted")

        from swmm_resilience.database.training_queries import load_continuity_excluded

        excluded = load_continuity_excluded(
            conn, config.validation.max_continuity_error_pct
        )
        if not excluded.empty:
            report_path = config.dataset.db_path.parent / "continuity_excluded.csv"
            excluded.to_csv(report_path, index=False)
            print(
                f"  WARNING: {len(excluded)} scenario(s) excluded from "
                f"training and from the main benchmark due to continuity > "
                f"{config.validation.max_continuity_error_pct}% -- "
                f"they remain persisted in the database (visible for curves and "
                f"historical reports); report in {report_path}"
            )

        # El entrenamiento legacy corre sobre el CSV crudo, no sobre SQL, asi que
        # el filtro por continuidad de training_samples_v17 no lo alcanza: se
        # replica aqui sobre `df` con la misma regla que SQL
        # (NULL / columna ausente => no excluido).
        df_training = df
        if "continuity_error_pct" in df.columns:
            continuity = pd.to_numeric(df["continuity_error_pct"], errors="coerce")
            keep = ~(
                continuity.notna()
                & (continuity.abs() > config.validation.max_continuity_error_pct)
            )
            if not bool(keep.all()):
                df_training = df.loc[keep].reset_index(drop=True)
                print(
                    f"  Legacy training: {int((~keep).sum())} of {len(df)} "
                    f"rows discarded due to continuity > "
                    f"{config.validation.max_continuity_error_pct}%"
                )
        if df_training.empty or (df_training["inunda"] == 1).sum() == 0:
            parser.error(
                "after excluding scenarios with continuity error > "
                f"{config.validation.max_continuity_error_pct}% no flooded rows "
                "remain to train the regressor; check continuity_excluded.csv"
            )

        print("\nTraining (GroupKFold5 by factor_mult) and persisting evidence...")
        training_run_id = persist_training_run(
            conn, df_training, info["run_id_by_key"], info["node_pk_by_id"], config
        )
        metrics = conn.execute(
            "SELECT metric_name, value FROM model_metrics WHERE owner_kind = 'model'"
        ).fetchall()
        print(f"  training_run_id={training_run_id} (status=COMPLETE)")
        for name, value in metrics:
            print(f"  {name}: {value:.4f}" if value is not None else f"  {name}: NULL")
        print(
            "\n  Note: this saves training_runs / model_evaluations / oof_predictions / "
            "trained_models with real evidence, but does NOT build the optional "
            "candidates/rankings/promotions system — the models remain valid "
            "historical artifacts, not an 'active selection'."
        )
        conn.close()
        return

    # ── Modo: solo mapas ─────────────────────────────────────────────────────
    if args.only_maps:
        try:
            df_all = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--only-maps requires data in {config.dataset.db_path}; "
                f"run --persist-sql first to populate it ({load_error})"
            )
        df = base_shape_rows(df_all)
        factors_in_dataset = sorted(df["factor_mult"].unique())
        print(f"Generating {len(factors_in_dataset)} SWMM maps...")
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
        print(f"  Maps saved to {config.visualization.output_path}")

        augmented = augmented_shape_rows(df_all)
        if not augmented.empty:
            print("\nGenerating flood maps per hydrograph shape...")
            written = generate_flood_maps_by_shape(
                config.network.inp_path, augmented, config.visualization.output_path,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )
            n_maps = sum(len(paths) for paths in written.values())
            print(f"  {n_maps} maps in {len(written)} folders (one per shape) under "
                  f"{config.visualization.output_path}")
        return

    # ── Pipeline completo (o parcial) ─────────────────────────────────────────
    use_existing_dataset = args.skip_extraction or args.only_ml

    _clf_path = MODELS_DIR / "classifier.joblib"
    _reg_path = MODELS_DIR / "regressor.joblib"
    if _clf_path.exists() or _reg_path.exists():
        import joblib
        from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17
        _existing_clf = joblib.load(_clf_path) if _clf_path.exists() else None
        _existing_reg = joblib.load(_reg_path) if _reg_path.exists() else None
        _n_expected = len(FEATURE_COLUMNS_V17)
        _mismatched = (
            (_existing_clf is not None and _existing_clf.n_features_in_ != _n_expected)
            or (_existing_reg is not None and _existing_reg.n_features_in_ != _n_expected)
        )
        if _mismatched:
            parser.error(
                f"The active model in {MODELS_DIR} has "
                f"{getattr(_existing_clf, 'n_features_in_', 'n/a')}/"
                f"{getattr(_existing_reg, 'n_features_in_', 'n/a')} features (clf/reg), "
                f"not {_n_expected} (the tabular_v3_17 contract that this mode "
                "trains). This check runs BEFORE extraction/simulation so as not to "
                "lose hours of compute: move or delete classifier.joblib/regressor.joblib "
                f"in {MODELS_DIR} manually first if you really want to replace the "
                "active model (`git checkout -- outputs/models/` restores them if you "
                "make a mistake)."
            )

    factors = config.factors()
    n_factors = len(factors)
    static_topo_df = None
    all_node_ids = []
    n_nodes = 0

    if not use_existing_dataset:
        print("\nExtracting static features...")
        static_df = extract_static_features(config.network.inp_path)
        print(f"  {len(static_df)} junction nodes")

        print("Computing topological features...")
        static_topo_df = compute_topology_features(static_df, config.network.inp_path)

        all_node_ids = static_topo_df["node_id"].tolist()
        n_nodes = len(all_node_ids)
        print(f"  Network: {n_nodes} nodes, {n_factors} factors ({factors[0]:.2f}–{factors[-1]:.2f})")

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
            f"\nRunning {n_factors} SWMM simulations (base shape, "
            f"dur={base_dur:.2f}h, t_peak={base_t_pico:.2f}h)..."
        )
        sim_results = run_batch(config, run_dir)

        print("\nExtracting labels and dynamic features...")
        simulation_results = []
        for factor, rpt_path in sim_results:
            dynamic_df = compute_dynamic_features(
                static_topo_df, factor,
                duracion_horas=base_dur,
                tiempo_al_pico_h=base_t_pico,
            )
            dynamic_df["shape_id"] = "base"
            from swmm_resilience.simulation.swmm_api_io import read_continuity_error

            dynamic_df["continuity_error_pct"] = read_continuity_error(rpt_path)

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
                f"\nRunning {len(shapes)} shapes × {n_factors} factors "
                f"= {n_extra} additional SWMM simulations..."
            )
            shape_results = run_batch_shapes(config, shapes, base_inflows, run_dir)
            graph, _ = build_network_graph(_load_inp(config.network.inp_path))
            for shape_id, factor, rpt_path in shape_results:
                dur_h, t_pico_h = get_shape_stats(shapes[shape_id])
                dynamic_df = compute_dynamic_features(
                    static_topo_df, factor,
                    duracion_horas=dur_h,
                    tiempo_al_pico_h=t_pico_h,
                )
                dynamic_df["shape_id"] = shape_id
                from swmm_resilience.simulation.swmm_api_io import read_continuity_error

                dynamic_df["continuity_error_pct"] = read_continuity_error(rpt_path)

                from swmm_resilience.simulation.hydrograph_shapes import apply_shape
                from swmm_resilience.simulation.inflow_closure import apply_closure_policy
                from swmm_resilience.validation.hydrograph_csv import HydrographScenario
                from swmm_resilience.extraction.dynamic_features import (
                    compute_tributary_dynamic_features,
                )

                node_series_for_features = apply_shape(shapes[shape_id], base_inflows, factor)
                time_grid = [t for t, _ in shapes[shape_id]]
                effective_scenario, _ = apply_closure_policy(
                    HydrographScenario(
                        scenario_id=f"{shape_id}_f{factor:.3f}",
                        node_series=node_series_for_features,
                        time_grid_hours=time_grid,
                        last_time_hours=shapes[shape_id][-1][0],
                    ),
                    config.validation.inflow_closure_tolerance_lps,
                )
                capacity_by_node = dict(
                    zip(
                        static_topo_df["node_id"].astype(str),
                        static_topo_df["upstream_capacity_lps"],
                    )
                )
                tributary_df = compute_tributary_dynamic_features(
                    all_node_ids, effective_scenario.node_series, graph, capacity_by_node,
                )
                assert len(tributary_df) == len(all_node_ids)
                # Computado para probar que la función corre contra datos reales
                # del pipeline de entrenamiento (misma función que usa
                # ScenarioPredictor); no se persiste todavía -- dataset_final.csv
                # conserva su esquema actual hasta que el contrato tabular_v4_21
                # se adopte formalmente (Task 6). NO mergear tributary_df en
                # dynamic_df: dynamic_df es justo lo que assemble_dataset()
                # escribe a disco.

                labels_df = extract_labels(
                    rpt_path, all_node_ids, config.dataset.flood_threshold_m3
                )
                simulation_results.append((factor, dynamic_df, labels_df))

        n_simulations = len(simulation_results)
        print("\nAssembling dataset...")
        df = assemble_dataset(static_topo_df, simulation_results, config.dataset.output_path)
        print("Validating dataset...")
        validate_dataset(df, n_nodes, n_simulations)
        print(f"  Dataset validated: {df.shape}")

        # XGBoost's `subsample` (config.yaml ml.classifier/regressor) draws a
        # random row fraction per tree, so which rows land in that sample
        # depends on the order rows are presented in — not just their values.
        # assemble_dataset() keeps CSV/assembly order (node_id-major) so the
        # CSV on disk stays unperturbed for --persist-sql and other readers,
        # but load_training_frame() (the --only-ml/--skip-extraction path)
        # returns rows `ORDER BY run_id, node_id`, equivalent to sorting by
        # (shape_id, factor_mult, node_id). Without this sort, a cold run and
        # a SQL-fed run over identical data would train on different row
        # orders and disagree on metrics. Sort only this in-memory frame —
        # the one handed to training/evaluation below — to match.
        df = df.sort_values(["shape_id", "factor_mult", "node_id"]).reset_index(drop=True)
    else:
        print(f"\nReading dataset from {config.dataset.db_path}...")
        try:
            df = load_training_frame(config.dataset.db_path)
        except ValueError as load_error:
            parser.error(
                f"--only-ml/--skip-extraction require data in "
                f"{config.dataset.db_path}; run --persist-sql first "
                f"to populate it ({load_error})"
            )
        print(f"  {df.shape[0]} rows × {df.shape[1]} cols")

    print("\nTraining final models...")
    clf, reg = train_models(df, config, MODELS_DIR)

    print("\nEvaluating models...")
    results = evaluate_models(df, config, METRICS_DIR)

    print("\nGenerating feature importance plots...")
    generate_feature_importance_plots(clf, reg, METRICS_DIR)

    if not args.only_ml:
        print("\nGenerating flood maps (base shape)...")
        df_base = base_shape_rows(df)
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

        augmented = augmented_shape_rows(df)
        if not augmented.empty:
            print("\nGenerating flood maps per hydrograph shape...")
            written = generate_flood_maps_by_shape(
                config.network.inp_path, augmented, config.visualization.output_path,
                config.network.name, config.visualization.colormap,
                config.visualization.show_labels_top_n,
            )
            n_maps = sum(len(paths) for paths in written.values())
            print(f"  {n_maps} maps in {len(written)} folders (one per shape) under "
                  f"{config.visualization.output_path}")

    print("\n" + "=" * 50)
    print("METRICS SUMMARY (LOSO)")
    print("=" * 50)
    loso = results.get("LOSO", {})
    labels_map = {
        "classifier": "Classifier",
        "regressor_oracle": "Regressor (oracle — true labels used for filtering)",
        "end_to_end": "End-to-end system",
    }
    for level, label in labels_map.items():
        m = loso.get(level, {})
        if m:
            print(f"\n{label}:")
            for k, v in m.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
